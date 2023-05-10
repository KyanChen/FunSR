import argparse
import json
import os
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR

import datasets
import models
import utils
from test_inr_diinn_arbrcan_sadnarc_funsr_overnet import eval_psnr

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    log('{} dataset: size={}'.format(tag, len(dataset)))
    for k, v in dataset[0].items():
        if torch.is_tensor(v):
            log('  {}: shape={}'.format(k, v.shape))
        elif isinstance(v, str):
            pass
        elif isinstance(v, dict):
            for k0, v0 in v.items():
                if hasattr(v0, 'shape'):
                    log('  {}: shape={}'.format(k0, v0.shape))
        else:
            raise NotImplementedError

    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=(tag == 'train'), num_workers=spec['num_workers'], pin_memory=True)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader


def prepare_training():
    if config.get('resume') is not None:
        sv_file = torch.load(config['resume'])
        model = models.make(sv_file['model'], load_sd=True).to(device)
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=False)
        # epoch_start = sv_file['epoch'] + 1
        epoch_start = 1

        lr_scheduler = config.get('lr_scheduler')
        lr_scheduler_name = lr_scheduler.pop('name')
        if 'MultiStepLR' == lr_scheduler_name:
            lr_scheduler = MultiStepLR(optimizer, **lr_scheduler)
        elif 'CosineAnnealingLR' == lr_scheduler_name:
            lr_scheduler = CosineAnnealingLR(optimizer, **lr_scheduler)
        elif 'CosineAnnealingWarmUpLR' == lr_scheduler_name:
            lr_scheduler = utils.warm_up_cosine_lr_scheduler(optimizer, **lr_scheduler)
        # for _ in range(epoch_start - 1):
        #     lr_scheduler.step()
    else:
        model = models.make(config['model']).to(device)
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
        lr_scheduler = config.get('lr_scheduler')
        lr_scheduler_name = lr_scheduler.pop('name')
        if 'MultiStepLR' == lr_scheduler_name:
            lr_scheduler = MultiStepLR(optimizer, **lr_scheduler)
        elif 'CosineAnnealingLR' == lr_scheduler_name:
            lr_scheduler = CosineAnnealingLR(optimizer, **lr_scheduler)
        elif 'CosineAnnealingWarmUpLR' == lr_scheduler_name:
            lr_scheduler = utils.warm_up_cosine_lr_scheduler(optimizer, **lr_scheduler)

    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler


def train(train_loader, model, optimizer):
    model.train()
    loss_fn = nn.L1Loss()
    train_loss = utils.AveragerList()

    data_norm = config['data_norm']
    t = data_norm['img']
    img_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).to(device)
    img_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).to(device)
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).to(device)
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).to(device)

    for batch in tqdm(train_loader, leave=False, desc='train'):
        # import pdb
        # pdb.set_trace()
        keys = list(batch.keys())
        batch = batch[keys[torch.randint(0, len(keys), [])]]
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device)

        img = (batch['img'] - img_sub) / img_div
        gt = (batch['gt'] - gt_sub) / gt_div
        pred = model(img, gt.shape[-2:])
        if isinstance(pred, tuple):
            loss = 0.2 * loss_fn(pred[0], gt) + loss_fn(pred[1], gt)
        elif isinstance(pred, list):
            losses = [loss_fn(x, gt) for x in pred]
            losses = [x * (idx + 1) for idx, x in enumerate(losses)]
            loss = sum(losses) / ((1 + len(losses)) * len(losses) / 2)
        else:
            loss = loss_fn(pred, gt)

        train_loss.add(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_loss.item()


def main(config_, save_path, args):
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'img': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    n_gpus = args.n_gpus
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

    epoch_max = config['epoch_max']
    epoch_val_interval = config.get('epoch_val_interval')
    epoch_save_interval = config.get('epoch_save_interval')
    max_val_v = -1e18

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        train_loss = train(train_loader, model, optimizer)
        if lr_scheduler is not None:
            lr_scheduler.step()

        log_info.append('train: loss={:.4f}'.format(train_loss))
        writer.add_scalars('loss', {'train': train_loss}, epoch)

        if device != 'cpu' and n_gpus > 1:
            model_ = model.module
        else:
            model_ = model
        model_spec = config['model']
        model_spec['sd'] = model_.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }

        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))

        if (epoch_save_interval is not None) and (epoch % epoch_save_interval == 0):
            torch.save(sv_file, os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if (epoch_val_interval is not None) and (epoch % epoch_val_interval == 0):
            if device != 'cpu' and n_gpus > 1 and (config.get('eval_bsize') is not None):
                model_ = model.module
            else:
                model_ = model

            file_names = json.load(open(config['val_dataset']['dataset']['args']['split_file']))['test']
            class_names = list(set([os.path.basename(os.path.dirname(x)) for x in file_names]))

            val_res_psnr, val_res_ssim = eval_psnr(val_loader, class_names, model_,
                                                   data_norm=config['data_norm'],
                                                   eval_type=config.get('eval_type'),
                                                   eval_bsize=config.get('eval_bsize'),
                                                   crop_border=4)

            log_info.append('val: psnr={:.4f}'.format(val_res_psnr['all']))
            writer.add_scalars('psnr', {'val': val_res_psnr['all']}, epoch)
            if val_res_psnr['all'] > max_val_v:
                max_val_v = val_res_psnr['all']
                torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train_1x-5x_INR_funsr.yaml')
    parser.add_argument('--n_gpus', default=1, type=int)
    parser.add_argument('--save_name', default='funsr')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.save_name
    save_path = os.path.join('./checkpoints', save_name)

    main(config, save_path, args)
