import argparse
import json
import os
from functools import partial
import seaborn as sns
import cv2.dnn
import numpy as np
import yaml
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def eval_psnr(loader, class_names, model,
              data_norm=None, eval_type=None, save_fig=False, save_featmap=False,
              scale_ratio=1, save_path=None, verbose=False, crop_border=4,
              cal_metrics=True,
              ):
    crop_border = int(crop_border) if crop_border else crop_border
    print('crop border: ', crop_border)
    model.eval()

    if data_norm is None:
        data_norm = {
            'img': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['img']
    img_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).to(device)
    img_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).to(device)
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).to(device)
    gt_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).to(device)

    if eval_type is None:
        metric_fn = [utils.calculate_psnr_pt, utils.calculate_ssim_pt]
    elif eval_type == 'psnr+ssim':
        metric_fn = [utils.calculate_psnr_pt, utils.calculate_ssim_pt]
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError

    val_res_psnr = utils.Averager(class_names)
    val_res_ssim = utils.Averager(class_names)

    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device)

        img = (batch['img'] - img_sub) / img_div
        with torch.no_grad():
            preds = model(img, batch['gt'].shape[-2:])
        if save_featmap:
            pred = preds[0][-1]
            returned_featmap = preds[1]
            assert returned_featmap.size(1) == 6
        else:
            if isinstance(preds, list):
                pred = preds[-1]
        # import pdb
        # pdb.set_trace()
        pred = pred * gt_div + gt_sub
        # if eval_type is not None:  # reshape for shaving-eval
        #     ih, iw = batch['img'].shape[-2:]
        #     s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
        #     if s > 1:
        #         shape = [batch['img'].shape[0], round(ih * s), round(iw * s), 3]
        #     else:
        #         shape = [batch['img'].shape[0], 32, batch['coord'].shape[1]//32, 3]
        #
        #     pred = pred.view(*shape) \
        #         .permute(0, 3, 1, 2).contiguous()
        #     batch['gt'] = batch['gt'].view(*shape) \
        #         .permute(0, 3, 1, 2).contiguous()

        # if crop_border is not None:
        #     h = math.sqrt(pred.shape[1])
        #     shape = [img.shape[0], round(h), round(h), 3]
        #     pred = pred.view(*shape).permute(0, 3, 1, 2).contiguous()
        #     batch['gt'] = batch['gt'].view(*shape).permute(0, 3, 1, 2).contiguous()
        # else:
        #     pred = pred.permute(0, 2, 1).contiguous()  # B 3 N
        #     batch['gt'] = batch['gt'].permute(0, 2, 1).contiguous()

        # import pdb
        # pdb.set_trace()

        if cal_metrics:
            res_psnr = metric_fn[0](
                pred,
                batch['gt'],
                crop_border=crop_border
            )
            res_ssim = metric_fn[1](
                pred,
                batch['gt'],
                crop_border=crop_border
            )
        else:
            res_psnr = torch.ones(len(pred))
            res_ssim = torch.ones(len(pred))

        file_names = batch.get('filename', None)
        if file_names is not None and save_featmap:
            for idx in range(len(batch['img'])):
                ori_img = batch['img'][idx].cpu().numpy() * 255
                ori_img = np.clip(ori_img, a_min=0, a_max=255)
                ori_img = ori_img.astype(np.uint8)
                ori_img = rearrange(ori_img, 'C H W -> H W C')

                pred_img = pred[idx].cpu().numpy() * 255
                pred_img = np.clip(pred_img, a_min=0, a_max=255)
                pred_img = pred_img.astype(np.uint8)
                pred_img = rearrange(pred_img, 'C H W -> H W C')

                is_normalize = True
                f_tensors = returned_featmap[idx]
                for idx_f in range(len(f_tensors)):
                    f_tensor = f_tensors[idx_f]
                    if is_normalize:
                        # normalize the features / feature maps
                        f_tensor = torch.sigmoid(f_tensor)
                    f_tensor = f_tensor.detach().cpu().numpy()
                    # for better visualization, you can normalize the feature heatmap
                    f_tensor = (f_tensor - np.min(f_tensor)) / (np.max(f_tensor) - np.min(f_tensor))
                    # f_tensor = (f_tensor - np.min(f_tensor)) / (np.max(f_tensor) - np.min(f_tensor))
                    sns.heatmap(f_tensor, vmin=0, vmax=1, cmap="jet", center=0.5)
                    plt.axis('off')
                    plt.xticks([])
                    plt.yticks([])
                    # plt.imshow(heatmap, cmap='YlGnBu', vmin=0, vmax=1)
                    # plt.show()
                    ori_file_name = f'{save_path}/{file_names[idx]}_{idx_f}.png'
                    plt.savefig(ori_file_name, dpi=600)
                    plt.close()

                gt_img = batch['gt'][idx].cpu().numpy() * 255
                gt_img = np.clip(gt_img, a_min=0, a_max=255)
                gt_img = gt_img.astype(np.uint8)
                gt_img = rearrange(gt_img, 'C H W -> H W C')

                psnr = res_psnr[idx].cpu().numpy()
                ssim = res_ssim[idx].cpu().numpy()
                ori_file_name = f'{save_path}/{file_names[idx]}_Ori.png'
                cv2.imwrite(ori_file_name, ori_img)
                pred_file_name = f'{save_path}/{file_names[idx]}_{scale_ratio}X_{psnr:.2f}_{ssim:.4f}.png'
                cv2.imwrite(pred_file_name, pred_img)
                gt_file_name = f'{save_path}/{file_names[idx]}_GT.png'
                cv2.imwrite(gt_file_name, gt_img)
                # import pdb
                # pdb.set_trace()

        if file_names is not None and save_fig:
            for idx in range(len(batch['img'])):
                ori_img = batch['img'][idx].cpu().numpy() * 255
                ori_img = np.clip(ori_img, a_min=0, a_max=255)
                ori_img = ori_img.astype(np.uint8)
                ori_img = rearrange(ori_img, 'C H W -> H W C')

                pred_img = pred[idx].cpu().numpy() * 255
                pred_img = np.clip(pred_img, a_min=0, a_max=255)
                pred_img = pred_img.astype(np.uint8)
                pred_img = rearrange(pred_img, 'C H W -> H W C')

                gt_img = batch['gt'][idx].cpu().numpy() * 255
                gt_img = np.clip(gt_img, a_min=0, a_max=255)
                gt_img = gt_img.astype(np.uint8)
                gt_img = rearrange(gt_img, 'C H W -> H W C')

                psnr = res_psnr[idx].cpu().numpy()
                ssim = res_ssim[idx].cpu().numpy()
                ori_file_name = f'{save_path}/{file_names[idx]}_Ori.png'
                cv2.imwrite(ori_file_name, ori_img)
                pred_file_name = f'{save_path}/{file_names[idx]}_{scale_ratio}X_{psnr:.2f}_{ssim:.4f}.png'
                cv2.imwrite(pred_file_name, pred_img)
                gt_file_name = f'{save_path}/{file_names[idx]}_GT.png'
                cv2.imwrite(gt_file_name, gt_img)

        val_res_psnr.add(batch['class_name'], res_psnr)
        val_res_ssim.add(batch['class_name'], res_ssim)

        if verbose:
            pbar.set_description(
                'val psnr: {:.4f} ssim: {:.4f}'.format(val_res_psnr.item()['all'], val_res_ssim.item()['all']))

    return val_res_psnr.item(), val_res_ssim.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/test_INR_diinn_arbrcan_funsr_overnet.yaml')
    parser.add_argument('--model', default='pretrain/UC_FunSR_RDN.pth')
    parser.add_argument('--scale_ratio', default=4, type=float)
    parser.add_argument('--save_fig', default=True, type=bool)
    parser.add_argument('--save_featmap', default=False, type=bool)
    parser.add_argument('--save_path', default='tmp', type=str)
    parser.add_argument('--cal_metrics', default=True, type=bool)
    parser.add_argument('--return_class_metrics', default=False, type=bool)
    parser.add_argument('--dataset_name', default='UC', type=str)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    root_split_file = {
        'UC':
        {
            'root_path': 'samples/UCMerced',
            'split_file': 'samples/uc_split.json'
        },
        'AID':
            {
                'root_path': 'samples/AID',
                'split_file': 'data_split/AID_split.json'
            }
    }
    config['test_dataset']['dataset']['args']['root_path'] = root_split_file[args.dataset_name]['root_path']
    config['test_dataset']['dataset']['args']['split_file'] = root_split_file[args.dataset_name]['split_file']

    config['test_dataset']['wrapper']['args']['scale_ratio'] = args.scale_ratio

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'], num_workers=0, pin_memory=True, shuffle=False, drop_last=False)
    if not os.path.exists(args.model):
        assert NameError
    model_spec = torch.load(args.model, map_location='cpu')['model']
    print(model_spec['args'])
    model = models.make(model_spec, load_sd=True).to(device)

    file_names = json.load(open(config['test_dataset']['dataset']['args']['split_file']))['test']
    class_names = list(set([os.path.basename(os.path.dirname(x)) for x in file_names]))

    crop_border = config['test_dataset']['wrapper']['args']['scale_ratio'] + 5
    dataset_name = os.path.basename(config['test_dataset']['dataset']['args']['split_file']).split('_')[0].lower()
    max_scale = {'uc': 5, 'aid': 12}
    if args.scale_ratio > max_scale[dataset_name]:
        crop_border = int((args.scale_ratio - max_scale[dataset_name]) / 2 * 48)

    if args.save_fig or args.save_featmap:
        os.makedirs(args.save_path, exist_ok=True)

    res = eval_psnr(
        loader, class_names, model,
        data_norm=config.get('data_norm'),
        eval_type=config.get('eval_type'),
        crop_border=crop_border,
        verbose=True,
        save_fig=args.save_fig,
        save_featmap=args.save_featmap,
        scale_ratio=args.scale_ratio,
        save_path=args.save_path,
        cal_metrics=args.cal_metrics
    )

    if args.return_class_metrics:
        keys = list(res[0].keys())
        keys.sort()
        print('psnr')
        for k in keys:
            print(f'{k}: {res[0][k]:0.2f}')
        print('ssim')
        for k in keys:
            print(f'{k}: {res[1][k]:0.4f}')
    print(f'psnr: {res[0]["all"]:0.2f}')
    print(f'ssim: {res[1]["all"]:0.4f}')
