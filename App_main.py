import numpy as np
import os
os.system('nvidia-smi')
os.system('ls /usr/local')
os.system('pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116')
os.system('pip install -U openmim')
os.system('mim install mmcv-full')
import models
import gradio as gr

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def construct_sample(img, mean=0.5, std=0.5):
    img = transforms.ToTensor()(img)
    img = transforms.Resize(48, InterpolationMode.BICUBIC)(img)
    img = transforms.Normalize(mean, std)(img)
    return img

def build_model(cp):
    model_spec = torch.load(cp, map_location='cpu')['model']
    print(model_spec['args'])
    model = models.make(model_spec, load_sd=True).to(device)
    return model


# Function for building extraction
def sr_func(img, cp, scale):
    if cp == 'UC':
        checkpoint = 'pretrain/UC_FunSR_RDN.pth'
    elif cp == 'AID':
        checkpoint = 'pretrain/AID_FunSR_RDN.pth'
    else:
        raise NotImplementedError
    sample = construct_sample(img)
    print('Use: ', device)
    model = build_model(checkpoint)
    model.eval()
    sample = sample.to(device)
    sample = sample.unsqueeze(0)

    ori_size = torch.tensor(sample.shape[2:])  # BCHW
    target_size = ori_size * scale
    target_size = target_size.long()
    lr_target_size_img = torch.nn.functional.interpolate(sample, scale_factor=scale, mode='nearest')
    with torch.no_grad():
        pred = model(sample, target_size.tolist())

    if isinstance(pred, list):
        pred = pred[-1]
    pred = pred * 0.5 + 0.5

    pred *= 255
    pred = pred[0].detach().cpu()
    lr_target_size_img = lr_target_size_img * 0.5 + 0.5
    lr_target_size_img = 255 * lr_target_size_img[0].detach().cpu()

    lr_target_size_img = torch.clamp(lr_target_size_img, 0, 255).permute(1,2,0).numpy().astype(np.uint8)
    pred = torch.clamp(pred, 0, 255).permute(1,2,0).numpy().astype(np.uint8)

    line = np.ones((pred.shape[0], 5, 3), dtype=np.uint8) * 255
    pred = np.concatenate((lr_target_size_img, line, pred), axis=1)
    return pred

title = "FunSR"
description = "Gradio demo for continuous remote sensing image super-resolution. Upload image from UCMerced or AID Dataset or click any one of the examples, " \
              "Then change the upscaling magnification, and click \"Submit\" and wait for the super-resolved result. \n" \
              "Paper: Continuous Remote Sensing Image Super-Resolution based on Context Interaction in Implicit Function Space"

article = "<p style='text-align: center'><a href='https://kyanchen.github.io/FunSR/' target='_blank'>FunSR Project " \
          "Page</a></p> "

default_scale = 4.0
examples = [
    ['examples/AID_school_161_LR.png', 'AID', default_scale],
    ['examples/AID_bridge_19_LR.png', 'AID', default_scale],
    ['examples/AID_parking_60_LR.png', 'AID', default_scale],
    ['examples/AID_commercial_32_LR.png', 'AID', default_scale],

    ['examples/UC_airplane95_LR.png', 'UC', default_scale],
    ['examples/UC_freeway35_LR.png', 'UC', default_scale],
    ['examples/UC_storagetanks54_LR.png', 'UC', default_scale],
    ['examples/UC_airplane00_LR.png', 'UC', default_scale],
]

with gr.Blocks() as demo:
    image_input = gr.inputs.Image(type='pil', label='Input Img')
    # with gr.Row().style(equal_height=True):
    # image_LR_output = gr.outputs.Image(label='LR Img', type='numpy')
    image_output = gr.outputs.Image(label='SR Result', type='numpy')
    with gr.Row():
        checkpoint = gr.inputs.Radio(['UC', 'AID'], label='Checkpoint')
        scale = gr.Slider(1, 12, value=4.0, step=0.1, label='scale')

io = gr.Interface(fn=sr_func,
                  inputs=[image_input,
                          checkpoint,
                          scale
                          ],
                  outputs=[
                      # image_LR_output,
                      image_output
                  ],
                  title=title,
                  description=description,
                  article=article,
                  allow_flagging='auto',
                  examples=examples,
                  cache_examples=True,
                  layout="grid"
                  )
io.launch()
