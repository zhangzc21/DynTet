import os
import argparse

import torch
from tqdm import tqdm
import pathlib

import numpy as np
import pickle as pkl
# from models.arcface.id_loss import IDLoss
import cv2

import face_alignment #https://github.com/1adrianb/face-alignment
import lpips
from evaluate_utils.metrics import PSNRMeter, LPIPSMeter, LMDMeter, FIDMeter, ArcfaceMeter
from rich.console import Console

console = Console()

device = torch.device('cuda')

def mse_to_psnr(mse):
  """Compute PSNR given an MSE (we assume the maximum pixel value is 1)."""
  return -10. / np.log(10.) * np.log(mse)

def read_image(path, range = '-1,1', type = 'tensor', size = None):
    assert range in ['-1,1', '0,1', '0,255']
    assert type in ['array', 'tensor']

    img = np.ascontiguousarray(cv2.imread(path)[..., ::-1])

    if size is not None:
        img = cv2.resize(img, size)

    if range == '0,1':
        img = img / 255
    elif range == '-1,1':
        img = (img - 127.5) / 127.5

    if type == 'array':
        return img
    elif type == 'tensor':
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()

    return img

@torch.no_grad()
def calc_metric(G_source_paths, G_target_paths):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda', face_detector='sfd')
    LPIPS = lpips.LPIPS(net='vgg').cuda().eval()
    # G_target_paths = G_source_paths
    resize256 = torch.nn.AdaptiveAvgPool2d((256, 256))
    pbar = tqdm(zip(G_source_paths, G_target_paths), total = len(G_target_paths))

    array_to_tensor = lambda x: ((torch.from_numpy(x) - 127.5) / 127.5).permute(2,0,1).unsqueeze(0)

    lmk_NMEs= []
    LPIPS_v = []
    PSNRs = []

    for i, (source_path, target_path) in tqdm(enumerate(pbar)):
        print(source_path, target_path)
        x = read_image(source_path, type = 'array', range = '0,255', size = (512, 512))
        y = read_image(target_path, type = 'array', range = '0,255', size = (512, 512))
        h,w = x.shape[0],x.shape[1]
        ### landmarks Normalized Mean Error ####
        x_landmarks = fa.get_landmarks_from_image(x)
        y_landmarks = fa.get_landmarks_from_image(y)

        if y_landmarks is not None and len(y_landmarks) > 0:
            lmk_NME = np.linalg.norm(x_landmarks[0] - y_landmarks[0], ord = 2) / h
            lmk_NMEs.append(lmk_NME)
        else:
            lmk_NME = '100'
        ### ID similarity #####
        l = LPIPS(array_to_tensor(x).cuda(), array_to_tensor(y).cuda(), normalize=False).mean().item()

        mse = torch.nn.functional.mse_loss(array_to_tensor((x+1)/2).cuda(), array_to_tensor((y+1)/2).cuda(), size_average=None, reduce=None, reduction='mean').item()
        psnr = mse_to_psnr(mse).item()

        LPIPS_v.append(l)
        PSNRs.append(psnr)
        # x = array_to_tensor(x).to(device).to(torch.float)
        # y = array_to_tensor(y).to(device).to(torch.float)
        pbar.set_postfix(lmk_NME = float(lmk_NME))
    return lmk_NMEs, LPIPS_v, PSNRs


@torch.no_grad()
def evaluate(G_source_paths, G_target_paths, report_fid = False):
    metrics = [
               PSNRMeter(),
               LPIPSMeter(device=device),
               LMDMeter(backend='fan'),
               LMDMeter(backend='fan', region = 'eye'),
               ArcfaceMeter(device = device)
    ]
    pbar = tqdm(zip(G_source_paths, G_target_paths), total = len(G_target_paths))
    for i, (source_path, target_path) in tqdm(enumerate(pbar)):
        x = read_image(source_path, type='tensor', range='0,1').permute(0,2,3,1)
        y = read_image(target_path, type='tensor', range='0,1').permute(0,2,3,1)
        for metric in metrics:
            metric.update(x, y)

    if report_fid:
        fid = FIDMeter()
        metrics.append(fid)
        fid(G_source_paths, G_target_paths)


    for metric in metrics:
        out = metric.report()
        console.print(out, style="red")

        key, value = out.split('=')
        save_dict[key].append(float(value))


if __name__ == '__main__':
    import natsort
    import glob
    import pandas as pd
    import collections
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default="out/obama")
    args = parser.parse_args()
    train_dir = args.train_dir

    save_dict = collections.defaultdict(list)
    keyword = os.path.basename(train_dir)

    print(f'<==========start========>')
    console.print(train_dir, style="blue")

    image_dir = os.path.join(train_dir, 'test')

    image_list = natsort.natsorted(glob.glob(os.path.join(image_dir, '*_opt.png')))
    image_list2 = natsort.natsorted(glob.glob(os.path.join(image_dir, '*_ref.png')))

    min_len = min(len(image_list), len(image_list2))

    image_list = image_list[:min_len]
    image_list2 = image_list2[:min_len]

    evaluate(image_list, image_list2, report_fid = False)
    save_dict['name'].append(keyword)
    print(f'<==========finish========>')

    df = pd.DataFrame(save_dict)
    df.to_excel(f'out/{keyword}.xlsx')
