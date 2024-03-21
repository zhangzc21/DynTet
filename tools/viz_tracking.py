import os
import cv2
import numpy as np
import torch
import sys
sys.path.append(os.path.abspath('data_utils/face_tracking'))

from facemodel import Face_3DMM
from util import *
from render_3dmm import Render_3DMM
import json
from tqdm import tqdm, trange
from torch.utils.data import Dataset

import time
import face_alignment
import open3d as o3d
from common_utils import Timer


def load_json(p):
    with open(p, 'r') as f:
        return json.load(f)


def _load_pt(pt):
    params = torch.load(pt)
    id = params['id']
    exp = params['exp'][Selected]
    euler = params['euler'][Selected]
    trans = params['trans'][Selected]
    tex = params['tex']
    light = params['light'][Selected]
    focal = params['focal']
    return id, exp, euler, trans, tex, light, focal


def _load_data_from_info_json(info_json):
    _input_info = load_json(info_json)

    img_path_list = [os.path.join(os.path.dirname(info_json), "ori_imgs", str(frame['img_id']) + '.jpg') for frame in
                     _input_info['frames']]
    lms_path_list = [os.path.join(os.path.dirname(info_json), "ori_imgs", str(frame['img_id']) + '.lms') for frame in
                     _input_info['frames']]

    total_num = len(img_path_list)
    lmss = [np.loadtxt(p, dtype = np.float32) for p in lms_path_list]
    lmss = np.stack(lmss)
    lmss = torch.as_tensor(lmss).cuda()
    h, w, _ = cv2.imread(img_path_list[0]).shape
    cxy = torch.tensor((w / 2.0, h / 2.0), dtype = torch.float).cuda()
    return lmss, img_path_list, total_num, h, w, cxy


def write_video(imgs, save_path, fps = 25):
    os.makedirs(os.path.dirname(save_path), exist_ok = True)
    video_width = imgs[0].shape[1]
    video_height = imgs[0].shape[0]
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    videoWriter = cv2.VideoWriter(save_path, fourcc, fps, (video_width, video_height), True)
    for img in tqdm(imgs, desc = 'write video'):
        videoWriter.write(img)
    videoWriter.release()
    print('video path', save_path)


class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = cv2.imread(image_path)
        return image


def collate_fn(batch):
    images = np.stack(batch)
    return images


Selected = slice(0, 512, 1)


def to_device(device, *args):
    args = list(args)
    for i, _ in enumerate(args):
        args[i] = args[i].to(device)
    return args


@torch.no_grad()
def visualize_tracking(pt_file = "/data2/zzc/nvdiffrec/data/video/may_test",
                       device = torch.device('cuda:0'), batch_size = 1):
    fine_pt = os.path.join(pt_file, "track_params.pt")
    save_dir = os.path.join('tools/viz_tracking', os.path.basename(pt_file))
    os.makedirs(save_dir, exist_ok = True)

    id_para, exp_para, euler, trans, tex, light, focal_length = to_device(device, *_load_pt(fine_pt))
    total_num = len(exp_para)
    id_para = id_para.expand(total_num, -1).detach()
    tex = tex.expand(total_num, -1).detach()

    lms1, img_paths1, total_num1, h, w, cxy = _load_data_from_info_json(
        fine_pt.replace("track_params.pt", "transforms_train.json"))
    lms2, img_paths2, total_num2, h, w, cxy = _load_data_from_info_json(
        fine_pt.replace("track_params.pt", "transforms_val.json"))

    lms = torch.cat([lms1, lms2], dim = 0)
    img_paths = img_paths1 + img_paths2

    # sel_tex_para = texture.expand(total_num, -1).detach()
    model_3dmm = Face_3DMM(os.path.join('data_utils/face_tracking', "3DMM"), 100, 79, 100, 34650)
    model_3dmm.to(device)
    renderer = Render_3DMM(float(focal_length), h, w, batch_size, device)
    tris = renderer.tris.detach().cpu().numpy()

    render_imgs = []
    masks = []
    lm3d = []

    # lms_numpy = lms.cpu().numpy()

    # dataset = ImageDataset(img_paths)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size = len(img_paths), shuffle = False, collate_fn = collate_fn,
    #                                          num_workers = 8)
    os.makedirs('tools/viz_tracking', exist_ok = True)
    for i in tqdm(range(int((total_num - 1) / batch_size + 1))):
        if (i + 1) * batch_size > total_num:
            start_n = total_num - batch_size
            end_n = total_num
        else:
            start_n = i * batch_size
            end_n = i * batch_size + batch_size
        sel_ids = slice(start_n, end_n)

        sel_id_para_, sel_exp_para_, sel_trans_, sel_euler_, sel_texture_, sel_light = id_para[sel_ids], \
            exp_para[sel_ids], trans[sel_ids], euler[sel_ids], tex[sel_ids], light[sel_ids]
        sel_lms_dect, sel_img_paths = lms[sel_ids].cpu().numpy(), img_paths[sel_ids]
        real_imgs = cv2.imread(sel_img_paths[0])

        lmk3d = model_3dmm.get_3dlandmarks(
            sel_id_para_, sel_exp_para_, sel_euler_, sel_trans_, focal_length, cxy
        )
        proj_lmk3d = forward_transform(lmk3d, sel_euler_, sel_trans_, focal_length, cxy)
        lm2d = proj_lmk3d[:, :, :2].detach().cpu().numpy()

        geometry = model_3dmm.forward_geo(sel_id_para_, sel_exp_para_)
        rott_geo = forward_rott(geometry, sel_euler_, sel_trans_)
        sel_texture = model_3dmm.forward_tex(sel_texture_)
        render_imgs_ = renderer(rott_geo, torch.ones_like(sel_texture)*200, sel_light)

        masks_ = 0.3 * ((render_imgs_[0, :, :, -1:]).detach() > 0.0).cpu().numpy()
        render_imgs_ = render_imgs_[0,...,:3].detach().cpu().numpy()
        render_imgs_ = np.ascontiguousarray(render_imgs_[...,::-1])
        render_imgs_ = real_imgs * (1 - masks_) + render_imgs_ * (masks_)

        for point in lm2d[0]:
            cv2.circle(render_imgs_, tuple(np.rint(point).astype(np.int).tolist()), 3,
                           (0, 0, 255), 1)

        for point in sel_lms_dect[0]:
            cv2.circle(render_imgs_, tuple(np.rint(point).astype(np.int).tolist()), 3,
                           (0, 255, 0), 1)

        cv2.imwrite(os.path.join(save_dir, str(i).zfill(6) + '.png'), np.clip(render_imgs_, 0, 255).astype(np.uint8))

        #

        #
        # sel_geometry = model_3dmm.forward_geo(sel_id_para_, sel_exp_para_)
        # sel_texture = model_3dmm.forward_tex(sel_tex_para_)

        # mesh = o3d.geometry.TriangleMesh()
        # mesh.vertices = o3d.utility.Vector3dVector(geometry[0].detach().cpu().numpy().reshape(-1, 3))
        # mesh.triangles = o3d.utility.Vector3iVector(tris.reshape(-1, 3))
        # o3d.io.write_triangle_mesh(os.path.join(os.path.dirname(fine_pt), f'geometry_frame_{i}_open.obj'), mesh)

        # break


if __name__ == '__main__':
    dir = "/data2/zzc/HDTF/processed/WRA_DavidVitter_000"
    visualize_tracking(dir)
    save_dir = os.path.join('tools/viz_tracking', os.path.basename(dir))
    os.system(f"ffmpeg -y -r 25 -pattern_type glob -i '{save_dir}/*.png' -c:v libx264 -pix_fmt yuv420p {save_dir}/output.mp4")
