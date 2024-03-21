import os
import cv2
import numpy as np
import torch
import sys

from face_tracking.facemodel import Face_3DMM
from face_tracking.util import *
from face_tracking.render_3dmm import Render_3DMM
import json
from tqdm import tqdm, trange
from torch.utils.data import Dataset

import time
import face_alignment
import open3d as o3d
from tools.common_utils import Timer


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


Selected = slice(0, 2048, 1)


def to_device(device, *args):
    args = list(args)
    for i, _ in enumerate(args):
        args[i] = args[i].to(device)
    return args


@torch.no_grad()
def visualize_tracking(pt_file = "/data2/zzc/nvdiffrec/data/video/may_test/track_params.pt",
                       device = torch.device('cuda:0'), batch_size = 1):
    fine_pt = pt_file

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

        cv2.imwrite(os.path.join('tools/viz_tracking', str(i).zfill(6) + '.png'), np.clip(render_imgs_, 0, 255).astype(np.uint8))


if __name__ == '__main__':
    visualize_tracking()
    os.system("ffmpeg -r 25 -pattern_type glob -i 'tools/viz_tracking/*.png' -c:v libx264 -pix_fmt yuv420p tools/viz_tracking/output.mp4")

# with Timer("loading image"):
#     imgs = iter(dataloader).__next__()


# import pdb;pdb.set_trace()


# fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input = False)
# falm3d = []
# for img in tqdm(imgs, desc = 'predict 3d lmk'):
#     img_ = img[..., ::-1]
#     pred = np.array(fa.get_landmarks(img_))[0]
#     print(pred.shape)
#     falm3d.append(pred[:, :2])


# falm3d = mean_filter(np.stack(falm3d), 5)
# for img_path in tqdm(img_paths, desc = 'read image'):
#     img = cv2.imread(img_path)
#     imgs.append(img)


## 2d lmk


# write_video(imgs, "/gruntdata/nas012/workspace/project/experiment/debug/lmk_2d.mp4", fps =25)

## 3d lmk


# rott_geo = forward_rott(geometry, sel_euler_, sel_trans_)
# mesh = o3d.geometry.TriangleMesh()
# mesh.vertices = o3d.utility.Vector3dVector(rott_geo[0].detach().cpu().numpy().reshape(-1, 3))
# mesh.triangles = o3d.utility.Vector3iVector(tris.reshape(-1, 3))
# o3d.io.write_triangle_mesh(os.path.join(os.path.dirname(fine_pt), f'geometry_frame_{i}.obj'), mesh)
#
# rott_geo = forward_transform(geometry, sel_euler_, sel_trans_, 1200, torch.tensor((450 / 2.0, 450 / 2.0), dtype = torch.float).cuda())
# mesh = o3d.geometry.TriangleMesh()
# mesh.vertices = o3d.utility.Vector3dVector(rott_geo[0].detach().cpu().numpy().reshape(-1, 3))
# mesh.triangles = o3d.utility.Vector3iVector(tris.reshape(-1, 3))
# o3d.io.write_triangle_mesh(os.path.join(os.path.dirname(fine_pt), 'geometry_frame.obj'), mesh)

# render_imgs_ = renderer(rott_geo, sel_texture, sel_light_)

#         masks_ = (render_imgs_[:, :, :, -1:]).detach() > 0.0
#         render_imgs_ = render_imgs_[..., :3]
#         masks_ = masks_.expand_as(render_imgs_)
#
#         render_imgs.append(render_imgs_.detach().cpu().numpy()[...])
#         masks.append(masks_.detach().cpu().numpy())
#         lm3d.append(lm3d_)
#
#     render_imgs = np.concatenate(render_imgs, axis = 0)
#     masks = np.concatenate(masks, axis = 0)
#     lm3d = np.concatenate(lm3d, axis = 0)
#
# # write_video(imgs, "/gruntdata/nas012/workspace/project/experiment/debug/lmk_2dvs3d.mp4", fps =25)
#
# merged_imgs = []
# for img, render_img, mask in tqdm(zip(imgs, render_imgs, masks), desc = 'merge'):
#     mask = 0.3 * mask
#     merged_img = img * (1 - mask) + render_img * mask
#
#     label = cv2.Canny((mask[..., 0] * 255).astype(np.uint8), 1, 1)
#     # kernel = np.ones((3, 3), np.uint8)
#     # edge_mask = cv2.dilate(label, kernel, iterations = 1)[...,None].repeat(3,2)/255
#     edge_mask = label[..., None].repeat(3, 2) / 255
#     blue_edge = edge_mask * (np.array([255, 0, 0])[None, None, :])
#     merged_img = merged_img * (1 - edge_mask) + blue_edge * blue_edge
#     merged_imgs.append(merged_img.astype(np.uint8))
#
# for img, lmk in zip(merged_imgs, lms_numpy):
#     for point in lmk:
#         cv2.circle(img, tuple(np.rint(point).astype(np.int).tolist()), 3,
#                    (0, 0, 255), 1)
#
# for img, lmk in zip(merged_imgs, lm3d):
#     for point in lmk:
#         cv2.circle(img, tuple(np.rint(point).astype(np.int).tolist()), 3,
#                    (255, 0, 0), 1)
#
# for img, lmk in zip(merged_imgs, falm3d):
#     for point in lmk:
#         cv2.circle(img, tuple(np.rint(point).astype(np.int).tolist()), 3,
#                    (0, 255, 0), 1)
#
# write_video(merged_imgs, "/gruntdata/nas012/workspace/project/experiment/debug/3d_fitting.mp4", fps = 25)
# #
#
#
#
#
#
