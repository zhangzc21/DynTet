# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import glob
import json

import torch
import numpy as np
import tqdm
from render import util
from scipy.spatial.transform import Slerp, Rotation
from packaging import version as pver
from .dataset import Dataset
import cv2
from dataset.utils import matrix_to_euler_angles, euler_angles_to_matrix


@torch.cuda.amp.autocast(enabled=False)
def convert_poses(poses):
    # poses: [B, 4, 4]
    # return [B, 3], 4 rot, 3 trans
    out = torch.empty(poses.shape[0], 6, dtype=torch.float32, device=poses.device)
    out[:, :3] = matrix_to_euler_angles(poses[:, :3, :3])
    out[:, 3:] = poses[:, :3, 3]
    return out

@torch.cuda.amp.autocast(enabled=False)
def inverse_poses(poses):
    # poses: [B, 6]
    # return [B, 4, 4]
    out = torch.zeros(poses.shape[0], 4,  4, dtype=torch.float32, device=poses.device)
    rot = euler_angles_to_matrix(poses[:, :3])
    out[:, :3, :3 ] = rot
    out[:, :3, 3] = poses[:, 3:]
    out[:, 3, 3] = 1
    return out

@torch.no_grad()
def gaussian_blur(input_signal):
    def gaussian_kernel(kernel_size, sigma):
        kernel = torch.tensor([np.exp(-(x - kernel_size // 2) ** 2 / (2 * sigma ** 2)) for x in range(kernel_size)],
                              dtype=torch.float32)
        kernel /= kernel.sum()
        return kernel

    kernel_size = 25
    sigma = 5

    gaussian = gaussian_kernel(kernel_size, sigma).to(input_signal)
    print(gaussian)

    output_signal = torch.nn.functional.conv1d(input_signal.permute(1,0).unsqueeze(1), gaussian.view(1, 1, -1), padding='same')

    return output_signal[:, 0].permute(1,0)

def focal_length_to_fovy(focal_length, sensor_height):
    return 2 * np.arctan(0.5 * sensor_height / focal_length)


def perspective(fovy = 0.7854, aspect = 1.0, n = 0.1, f = 1000.0, device = None):
    y = np.tan(fovy / 2)
    return torch.tensor([[-1 / (y * aspect), 0, 0, 0],
                         [0, 1 / y, 0, 0],
                         [0, 0, (f + n) / (f - n), -(2 * f * n) / (f - n)],
                         [0, 0, 1, 0]], dtype = torch.float32, device = device)


###############################################################################
# NERF image based dataset (synthetic)
###############################################################################
def get_audio_features(features, att_mode, index):
    if att_mode == 0:
        return features[[index]]
    elif att_mode == 1:
        left = index - 8
        pad_left = 0
        if left < 0:
            pad_left = -left
            left = 0
        auds = features[left:index]
        if pad_left > 0:
            # pad may be longer than auds, so do not use zeros_like
            auds = torch.cat([torch.zeros(pad_left, *auds.shape[1:], device=auds.device, dtype=auds.dtype), auds], dim=0)
        return auds
    elif att_mode == 2:
        left = index - 4
        right = index + 4
        pad_left = 0
        pad_right = 0
        if left < 0:
            pad_left = -left
            left = 0
        if right > features.shape[0]:
            pad_right = right - features.shape[0]
            right = features.shape[0]
        auds = features[left:right]
        if pad_left > 0:
            auds = torch.cat([torch.zeros_like(auds[:pad_left]), auds], dim=0)
        if pad_right > 0:
            auds = torch.cat([auds, torch.zeros_like(auds[:pad_right])], dim=0) # [8, 16]
        return auds
    else:
        raise NotImplementedError(f'wrong att_mode: {att_mode}')

def smooth_camera_path(poses, kernel_size = 5):
    # smooth the camera trajectory...
    # poses: [N, 4, 4], numpy array

    N = poses.shape[0]
    K = kernel_size // 2

    trans = poses[:, :3, 3].copy()  # [N, 3]
    rots = poses[:, :3, :3].copy()  # [N, 3, 3]

    for i in range(N):
        start = max(0, i - K)
        end = min(N, i + K + 1)
        poses[i, :3, 3] = trans[start:end].mean(0)
        poses[i, :3, :3] = Rotation.from_matrix(rots[start:end]).mean().as_matrix()

    return poses

def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

@torch.cuda.amp.autocast(enabled=False)
def get_bg_coords(H, W, device):
    X = torch.arange(H, device=device) / (H - 1) * 2 - 1 # in [-1, 1]
    Y = torch.arange(W, device=device) / (W - 1) * 2 - 1 # in [-1, 1]
    xs, ys = custom_meshgrid(X, Y)
    bg_coords = torch.cat([xs.reshape(-1, 1), ys.reshape(-1, 1)], dim=-1).unsqueeze(0) # [1, H*W, 2], in [-1, 1]
    return bg_coords


def polygon_area(x, y):
    x_ = x - x.mean()
    y_ = y - y.mean()
    correction = x_[-1] * y_[0] - y_[-1]* x_[0]
    main_area = np.dot(x_[:-1], y_[1:]) - np.dot(y_[:-1], x_[1:])
    return 0.5 * np.abs(main_area + correction)

def _load_img(path, size):
    # img = util.load_image_raw(path)
    # if img.dtype != np.float32: # LDR image
    #     img = torch.tensor(img / 255, dtype=torch.float32)
    #     img[..., 0:3] = util.srgb_to_rgb(img[..., 0:3])
    # else:
    #     img = torch.tensor(img, dtype=torch.float32)
    images = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # [H, W, 3]
    if images.shape[-1] == 3:
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
    elif images.shape[-1] == 4:
        images = np.ascontiguousarray(images[...,[2,1,0,3]])
    if size:
        images = cv2.resize(images, size)
    images = images.astype(np.float32) / 255  # [H, W, 3]
    images = torch.from_numpy(images)
    return images




class DatasetTalkingHead(Dataset):
    '''
    Talking head dataset
    '''
    def __init__(self, cfg_path, FLAGS, examples = None):
        self.FLAGS = FLAGS

        self.examples = examples
        self.base_dir = os.path.dirname(cfg_path)
        self.root_path = self.base_dir

        self.cfg = json.load(open(cfg_path, 'r'))
        self.start_index = self.FLAGS.data_range[0]
        self.end_index = self.FLAGS.data_range[-1]
        self.n_images = len(self.cfg['frames'])
        if self.end_index == -1:  # abuse...
            self.end_index = self.n_images
        if self.end_index != self.n_images:
            self.cfg['frames'] = self.cfg['frames'][self.start_index:self.end_index] + self.cfg['frames'][self.start_index:self.end_index][::-1]
            self.n_images = len(self.cfg['frames'])

        # Determine resolution & aspect ratio
        f_path = os.path.join(self.root_path, 'gt_imgs', str(self.cfg['frames'][0]['img_id']) + '.jpg')
        self.resolution = cv2.imread(f_path).shape[:2]
        self.H, self.W = self.resolution[0], self.resolution[1]
        self.aspect = self.resolution[1] / self.resolution[0]

        if self.FLAGS.local_rank == 0:
            print(
                "DatasetTalkingHead: %d images with shape [%d, %d]" % (self.n_images, self.resolution[0], self.resolution[1]))

        # self._parse_common_attributes()
        # Loading small files
        self._parse_eye_lip()
        self._parse_3dmm()
        self._parse_pose()
        # Pre-load from disc to avoid slow png parsingresolution
        if self.FLAGS.pre_load:
            self.preloaded_data = []
            for i in tqdm.tqdm(range(self.n_images), desc = f'Loading data'):
                self.preloaded_data += [self._parse_frame(i)]

    def _parse_3dmm(self):
        '''
        loading 3DMM coefficients.
            - _3dmmexp_gt: 3DMM coefficients extracted by AD-NeRF, used for 3DMM guided losses
            - _3dmmexp: 3DMM coefficients predicted by Deep3DFaceRecon, used for DynTet input
            - scale: Elastic score estimated by 3DMM model
        '''
        _3dmmexp = np.stack([self.cfg['frames'][idx]['exp'] for idx in range(self.n_images)])
        self._3dmmexp_gt = torch.from_numpy(_3dmmexp).float()

        _3dmmexp70 = np.load(os.path.join(self.root_path, 'face_recon3dmm.npy'))[:,list(range(80,144))+list(range(224,227)) + list(range(254,257))]

        _3dmmexp = np.stack([_3dmmexp70[int(self.cfg['frames'][idx]['img_id'])] for idx in range(self.n_images)])
        _3dmmexp = torch.from_numpy(_3dmmexp).float()
        print(_3dmmexp.shape)

        def obtain_seq_index(index, num_frames, window_size = 13):
            seq = list(range(index - window_size, index + window_size + 1))
            seq = [min(max(item, 0), num_frames - 1) for item in seq]
            return seq

        self._3dmmexp = []
        for index in range(len(_3dmmexp)):
            seq = obtain_seq_index(index, len(_3dmmexp))
            self._3dmmexp.append(_3dmmexp[seq])
        self._3dmmexp = torch.stack(self._3dmmexp).float().permute(0,2,1)


        with torch.no_grad():
            from data_utils.face_3dmm import Face_3DMM
            model_3dmm = Face_3DMM(os.path.join('data_utils', 'face_tracking', '3DMM'), 100, 79, 100, 34650).cuda()
            face_vert0 = model_3dmm.forward_geo(torch.zeros(1,100).cuda(), torch.zeros(1,79).cuda())
            diff = torch.zeros(len(face_vert0)).cuda()
            for i in tqdm.tqdm(range(len(self._3dmmexp_gt))):
                exp = self._3dmmexp_gt[[i]].cuda()
                face_vert = model_3dmm.forward_geo(torch.zeros(1,100).cuda(), exp)
                diff = torch.max((face_vert - face_vert0).square().sum(dim=-1).sqrt(), diff)
            self.scale = diff.detach().cpu()
            self.scale[self.scale<0.02] = 0.02

    def _parse_eye_lip(self):
        '''
        loading eye and lip information
        '''
        transform = self.cfg
        frames = transform["frames"]

        self.eye_area = []
        self.rects = []
        # frames = frames[self.start_index:self.end_index]
        for f in tqdm.tqdm(frames, desc = f'Loading eye lip data'):
            lms = np.loadtxt(os.path.join(self.root_path, 'ori_imgs', str(f['img_id']) + '.lms'))  # [68, 2]
            eyes_left = slice(36, 42)
            eyes_right = slice(42, 48)
            lips = slice(48, 60)

            eyes_left_rect = self.get_rect_from_lms(eyes_left, lms)
            eyes_right_rect = self.get_rect_from_lms(eyes_right, lms)
            lips_rect = self.get_rect_from_lms(lips, lms)
            rect = torch.tensor([eyes_left_rect, eyes_right_rect, lips_rect])

            area_left = polygon_area(lms[eyes_left, 0], lms[eyes_left, 1])
            area_right = polygon_area(lms[eyes_right, 0], lms[eyes_right, 1])
            area = (area_left + area_right) / (self.H * self.W) * 100

            self.rects.append(rect)
            self.eye_area.append(area)

        self.eye_area = np.array(self.eye_area, dtype=np.float32)
        ori_eye = self.eye_area.copy()
        for i in range(ori_eye.shape[0]):
            start = max(0, i - 1)
            end = min(ori_eye.shape[0], i + 2)
            self.eye_area[i] = ori_eye[start:end].mean()
        self.eye_area = torch.from_numpy(self.eye_area).view(-1, 1) # [N, 1]
        print('eye area:', self.eye_area.min(), self.eye_area.max())
        # self.eye_area = (self.eye_area - self.eye_area.min())/ (self.eye_area.max() - self.eye_area.min())

    def get_rect_from_lms(self, index, lms):
        xmin, xmax = int(lms[index, 0].min()), int(lms[index, 0].max())
        ymin, ymax = int(lms[index, 1].min()), int(lms[index, 1].max())

        padding = 10
        xmin = max(0, xmin - padding) * self.FLAGS.train_res[0] / self.W
        xmax = min(self.W, xmax + padding) * self.FLAGS.train_res[0] / self.W
        ymin = max(0, ymin - padding) * self.FLAGS.train_res[1] / self.H
        ymax = min(self.H, ymax + padding) * self.FLAGS.train_res[1] / self.H
        rect = [xmin, ymin, xmax, ymax]
        return rect

    # def _parse_common_attributes(self,):
    #     '''
    #     loading
    #         - background images
    #     '''
    #     transform = self.cfg
    #
    #     if self.FLAGS.bg_img == 'white':  # special
    #         bg_img = np.ones((self.H, self.W, 3), dtype = np.float32)
    #     elif self.FLAGS.bg_img == 'black':  # special
    #         bg_img = np.zeros((self.H, self.W, 3), dtype = np.float32)
    #     else:  # load from file
    #         # default bg
    #         if self.FLAGS.bg_img == '':
    #             self.FLAGS.bg_img = os.path.join(self.root_path, 'bc.jpg')
    #         bg_img = cv2.imread(self.FLAGS.bg_img, cv2.IMREAD_UNCHANGED)  # [H, W, 3]
    #         if bg_img.shape[0] != self.H or bg_img.shape[1] != self.W:
    #             bg_img = cv2.resize(bg_img, (self.W, self.H), interpolation = cv2.INTER_AREA)
    #         bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
    #         bg_img = bg_img.astype(np.float32) / 255  # [H, W, 3/4]
    #
    #     self.bg_img = bg_img

    def _parse_pose(self):
        '''
        loading pose information
            - mv: Model View Projection matrix

            - smooth_mv: smoothed transformation matrix
            - mean_mv: mean of the transformation matrix
            - poses6: Euler angle and translation vector
            - mean_pose: mean of Euler angle and translation vector
        '''
        poses = []
        frames = self.cfg['frames']
        mvs = []
        for f in tqdm.tqdm(frames, desc=f'Loading pose data'):
            mv = torch.tensor(f['transform_matrix'], dtype=torch.float32)
            mvs.append(mv)
            pose = convert_poses(mv[None])[0]
            poses.append(pose)
        poses = torch.stack(poses)
        self.mv = torch.stack(mvs)

        self.poses6 = gaussian_blur(poses)
        self.smooth_mv = inverse_poses(self.poses6)
        self.mean_pose = self.poses6.mean(dim = 0)
        self.mean_mv = inverse_poses(self.mean_pose[None])[0]

    def _parse_frame(self, idx, downscale = 1):
        '''
        Parse attributes of each frame for dataset __getitem__
        '''
        cfg = self.cfg

        img = _load_img(os.path.join(self.root_path, 'gt_imgs', str(cfg['frames'][idx]['img_id']) + '.jpg'), size = self.FLAGS.train_res)
        bg_image = _load_img(os.path.join(self.root_path, 'bg_torso', str(cfg['frames'][idx]['img_id']) + '.jpg'),
                        size=self.FLAGS.train_res)
        seg = cv2.imread(os.path.join(self.root_path, 'parsing', str(cfg['frames'][idx]['img_id']) + '.png'))
        head_part = (seg[..., 0] == 255) & (seg[..., 1] == 0) & (seg[..., 2] == 0)
        head_part = cv2.resize(head_part.astype(np.float32), self.FLAGS.train_res)
        head_mask = torch.from_numpy(head_part)

        img = torch.cat([img, torch.ones_like(img)[..., :1]], dim=-1)

        if 'focal_len' in cfg:
            fl_x = fl_y = cfg['focal_len']
        elif 'fl_x' in cfg or 'fl_y' in cfg:
            fl_x = (cfg['fl_x'] if 'fl_x' in cfg else cfg['fl_y']) / downscale
            fl_y = (cfg['fl_y'] if 'fl_y' in cfg else cfg['fl_x']) / downscale
        elif 'camera_angle_x' in cfg or 'camera_angle_y' in cfg:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = self.W / (
                    2 * np.tan(cfg['camera_angle_x'] / 2)) if 'camera_angle_x' in cfg else None
            fl_y = self.H / (
                    2 * np.tan(cfg['camera_angle_y'] / 2)) if 'camera_angle_y' in cfg else None
            if fl_x is None: fl_x = fl_y
            if fl_y is None: fl_y = fl_x
        else:
            raise RuntimeError('Failed to load focal length, please check the transforms.json!')

        cx = (cfg['cx'] / downscale) if 'cx' in cfg else (self.W / 2)
        cy = (cfg['cy'] / downscale) if 'cy' in cfg else (self.H / 2)
        fovy = focal_length_to_fovy(fl_y, 2 * cy)
        proj = util.perspective(fovy, self.aspect, self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])

        # Load image data and modelview matrix
        exp = self._3dmmexp[[idx]].float()
        exp_gt = self._3dmmexp_gt[[idx]].float()
        id = torch.tensor(cfg['frames'][idx]['id'])
        rect = self.rects[idx]

        mv = self.mv[idx]
        campos = -mv[:3, 3]
        mvp = proj @ mv

        return {
            'mv': mv[None],  # model to world
            'mvp': mvp[None], # model to view projection
            'campos': campos[None], # camera location
            'resolution': self.FLAGS.train_res, # image resolution
            'spp': self.FLAGS.spp,
            'img': img[None],
            'exp': exp, # 3dmm exp coefficient predicted by network
            'exp_gt': exp_gt, # 3dmm exp coefficient extracted by AD-NeRF
            'id': id, # 3dmm id coefficient extracted by AD-NeRF
            'rect': rect[None].long(), # eye and lip bounding box
            'scale': self.scale, # elastic score estimated by 3DMM
            'head_mask': head_mask[None, ..., None],
            'bg': bg_image[None], # background + torso
        }

    def __len__(self):
        return self.n_images if self.examples is None else self.examples

    def __getitem__(self, itr):
        if self.FLAGS.pre_load:
            data = self.preloaded_data[itr % self.n_images]
        else:
            data = self._parse_frame(itr % self.n_images)
        return data

    def collate(self, batch):
        iter_res, iter_spp = batch[0]['resolution'], batch[0]['spp']
        base =  {
            'resolution' : iter_res,
            'spp' : iter_spp,
        }
        for key in batch[0].keys():
            if key not in base.keys():
                base.update({key: torch.cat(list([item[key] for item in batch]), dim=0)})
        return base

@torch.no_grad()
def prepare_batch(target, bg_type='black'):
    assert len(target['img'].shape) == 4, "Image shape should be [n, h, w, c]"
    if bg_type == 'checker':
        background = torch.tensor(util.checkerboard(target['img'].shape[1:3], 8), dtype=torch.float32, device='cuda')[None, ...]
    elif bg_type == 'black':
        background = torch.zeros(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    elif bg_type == 'white':
        background = torch.ones(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    elif bg_type == 'reference':
        background = target['img'][..., 0:3]
    elif bg_type == 'random':
        background = torch.rand(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    elif bg_type == 'target':
        background = target['bg']
    else:
        assert False, "Unknown background type %s" % bg_type
    target['background'] = background

    for key in target.keys():
        if isinstance(target[key], torch.Tensor):
            target[key] = target[key].cuda()

    return target