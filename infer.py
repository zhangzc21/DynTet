# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import glob
import os
import cv2
import numpy as np
import torch
import nvdiffrast.torch as dr
# Import data readers / generators
from dataset.dataset_face import DatasetTalkingHead, prepare_batch
# Import topology / geometry trainers
from dyntet.model import DMTetGeometry
from render import material
from dyntet.mlptexture import initial_guess_material
from render import light
from render.bfr import GFPGAN, DummyGFPGAN
import tqdm
from natsort import natsorted
from scipy import io
from params import get_FLAGS
from tools import common_utils
import imageio
###############################################################################
# Validation & testing
###############################################################################
def sliding_window_average(data, window_size):
    def gaussian_kernel(kernel_size, sigma):
        kernel = np.array([np.exp(-(x - kernel_size // 2) ** 2 / (2 * sigma ** 2)) for x in range(kernel_size)])
        kernel /= kernel.sum()
        return kernel
    window = gaussian_kernel(window_size, 1.5) #np.ones(window_size) / window_size
    result = np.apply_along_axis(lambda x: np.convolve(x, window, mode = 'same'), axis = 0, arr = data)
    return result

def obtain_seq_index(index, num_frames, window_size = 13):
    seq = list(range(index - window_size, index + window_size + 1))
    seq = [min(max(item, 0), num_frames - 1) for item in seq]
    return seq

def load_3dmm(target_3dmm):
    _3dmmexp64 = None
    if type(target_3dmm) is str:
        if target_3dmm.endswith('.npy'):
            _3dmmexp64 = np.load(target_3dmm)[:, list(range(80, 144))]
        elif target_3dmm.endswith('.mat'):
            _3dmmexp64 = io.loadmat(target_3dmm)['coeff_3dmm'][:, :64] # from sadtalker
        else:
            assert 'file type error'
    else:
        assert 'file type error'

    _3dmmexp_ = None
    if _3dmmexp64 is not None:
        ### An important step to correct the talking style
        mean_3dmm = \
            np.mean(np.load(os.path.join(FLAGS.ref_mesh, 'face_recon3dmm.npy'))[:, list(range(80, 144))], axis = 0)[
                None]
        _3dmmexp64 = _3dmmexp64 - np.mean(_3dmmexp64, axis = 0)[None] + mean_3dmm
        ###

        _3dmmexp64 = torch.from_numpy(_3dmmexp64).float()

        _3dmmexp_ = []
        for index in range(len(_3dmmexp64)):
            seq = obtain_seq_index(index, len(_3dmmexp64))
            _3dmmexp_.append(_3dmmexp64[seq])
        _3dmmexp_ = torch.stack(_3dmmexp_).float().permute(0, 2, 1)

    return _3dmmexp_

# ----------------------------------------------------------------------------
# Main function.
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    FLAGS = get_FLAGS()
    FLAGS.pre_load = True
    pretrain_dir = FLAGS.out_dir
    audio_file = FLAGS.audio
    drive_3dmm = FLAGS.drive_3dmm
    save_dir = os.path.join(pretrain_dir, "infer")


    glctx = dr.RasterizeCudaContext()

    # ==============================================================================================
    #  load pretrained models from stage 1
    # ==============================================================================================
    if FLAGS.learn_light:
        lgt = light.create_trainable_env_rnd(512, scale=0.0, bias=0.5)
    else:
        lgt = light.load_env(FLAGS.envmap, scale=FLAGS.env_scale)
    # Setup geometry for optimization
    geometry = DMTetGeometry(FLAGS.dmtet_grid, FLAGS.mesh_scale, FLAGS)
    # Setup textures, make initial guess from reference if possible
    mat = initial_guess_material(geometry, True, FLAGS, internal_dims = 256)
    mat['bsdf'] = 'diffuse'



    _3dmmexp_ = load_3dmm(drive_3dmm)
    if _3dmmexp_ is not None:
        n_images = len(_3dmmexp_)
    else: # if target 3dmm file is empty, we use the validation dataset.
        n_images = None

    params_pth = natsorted(glob.glob(os.path.join(pretrain_dir, 'params_*.pth')))[-1]
    print(f'load params from {params_pth}')
    state_dict = torch.load(params_pth)
    geometry.load_state_dict(state_dict['geometry'])
    mat.load_state_dict(state_dict['material'])
    lgt.load_state_dict(state_dict['light'])
    lgt.build_mips()

    try:
        ir_model = GFPGAN(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=2,
                decoder_load_path=None,
                fix_decoder=False,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True).cuda()
        file_path = natsorted(glob.glob(os.path.join(pretrain_dir, 'finetune_bfr', '*.pth')))[-1]
        ir_model.load_state_dict(torch.load(file_path))
        print(f'load tuned gfpgan from {file_path}')
        flag = '_bfr'

    except:
        ir_model = DummyGFPGAN()
        flag = ''

    save_dir = save_dir + flag

    FLAGS.data_range = [0, 200]
    dataset_validate = DatasetTalkingHead(os.path.join(FLAGS.ref_mesh, 'mv_transforms_val.json'),
                                          FLAGS, examples = n_images)
    dataset_validate.mv = dataset_validate.smooth_mv # use smooth mv to eliminate shaking


    def split_batch(batch):
        iter_res, iter_spp = batch['resolution'], batch['spp']
        base = {
            'resolution': iter_res,
            'spp': iter_spp,
        }
        num = len(batch['img'])
        res = []
        for i in range(num):
            d = dict(**base)
            for key in batch.keys():
                if key not in base.keys():
                    d.update({key: batch[key][[i]]})
            res.append(d)
        return res


    target = dataset_validate.__getitem__(0)
    mvp = target ['mvp'].cuda()
    exp_pose = target['exp'][:, -6:, :]
    @torch.no_grad()
    def validate_ir(glctx,  geometry, opt_material, lgt, dataset, out_dir, FLAGS, log_interval = 1):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=dataset.collate, shuffle = False, num_workers = 1)
        os.makedirs(out_dir, exist_ok=True)
        from collections import defaultdict
        results = []
        for it, target in tqdm.tqdm(enumerate(dataloader), desc = 'infer'):
            if _3dmmexp_ is not None:
                target['exp'] = torch.cat([_3dmmexp_[[it],:64], target['exp'][:, -6:, :]], dim = 1)

            target = prepare_batch(target, FLAGS.background)

            with common_utils.Timer('inference time', show = True):
                buffers = geometry.infer(glctx, target, lgt, opt_material, show_time = False)
                fg_image = buffers['shaded'][..., 0:3]
                fg_image = ir_model.restore_from_render(fg_image)
                fg_image = torch.clip(fg_image * 255, 0,255 )[0].detach().cpu().numpy().astype(np.uint8)
                results.append(fg_image)

        imageio.mimwrite(os.path.join(save_dir, 'video.mp4'), results, fps = 25, quality = 8,
                             macro_block_size = 1)

    validate_ir(glctx, geometry, mat, lgt, dataset_validate, save_dir, FLAGS, log_interval=1)

    # merge video and audio
    if audio_file is not None:
        output_file = os.path.join(save_dir, 'video_audio.mp4')
        os.system(f"ffmpeg -y -i {os.path.join(save_dir, 'video.mp4')} -i {audio_file} -c:v copy -c:a aac -strict experimental {output_file}")
# ----------------------------------------------------------------------------
