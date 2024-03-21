# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import time

import numpy as np
import torch
import nvdiffrast.torch as dr

# Import data readers / generators
from dataset.dataset_face import DatasetTalkingHead, prepare_batch
# Import topology / geometry trainers
from dyntet.model import DMTetGeometry
from dyntet.mlptexture import initial_guess_material

from render import material
from render import util
from render import light
import tqdm
import trimesh
from tools import common_utils
from params import get_FLAGS

# Enable to debug back-prop anomalies
# torch.autograd.set_detect_anomaly(True)

###############################################################################
# Utility functions for material
###############################################################################





###############################################################################
# Validation & testing
###############################################################################

def validate_itr(glctx, target, geometry, opt_material, lgt, FLAGS):
    result_dict = {}
    with torch.no_grad():
        lgt.build_mips()
        if FLAGS.camera_space_light:
            lgt.xfm(target['mv'])

        buffers = geometry.predict(glctx, target, lgt, opt_material)

        result_dict['ref'] = torch.clamp(target['img'][..., 0:3][0], 0.0, 1.0)
        result_dict['opt'] = torch.clamp(buffers['shaded'][..., 0:3][0], 0.0, 1.0)
        result_dict['scale'] = torch.clamp(buffers['scale'][..., 0:3][0] * 5, 0.0, 1.0)

        result_image = torch.cat([result_dict['ref'], result_dict['opt'], result_dict['scale']], axis=1)

        if FLAGS.display is not None:
            for layer in FLAGS.display:
                if 'latlong' in layer and layer['latlong']:
                    if isinstance(lgt, light.EnvironmentLight):
                        result_dict['light_image'] = util.cubemap_to_latlong(lgt.base, FLAGS.display_res)
                    result_image = torch.cat([result_image, result_dict['light_image']], axis=1)
                elif 'relight' in layer:
                    if not isinstance(layer['relight'], light.EnvironmentLight):
                        layer['relight'] = light.load_env(layer['relight'])
                    img = geometry.predict(glctx, target, layer['relight'], opt_material)
                    result_dict['relight'] = img[..., 0:3][0]
                    result_image = torch.cat([result_image, result_dict['relight']], axis=1)
                elif 'bsdf' in layer:
                    buffers = geometry.predict(glctx, target, lgt, opt_material, bsdf=layer['bsdf'])
                    if layer['bsdf'] == 'kd':
                        result_dict[layer['bsdf']] = buffers['shaded'][0, ..., 0:3]
                    elif layer['bsdf'] == 'normal':
                        result_dict[layer['bsdf']] = (buffers['shaded'][0, ..., 0:3] + 1) * 0.5
                    else:
                        result_dict[layer['bsdf']] = buffers['shaded'][0, ..., 0:3]
                    result_image = torch.cat([result_image, result_dict[layer['bsdf']]], axis=1)

        return result_image, result_dict


def validate(glctx, geometry, opt_material, lgt, dataset_validate, out_dir, FLAGS):
    # ==============================================================================================
    #  Validation loop
    # ==============================================================================================
    mse_values = []
    psnr_values = []

    dataloader_validate = torch.utils.data.DataLoader(dataset_validate, batch_size=1,
                                                      collate_fn=dataset_validate.collate)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'metrics.txt'), 'w') as fout:
        fout.write('ID, MSE, PSNR\n')

        print("Running validation")
        for it, target in enumerate(dataloader_validate):

            # Mix validation background
            target = prepare_batch(target, FLAGS.background)

            result_image, result_dict = validate_itr(glctx, target, geometry, opt_material, lgt, FLAGS)

            # Compute metrics
            opt = torch.clamp(result_dict['opt'], 0.0, 1.0)
            ref = torch.clamp(result_dict['ref'], 0.0, 1.0)

            mse = torch.nn.functional.mse_loss(opt, ref, size_average=None, reduce=None, reduction='mean').item()
            mse_values.append(float(mse))
            psnr = util.mse_to_psnr(mse)
            psnr_values.append(float(psnr))

            line = "%d, %1.8f, %1.8f\n" % (it, mse, psnr)
            fout.write(str(line))

            for k in result_dict.keys():
                np_img = result_dict[k].detach().cpu().numpy()
                util.save_image(out_dir + '/' + ('val_%06d_%s.png' % (it, k)), np_img)

        avg_mse = np.mean(np.array(mse_values))
        avg_psnr = np.mean(np.array(psnr_values))
        line = "AVERAGES: %1.4f, %2.3f\n" % (avg_mse, avg_psnr)
        fout.write(str(line))
        print("MSE,      PSNR")
        print("%1.8f, %2.3f" % (avg_mse, avg_psnr))
    return avg_psnr


###############################################################################
# Main shape fitter function / optimization loop
###############################################################################

class Trainer(torch.nn.Module):
    def __init__(self, glctx, geometry, lgt, mat, optimize_geometry, optimize_light, FLAGS):
        super(Trainer, self).__init__()

        self.glctx = glctx
        self.geometry = geometry
        self.light = lgt
        self.material = mat
        self.optimize_geometry = optimize_geometry
        self.optimize_light = optimize_light
        self.FLAGS = FLAGS

        if not self.optimize_light:
            with torch.no_grad():
                self.light.build_mips()

        self.params = list(self.material.parameters())
        self.params += list(self.light.parameters()) if optimize_light else []
        self.geo_params = list(self.geometry.parameters()) if optimize_geometry else []

    def forward(self, target, it):
        if self.optimize_light:
            self.light.build_mips()
            if self.FLAGS.camera_space_light:
                self.light.xfm(target['mv'])

        return self.geometry.tick(glctx, target, self.light, self.material, it)


def optimize_mesh(
        glctx,
        geometry,
        opt_material,
        lgt,
        dataset_train,
        dataset_validate,
        FLAGS,
        warmup_iter=0,
        log_interval=10,
        optimize_light=True,
        optimize_geometry=True
):
    # ==============================================================================================
    #  Setup torch optimizer
    # ==============================================================================================

    learning_rate =  FLAGS.learning_rate
    learning_rate_pos = learning_rate[0] if isinstance(learning_rate, list) or isinstance(learning_rate,
                                                                                          tuple) else learning_rate
    learning_rate_mat = learning_rate[1] if isinstance(learning_rate, list) or isinstance(learning_rate,
                                                                                          tuple) else learning_rate

    def lr_schedule(iter, fraction):
        if iter < warmup_iter:
            return iter / warmup_iter
        return max(1e-2,
                   10 ** (-(iter - warmup_iter) * 0.0002))  # Exponential falloff from [1.0, 0.1] over 5k epochs.

    # ==============================================================================================
    #  Image loss
    # ==============================================================================================
    trainer_noddp = Trainer(glctx, geometry, lgt, opt_material, optimize_geometry, optimize_light, FLAGS)

    if FLAGS.multi_gpu:
        # Multi GPU training mode
        import apex
        from apex.parallel import DistributedDataParallel as DDP

        trainer = DDP(trainer_noddp)
        trainer.train()
        if optimize_geometry:
            optimizer_mesh = apex.optimizers.FusedAdam(trainer_noddp.geo_params, lr=learning_rate_pos)
            scheduler_mesh = torch.optim.lr_scheduler.LambdaLR(optimizer_mesh, lr_lambda=lambda x: lr_schedule(x, 0.9))

        optimizer = apex.optimizers.FusedAdam(trainer_noddp.params, lr=learning_rate_mat)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_schedule(x, 0.9))
    else:
        # Single GPU training mode
        trainer = trainer_noddp
        if optimize_geometry:
            optimizer_mesh = torch.optim.Adam(trainer_noddp.geo_params, lr=learning_rate_pos)
            scheduler_mesh = torch.optim.lr_scheduler.LambdaLR(optimizer_mesh, lr_lambda=lambda x: lr_schedule(x, 0.9))

        optimizer = torch.optim.Adam(trainer_noddp.params, lr=learning_rate_mat)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_schedule(x, 0.9))

        # ==============================================================================================
    #  Training loop
    # ==============================================================================================
    img_cnt = 0
    from collections import defaultdict
    losses_recorder = defaultdict(list)

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=FLAGS.batch,
                                                   collate_fn=dataset_train.collate, shuffle=True, num_workers = 8)
    dataloader_validate = torch.utils.data.DataLoader(dataset_validate, batch_size=1, collate_fn=dataset_train.collate)

    def cycle(iterable):
        iterator = iter(iterable)
        while True:
            try:
                yield next(iterator)
            except StopIteration:
                iterator = iter(iterable)

    v_it = cycle(dataloader_validate)

    for it, target in tqdm.tqdm(enumerate(dataloader_train)):

        target = prepare_batch(target, FLAGS.background)

        # ==============================================================================================
        #  Display / save outputs. Do it before training so we get initial meshes
        # ==============================================================================================

        # Show/save image before training step (want to get correct rendering of input)
        if FLAGS.local_rank == 0:
            display_image = FLAGS.display_interval and (it % FLAGS.display_interval == 0)
            save_image = FLAGS.save_interval and (it % FLAGS.save_interval == 0)
            if display_image or save_image:
                result_image, result_dict = validate_itr(glctx, prepare_batch(next(v_it), FLAGS.background), geometry,
                                                         opt_material, lgt, FLAGS)
                np_result_image = result_image.detach().cpu().numpy()
                if display_image:
                    util.display_image(np_result_image, title='%d / %d' % (it, FLAGS.iter))
                if save_image:
                    util.save_image(FLAGS.out_dir + '/' + ('img_%06d.png' % (img_cnt)), np_result_image)
                    img_cnt = img_cnt + 1

        iter_start_time = time.time()

        # ==============================================================================================
        #  Zero gradients
        # ==============================================================================================
        optimizer.zero_grad()
        if optimize_geometry:
            optimizer_mesh.zero_grad()

        # ==============================================================================================
        #  Training
        # ==============================================================================================
        losses = trainer(target, it)

        # ==============================================================================================
        #  Final loss
        # ==============================================================================================
        total_loss =   1 *  losses['img_loss'] \
                     + 10 * losses['silhouette_loss'] \
                     + 0.1 * losses['region_loss'] \
                     + 0.1 * losses['lpips_loss'] \
                     + (100 if it < 5000 else 0) * losses['geo_loss']  \
                     + (100 if it < 5000 else 0) * losses['deform3dmm_loss'] \
                     + 100 * losses['scale_reg'] \
                     + losses['depth_3dmm'] \
                     + losses['reg_loss']

        # ==============================================================================================
        #  Backpropagate
        # ==============================================================================================
        total_loss.backward()
        if hasattr(lgt, 'base') and lgt.base.grad is not None and optimize_light:
            lgt.base.grad *= 64
        if 'kd_ks_normal' in opt_material:
            try:
                opt_material['kd_ks_normal'].encoder.params.grad /= 8.0
            except:
                pass

        optimizer.step()
        scheduler.step()

        if optimize_geometry:
            optimizer_mesh.step()
            scheduler_mesh.step()

        # ==============================================================================================
        #  Clamp trainables to reasonable range
        # ==============================================================================================
        with torch.no_grad():
            if 'kd' in opt_material:
                opt_material['kd'].clamp_()
            if 'ks' in opt_material:
                opt_material['ks'].clamp_()
            if 'normal' in opt_material:
                opt_material['normal'].clamp_()
                opt_material['normal'].normalize_()
            if lgt is not None:
                lgt.clamp_(min=0.0)

        torch.cuda.current_stream().synchronize()
        losses.update(dict(iter_dur=time.time() - iter_start_time))
        for key, value in losses.items():
            losses_recorder[key].append(float(value))
        # ==============================================================================================
        #  Logging
        # ==============================================================================================
        if ((it + 1) % log_interval == 0 or it == 0)  == 0 and FLAGS.local_rank == 0:
            avg_recorder = defaultdict(int)
            for key, value in losses.items():
                avg_recorder[key] = np.mean(np.asarray(losses_recorder[key][-log_interval:]))
            avg_recorder['lr'] = float(optimizer.param_groups[0]['lr'])
            avg_recorder['rem'] = (FLAGS.iter - it) * avg_recorder['iter_dur']

            out_str = "iter=%5d" % (it)
            for key, value in avg_recorder.items():
                out_str += f', {key}=%6f' % (value)
            out_str += f', vertices {geometry.opt_mesh.v_pos.shape[1]}, faces {len(geometry.opt_mesh.t_pos_idx)}'
            print(out_str)

        if ((it + 1) % 5000 == 0 or it == 0) and FLAGS.local_rank == 0:
            state_dict = {'geometry': geometry.state_dict(),
                          'material': opt_material.state_dict(),
                          'light': lgt.state_dict() if trainer.optimize_light else [],
                          'it': it}
            torch.save(state_dict, os.path.join(FLAGS.out_dir, f'params_{it}.pth'))


        if it % (10 * log_interval) == 0 and FLAGS.local_rank == 0:
            common_utils.save_dict_to_json(losses_recorder, os.path.join(FLAGS.out_dir, 'logs', 'loss.json'))
            state_dict = {'geometry': geometry.state_dict(),
                          'material': opt_material.state_dict(),
                          'light': lgt.state_dict() if trainer.optimize_light else [],
                          'it': it}
            torch.save(state_dict, os.path.join(FLAGS.out_dir, 'params.pth'))

            opt_mesh = geometry.get_dynamic_mesh(None, torch.zeros(1, target['exp'].shape[1], 27).cuda())

            mesh = trimesh.Trimesh(vertices=opt_mesh.v_pos[0].detach().cpu().numpy(),
                                   faces=opt_mesh.t_pos_idx.detach().cpu().numpy())

            os.makedirs(os.path.join(FLAGS.out_dir, 'mesh_train'), exist_ok=True)
            trimesh.exchange.export.export_mesh(mesh, os.path.join(FLAGS.out_dir, 'mesh_train', f'mesh_close_{it}.obj'))

    return geometry, opt_material


# ----------------------------------------------------------------------------
# Main function.
# ----------------------------------------------------------------------------

if __name__ == "__main__":

    FLAGS = get_FLAGS()
    os.makedirs(FLAGS.out_dir, exist_ok=True)
    file_list = [FLAGS.config, 'train.py', 'dyntet/model.py', 'dyntet/mlptexture.py']
    target_folder = os.path.join(FLAGS.out_dir, 'key_files')
    common_utils.save_files_to_folder(file_list, target_folder)

    glctx = dr.RasterizeCudaContext()

    # ==============================================================================================
    #  Create data pipeline
    # ==============================================================================================
    assert os.path.isfile(os.path.join(FLAGS.ref_mesh, 'mv_transforms_train.json'))

    dataset_train = DatasetTalkingHead(os.path.join(FLAGS.ref_mesh, 'mv_transforms_train.json'), FLAGS,
                                       examples=(FLAGS.iter + 1) * FLAGS.batch)
    dataset_validate = DatasetTalkingHead(os.path.join(FLAGS.ref_mesh, 'mv_transforms_val.json'), FLAGS)

    # ==============================================================================================
    #  Create env light with trainable parameters
    # ==============================================================================================
    if FLAGS.learn_light:
        lgt = light.create_trainable_env_rnd(512, scale=0.0, bias=0.5)
    else:
        lgt = light.load_env(FLAGS.envmap, scale=FLAGS.env_scale)

    # ==============================================================================================
    # Create DynTet
    # ==============================================================================================

    # Setup geometry for optimization
    geometry = DMTetGeometry(FLAGS.dmtet_grid, FLAGS.mesh_scale, FLAGS)
    geometry.initialize_shape(shape_init = 'face')

    # Setup textures, make initial guess from reference if possible
    mat = initial_guess_material(geometry, True, FLAGS)
    mat['bsdf'] = 'diffuse'

    if FLAGS.resume:
        params_pth = os.path.join(FLAGS.out_dir, 'params.pth')
        print(f'load params from {params_pth}')
        state_dict = torch.load(params_pth)
        geometry.load_state_dict(state_dict['geometry'])
        mat.load_state_dict(state_dict['material'])
        lgt.load_state_dict(state_dict['light'])

    # ==============================================================================================
    # Train
    # ==============================================================================================

    geometry, mat = optimize_mesh(glctx, geometry, mat, lgt, dataset_train, dataset_validate,
                                  FLAGS, optimize_light=FLAGS.learn_light)

    # ==============================================================================================
    # Validation
    # ==============================================================================================
    if FLAGS.local_rank == 0 and FLAGS.validate:
        validate(glctx, geometry, mat, lgt, dataset_validate, os.path.join(FLAGS.out_dir, "test"), FLAGS)
