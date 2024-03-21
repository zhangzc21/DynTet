# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import os
from typing import Iterator

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter

from render import mesh
from render import light

from geometry.network import get_rank, get_freq_embedder, MLP
from easydict import EasyDict as edict
import itertools
from render import util
import nvdiffrast.torch as dr
from render import renderutils as ru
import lpips
import torchvision.ops as ops
import trimesh
from tools import common_utils
from data_utils.face_3dmm import Face_3DMM
from geometry.dmtet import DMTet, sdf_reg_loss
###############################################################################
# Marching tetrahedrons implementation (differentiable), adapted from
# https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/ops/conversions/tetmesh.py
###############################################################################


LPIPS = lpips.LPIPS(net = 'alex').cuda()
model_3dmm = Face_3DMM(os.path.join('data_utils', 'face_tracking', '3DMM'), 100, 79, 100, 34650).cuda()
# exp_open_mouth = torch.from_numpy(np.load('data/open_mouth_3dmm.npy')).cuda()


def params_num(net):
    return sum(param.numel() for param in net.parameters())

def enlarge_mask(mask, dilation_radius = 9):
    '''
    :param mask: (B H W 1)
    :param dilation_radius: int
    :return: (B H W 1)
    '''
    mask = mask.permute(0, 3, 1, 2)
    dilation_kernel_size = 2 * dilation_radius + 1
    dilation_kernel = torch.ones((1, 1, dilation_kernel_size, dilation_kernel_size)).to(mask)
    dilated_mask = torch.nn.functional.conv2d(mask, dilation_kernel, padding = dilation_radius)
    dilated_mask = (dilated_mask > 0).float()
    return dilated_mask.permute(0, 2, 3, 1)


def photo_loss(pred_img, gt_img, img_mask):
    loss = torch.sum(torch.square(pred_img - gt_img), 3) * img_mask
    loss = torch.sum(loss, dim = (1, 2)) / torch.sum(img_mask, dim = (1, 2))
    loss = torch.mean(loss)
    return loss


def mse_loss_in_boxes(tensor1, tensor2, boxes):
    '''
    ROI loss for two images
    :param tensor1: (B, C, H, W)
    :param tensor2: (B, C, H, W)
    :param boxes: (B, 4)
    :return: loss
    '''

    boxes = boxes.float()
    box_width = (boxes[:, 2] - boxes[:, 0] + 1) // 2
    box_height = (boxes[:, 3] - boxes[:, 1] + 1) // 2

    # Find the maximum width and height of the boxes for ROI pooling
    max_box_width = torch.max(box_width)
    max_box_height = torch.max(box_height)

    # Calculate the pooled size for ROI pooling
    pooled_height = int(max_box_height.item())
    pooled_width = int(max_box_width.item())

    # Combine box coordinates into a single tensor with shape (B, 4)
    combined_boxes = torch.stack([boxes[:, 0], boxes[:, 1], boxes[:, 0] + max_box_width, boxes[:, 1] + max_box_height],
                                 dim = 1).float()
    combined_boxes = torch.split(combined_boxes, 1)

    # Apply ROI pooling to tensor1 and tensor2
    pooled_tensor1 = ops.roi_pool(tensor1, combined_boxes, output_size = (pooled_height, pooled_width))
    pooled_tensor2 = ops.roi_pool(tensor2, combined_boxes, output_size = (pooled_height, pooled_width))

    # MSE loss
    loss = torch.mean((pooled_tensor1 - pooled_tensor2) ** 2)

    # LPIPS loss
    target_height = int(np.exp2(max(int(np.log2(pooled_height - 1)) + 1, 5)))
    target_width = int(np.exp2(max(int(np.log2(pooled_width - 1)) + 1, 5)))
    padding_h = max(0, (target_height - pooled_tensor1.shape[-2] + 1) // 2)
    padding_w = max(0, (target_width - pooled_tensor2.shape[-1] + 1) // 2)

    if padding_w or padding_h:
        pooled_tensor1 = torch.nn.functional.pad(pooled_tensor1, (padding_w, padding_w, padding_h, padding_h))
        pooled_tensor2 = torch.nn.functional.pad(pooled_tensor2, (padding_w, padding_w, padding_h, padding_h))

    loss = loss + 0.1 * LPIPS(pooled_tensor1, pooled_tensor2, normalize = True).mean()

    return loss


def truncated_mse(a, b, t = 0.1):
    return torch.nn.functional.mse_loss(torch.clamp(a, -t, t), torch.clamp(b, -t, t))


class Conv3DMM(nn.Module):
    '''
    Convolution smoothing for 3DMM coefficients
    '''
    def __init__(self, coeff_nc = 70, descriptor_nc = 256, layer = 3):
        super(Conv3DMM, self).__init__()

        self.layer = layer
        nonlinearity = nn.LeakyReLU(0.1)

        self.first = nn.Sequential(
            torch.nn.Conv1d(coeff_nc, descriptor_nc, kernel_size = 7, padding = 0, bias = True))

        for i in range(layer):
            net = nn.Sequential(nonlinearity,
                                torch.nn.Conv1d(descriptor_nc, descriptor_nc, kernel_size = 3, padding = 0,
                                                dilation = 3))
            setattr(self, 'encoder' + str(i), net)

        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.output_nc = descriptor_nc

    def forward(self, input_3dmm):
        out = self.first(input_3dmm)
        for i in range(self.layer):
            model = getattr(self, 'encoder' + str(i))
            out = model(out) + out[:, :, 3:-3]
        out = self.pooling(out)
        return out.reshape(len(out), -1)


class SimpleMLP(nn.Module):
    def __init__(self, config: dict, aabb = None):
        super().__init__()

        self.AABB = aabb
        self.pos_enc = get_freq_embedder(multires = 6)
        ################
        self.n_neurons = config["n_neurons"]

        self.to_sdf = MLP(dim_in = self.pos_enc.n_output_dims,
                          dim_out = 2,
                          dim_hidden = self.n_neurons,
                          num_layers = config["to_sdf_num_layers"]) # get sdf and elastic score
        self.delta_feat = MLP(dim_in = self.pos_enc.n_output_dims,
                              dim_out = self.n_neurons,
                              dim_hidden = self.n_neurons,
                              num_layers = 1) # get coordinate feature
        self.cond_feat = MLP(dim_in = 256,
                             dim_out = self.n_neurons,
                             dim_hidden = self.n_neurons,
                             num_layers = 1) # get 3dmm feature
        self.to_delta = MLP(dim_in = self.n_neurons,
                            dim_out = 3,
                            dim_hidden = self.n_neurons,
                            num_layers = config["to_delta_num_layers"]) # get offset vector

        print(f'params:to_sdf {params_num(self.to_sdf)}, delta_feat {params_num(self.delta_feat)}, cond_feat {params_num(self.cond_feat)}, to_delta {params_num(self.to_delta)}')

    def get_hash_encode(self, x, hash_encode):
        shape = x.shape
        x = (x - self.AABB[0][None, ...]) / (self.AABB[1][None, ...] - self.AABB[0][None, ...])
        x_feature = hash_encode(x.view(-1, shape[-1]))
        x_feature = x_feature.reshape(*shape[:-1], -1)
        return x_feature

    def predict_sdf(self, x, **kwargs):
        '''
       predicting sdf values and elastic scores of the tetrahedral grid
       :param x: tetrahedral grid (len_v, 1)
       :param kwargs: invalid params
       :return: sdf values, elastic scores
       '''
        with torch.cuda.amp.autocast(enabled = False):
            feat = self.get_hash_encode(x, self.pos_enc)
            out = self.to_sdf(feat)
            sdf = out[..., :1]
            scale = torch.sigmoid(out[..., 1:]) / 2
            self.scale = scale
        return sdf, scale

    def get_deformation_feature(self, x, cond, **kwargs):
        '''
        embedding coordinates into deformation features, conditioned by 3DMM feature
        :param x: tetrahedral grid (len_v, 1)
        :param cond: 3dmm feature from Conv3DMMM (batch, 256)
        :param kwargs: invalid params
        :return: deformation features (batch, len_v, channel)
        '''

        with torch.cuda.amp.autocast(enabled = False):
            x_feature = self.get_hash_encode(x, self.pos_enc)
            if x_feature.ndim == 2:
                x_feature = x_feature.unsqueeze(0)

            x_feature_ = self.delta_feat(x_feature)
            cond_feature = self.cond_feat(cond)  # (batch, c)
            cond_feature = cond_feature.unsqueeze(1).expand(len(cond), x_feature_.shape[-2], -1)  #
            delta_feature = x_feature_.expand(len(cond), -1, -1) + cond_feature  # torch.cat([, cond_feature], dim=-1)

        return delta_feature

    def predict_delta(self, x, cond, valid_pt_idx = None, use_elastic = False, **kwargs):
        '''
        predicting offset vectors to drive the tetrahedral grid
        :param x: tetrahedral grid (len_v, 1)
        :param cond: 3dmm feature from Conv3DMMM (batch, 256)
        :param valid_pt_idx: a mask labelling valid points
        :param kwargs: invalid params
        :return: (batch, len_v, 3)
        '''

        scale = self.scale
        if valid_pt_idx is not None:
            scale = scale[valid_pt_idx]
            x = x[valid_pt_idx]

        with torch.cuda.amp.autocast(enabled = False):
            deformed_feature = self.get_deformation_feature(x, cond)
            delta = torch.tanh(self.to_delta(deformed_feature))
        if use_elastic:
            return delta  # * torch.sigmoid(scale)
        else:
            return delta * scale



class DMTetGeometry(torch.nn.Module):
    '''
    DMTetGeometry for DynTet
    '''
    def __init__(self, grid_res, scale, FLAGS, use_subdivison = False):
        super(DMTetGeometry, self).__init__()

        self.FLAGS = FLAGS
        self.grid_res = grid_res
        self.marching_tets = DMTet()
        self.use_subdivison = use_subdivison
        self.test_prepare_flag = False

        tets = np.load('data/tets/{}_tets.npz'.format(self.grid_res))

        self.verts = torch.tensor(tets['vertices'], dtype = torch.float32, device = 'cuda') * scale
        self.indices = torch.tensor(tets['indices'], dtype = torch.long, device = 'cuda')
        self.generate_edges()
        self.AABB = self.getAABB()

        self.conv3dmm = Conv3DMM().to('cuda')
        if True: # load weight from pretrained model. Random initialization is also fine
            state_dict = torch.load('data/Conv3DMM.pkl') #
            new_state_dict = {key.replace('mapping_3DMM.', ''): state_dict[key] for key in state_dict.keys() if "mapping_3DMM" in key}
            self.conv3dmm.load_state_dict(new_state_dict)
        self.conv3dmm.requires_grad = True

        mlp_network_config = edict({
            "n_neurons": 128,
            "to_sdf_num_layers": 3,
            "to_delta_num_layers": 4,
        })

        self.sdf_net = SimpleMLP(mlp_network_config, aabb = self.AABB).to('cuda')

    def initialize_shape(self, shape_init = "ellipsoid", shape_init_params = 0.8) -> None:
        '''
        Initializing the sdf attributes of the tetrahedral grid
        :param shape_init: face, ellipsoid, sphere
        '''
        if shape_init is None:
            return
        elif shape_init == 'face':
            face_mesh = trimesh.load("data/face.obj")
            face_vert = torch.from_numpy(face_mesh.vertices)[None].float().cuda()
            from pysdf import SDF
            f = SDF(face_mesh.vertices, face_mesh.faces)
            sdf = -f(self.verts.cpu().numpy())
            sdf = torch.from_numpy(sdf).float().cuda()[:,None]

        # Initialize SDF to a given shape when no weights are provided or force_shape_init is True
        optim = torch.optim.Adam(itertools.chain(self.parameters()), lr = 1e-3)
        from tqdm import tqdm
        pbar = tqdm(
            range(1000),
            desc = f"Set SDF to a(n) {shape_init}:",
            disable = get_rank() != 0,
        )
        for _ in pbar:
            if shape_init == "ellipsoid":
                points_rand = (
                        (torch.rand((10000, 3), dtype = torch.float32).to('cuda')) * (
                        self.verts.max(dim = 0).values[None, ...] - self.verts.min(dim = 0).values[None, ...]) +
                        self.verts.min(dim = 0).values[None, ...]
                )
                size = torch.as_tensor(shape_init_params).to(points_rand)
                sdf_gt = ((points_rand / size) ** 2).sum(
                    dim = -1, keepdim = True
                ).sqrt() - 1.0  # pseudo signed distance of an ellipsoid
            elif shape_init == "sphere":
                radius = shape_init_params
                sdf_gt = (points_rand ** 2).sum(dim = -1, keepdim = True).sqrt() - radius
            elif shape_init == "face":
                points_rand = self.verts
                sdf_gt = sdf

            else:
                raise ValueError(
                    f"Unknown shape initialization type: {self.cfg.shape_init}"
                )

            sdf_pred, _ = self.sdf_net.predict_sdf(points_rand)

            loss = torch.nn.functional.mse_loss(sdf_pred, sdf_gt)
            pbar.set_postfix(dict(loss = float(loss)))
            pbar.update()
            optim.zero_grad()
            loss.backward()
            optim.step()

    def generate_edges(self):
        with torch.no_grad():
            edges = torch.tensor([0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], dtype = torch.long, device = "cuda")
            all_edges = self.indices[:, edges].reshape(-1, 2)
            all_edges_sorted = torch.sort(all_edges, dim = 1)[0]
            self.all_edges = torch.unique(all_edges_sorted, dim = 0)

    @torch.no_grad()
    def getAABB(self):
        return torch.min(self.verts, dim = 0).values, torch.max(self.verts, dim = 0).values

    def get_3dmm_sdf(self, id, exp):
        '''
        Generating (points, sdf values) data according to 3dmm mesh
        :param id: 3dmm id coefficients, (B, c1)
        :param exp: 3dmm exp coefficients, (B, c2)
        '''
        face_vert = model_3dmm.forward_geo(id, exp)  # [B, V, 3]
        face_norm = model_3dmm.compute_normal(face_vert)  # [B, V,3]
        dist = 0.05 * (torch.randn(1, *face_vert.shape[:2], 1)).cuda()
        face_vert_perturb = face_vert[None, ...] + dist * (face_norm[None, ...])  # [b, B, V, 3]
        face_vert_perturb = face_vert_perturb.transpose(1, 0)  # [B, b,  V, 3]
        dist = dist.transpose(1, 0)  # [B, b, V, 1]

        return face_vert, face_norm, face_vert_perturb, dist

    def get_dynamic_mesh(self, material, cond, deformation = True, target = None):
        '''
        Leveraging Marching Tetrahedra to decode triangular mesh from tetrahedral representation
        :param material:
        :param cond: 3dmm coefficients (B, C)
        :param deformation: bool, using offset vectors or not
        :param target: training data
        '''

        cond = self.conv3dmm(cond)
        if target is not None:
            target['exp_conv'] = cond

        self.sdf, scale = self.sdf_net.predict_sdf(self.verts)

        faces, edges_to_interp_sdf, interp_v, uvs, uv_idx = self.marching_tets.get_triangle(self.sdf, self.indices)

        valid_pt_idx = torch.unique(interp_v)
        cond = torch.cat([cond, torch.zeros_like(cond)[[0]]], dim = 0)
        batch = len(cond)

        delta_value = torch.zeros(batch, *self.verts.shape).to(self.verts.device)
        if deformation:
            delta_value[:, valid_pt_idx] = self.sdf_net.predict_delta(self.verts, cond = cond,
                                                                           valid_pt_idx = valid_pt_idx, use_elastic = self.FLAGS.use_elastic)
        deformed_tet = self.verts + delta_value

        verts = self.marching_tets.get_vert(deformed_tet, edges_to_interp_sdf, interp_v)

        if self.use_subdivison:
            verts, faces = common_utils.subdivide_trianglemesh(verts, faces, 1)

        normals = auto_normals(verts, faces)
        verts, base_model_verts = verts[:-1], verts[-1:]
        normals, base_model_normals = normals[:-1], normals[-1:]

        Mesh = mesh.Mesh(v_pos = verts, t_pos_idx = faces,
                         v_nrm = normals, t_nrm_idx = faces,
                         material = material)
        Mesh.scale = self.marching_tets.get_vert(self.sdf_net.scale[None].expand(len(verts), -1, 3),
                                                 edges_to_interp_sdf, interp_v)
        Mesh.base_model_verts = base_model_verts
        Mesh.base_model_normals = base_model_normals

        return Mesh

    def predict(self, glctx, target, lgt, opt_material, bsdf = None, deformation = True):
        '''
        :param glctx: rendering utility
        :param target: Dict including a batch of training data
        :param lgt: trainable lighting map
        :param opt_material: trainable material MLP
        :param bsdf: bsdf type
        :param deformation: deformation
        '''
        opt_mesh = self.get_dynamic_mesh(opt_material, target['exp'], deformation, target)
        self.opt_mesh = opt_mesh
        out = render_dynamic_mesh(glctx, opt_mesh, target['mvp'], target['campos'], lgt, target['resolution'],
                                  spp = target['spp'],
                                  msaa = True, background = target['background'], bsdf = bsdf, target = target)
        return out

    def tick(self, glctx, target, lgt, opt_material, iteration):
        '''
        One-iteration training process
        :param glctx: rendering utility
        :param target: Dict including a batch of training data
        :param lgt: trainable lighting map
        :param opt_material: trainable material MLP
        :param iteration: Current iteration number
        :return: loss dict
        '''

        # ==============================================================================================
        #  Render optimizable object with identical conditions
        # ==============================================================================================
        buffers = self.predict(glctx, target, lgt, opt_material)

        # ==============================================================================================
        #  Compute loss
        # ==============================================================================================
        t_iter = iteration / self.FLAGS.iter

        # Image-space loss
        full_image = target['img'][..., :3]
        silhouette_loss = torch.nn.functional.mse_loss(buffers['shaded'][..., 3:], target['head_mask'])
        img_loss = photo_loss(buffers['shaded'][..., 0:3], full_image, enlarge_mask(target['head_mask'])[..., 0])
        lpips_loss = LPIPS(buffers['shaded'][..., 0:3].permute(0, 3, 1, 2), full_image.permute(0, 3, 1, 2),
                           normalize = True).mean()

        # Region loss of mouth and eyes
        region_loss = 0
        for k in range(target['rect'].shape[1]):
            region_loss += mse_loss_in_boxes(buffers['shaded'][..., :3].permute(0, 3, 1, 2),
                                             full_image.permute(0, 3, 1, 2), target['rect'][:, k, :])

        # 3DMM-based losses
        with torch.no_grad():
            face_vert, facenorm, face_vert_perturb, dist = self.get_3dmm_sdf(target['id'][[0]],
                                                                             torch.zeros(1, 79).to(target['id']))
            face_vert_exp, _, _, _ = self.get_3dmm_sdf(target['id'], target['exp_gt'])

        ## normal distance loss
        face_vert_perturb = face_vert_perturb.reshape(-1, 3)  # (Vert, 3)
        dist = dist.reshape(-1, 1)  # (Vert, 1)
        sdf_pred, scale_pred = self.sdf_net.predict_sdf(face_vert_perturb)
        sdf3dmm_loss = truncated_mse(sdf_pred, dist)

        ## facial deformation loss
        delta_gt = face_vert_exp - face_vert
        delta_pred = self.sdf_net.predict_delta(face_vert_exp, target['exp_conv'],  use_elastic = self.FLAGS.use_elastic)
        deform3dmm_loss = torch.nn.functional.mse_loss(delta_pred, delta_gt)

        # adaptive scale
        scale_reg =  (torch.relu(scale_pred.view(-1) - target['scale'][0].view(-1))).abs().mean()

        # SDF regularizer from DMTet. Optional
        sdf_weight = self.FLAGS.sdf_regularizer - (self.FLAGS.sdf_regularizer - 0.01) * min(1.0, 4.0 * t_iter)
        reg_loss = sdf_reg_loss(self.sdf, self.all_edges).mean() * sdf_weight  # Dropoff to 0.01

        # Visibility regularizer from DMTet. Optional
        reg_loss += torch.mean(buffers['occlusion'][..., :-1] * buffers['occlusion'][..., -1:]) * 0.001 * min(1.0,
                                                                                                              iteration / 500)

        # Light white balance regularizer from DMTet. Optional
        reg_loss = reg_loss + lgt.regularizer() * 0.005

        # A depth constrained loss based on 3DMM. Optional
        depth_3dmm = torch.relu(self.opt_mesh.v_pos[..., -1].max(dim = 1)[0] - face_vert_exp[..., -1].max()).mean()

        losses = dict(
            silhouette_loss = silhouette_loss,
            img_loss = img_loss,
            reg_loss = reg_loss,
            lpips_loss = lpips_loss,
            region_loss = region_loss,
            geo_loss = sdf3dmm_loss,
            scale_reg = scale_reg,
            deform3dmm_loss = deform3dmm_loss,
            depth_3dmm = depth_3dmm,
        )
        return losses
    
    @torch.no_grad()
    def infer(self, glctx, target, lgt, material, bsdf = None, show_time = False):
        '''
        Inference process
        :param glctx: rendering utility
        :param target: test data dict, including 3dmm coefficients and poses
        :param lgt: pretrained lighting map
        :param material: pretrained material MLP
        :param bsdf: bsdf type
        :param show_time: bool, show the inference time of each inference
        :return: output dict
        '''

        if not self.test_prepare_flag:
            sdf, scale = self.sdf_net.predict_sdf(self.verts)
            faces, edges_to_interp_sdf, interp_v, uvs, uv_idx = self.marching_tets.get_triangle(sdf, self.indices)
            valid_pt_idx = torch.unique(interp_v)

            self.zero_cond = torch.zeros(1, 256).cuda()

            self.delta_value = torch.zeros(len(self.zero_cond), *self.verts.shape).to(self.verts.device)
            self.delta_value[:, valid_pt_idx] = self.sdf_net.predict_delta(
                self.verts, cond = self.zero_cond, valid_pt_idx = valid_pt_idx,  use_elastic = self.FLAGS.use_elastic)
            deformed_tet = self.verts + self.delta_value
            base_model_verts = self.marching_tets.get_vert(deformed_tet, edges_to_interp_sdf,
                                                           interp_v)
            base_model_normals = auto_normals(base_model_verts, faces)

            self.topology_info = dict(
                faces = faces,
                edges_to_interp_sdf = edges_to_interp_sdf,
                interp_v = interp_v,
                valid_pt_idx = valid_pt_idx,
                base_model_verts = base_model_verts,
                base_model_normals = base_model_normals
            )

            self.test_prepare_flag = True

        with common_utils.Timer('conv 3dmm', show = show_time):
            if target is not None:
                cond = self.conv3dmm(target['exp'])
            target['exp_conv'] = cond

        with common_utils.Timer('get mesh', show = show_time):
            self.delta_value = torch.zeros(len(cond), *self.verts.shape).to(self.verts.device)
            self.delta_value[:, self.topology_info['valid_pt_idx']] = \
                self.sdf_net.predict_delta(self.verts, cond = cond,
                                           valid_pt_idx =self.topology_info['valid_pt_idx'],
                                           use_elastic = self.FLAGS.use_elastic)  # (B, V, 3)

            deformed_tet = self.verts + self.delta_value
            verts = self.marching_tets.get_vert(deformed_tet, self.topology_info['edges_to_interp_sdf'],
                                                self.topology_info['interp_v'])

            normals = auto_normals(verts, self.topology_info['faces'])
            Mesh = mesh.Mesh(v_pos = verts, t_pos_idx = self.topology_info['faces'],
                             v_nrm = normals, t_nrm_idx = self.topology_info['faces'],
                             material = material)

            Mesh.scale = torch.zeros_like(verts)
            Mesh.base_model_verts = self.topology_info['base_model_verts']
            Mesh.base_model_normals = self.topology_info['base_model_normals']
            self.opt_mesh = Mesh

        with common_utils.Timer('render', show = show_time):
            out = render_dynamic_mesh(glctx, Mesh, target['mvp'], target['campos'], lgt, target['resolution'],
                                      spp = target['spp'],
                                      msaa = True, background = target['background'], bsdf = bsdf, target = target)

        return out

#########################################################################
#########################################################################
    # The following rendering code comes from the DMTet rendering.
    # We adapt the code to the dynamic meshes.
#########################################################################
#########################################################################

def render_dynamic_mesh(
        ctx,
        mesh,
        mtx_in,
        view_pos,
        lgt,
        resolution,
        spp = 1,
        num_layers = 1,
        msaa = False,
        background = None,
        bsdf = None,
        target = None,
):
    def prepare_input_vector(x):
        x = torch.tensor(x, dtype = torch.float32, device = 'cuda') if not torch.is_tensor(x) else x
        return x[:, None, None, :] if len(x.shape) == 2 else x

    def composite_buffer(key, layers, background, antialias):
        accum = background
        for buffers, rast in reversed(layers):
            alpha = (rast[..., -1:] > 0).float() * buffers[key][..., -1:]
            accum = torch.lerp(accum,
                               torch.cat((buffers[key][..., :-1], torch.ones_like(buffers[key][..., -1:])), dim = -1),
                               alpha)
            if antialias:
                accum = dr.antialias(accum.contiguous(), rast, v_pos_clip, mesh.t_pos_idx.int())
        return accum

    assert mesh.t_pos_idx.shape[0] > 0, "Got empty training triangle mesh (unrecoverable discontinuity)"
    assert background is None or (background.shape[1] == resolution[0] and background.shape[2] == resolution[1])

    full_res = [resolution[0] * spp, resolution[1] * spp]

    # Convert numpy arrays to torch tensors
    mtx_in = torch.tensor(mtx_in, dtype = torch.float32, device = 'cuda') if not torch.is_tensor(mtx_in) else mtx_in
    view_pos = prepare_input_vector(view_pos)

    # clip space transform
    v_pos_clip = ru.xfm_points(mesh.v_pos, mtx_in)

    # Render all layers front-to-back
    layers = []
    with dr.DepthPeeler(ctx, v_pos_clip, mesh.t_pos_idx.int(), full_res) as peeler:
        for _ in range(num_layers):
            rast, db = peeler.rasterize_next_layer()
            layers += [(render_layer_simple(rast, db, mesh, view_pos, lgt, resolution, spp, msaa, bsdf, target), rast)]

    # Setup background
    if background is not None:
        if spp > 1:
            background = util.scale_img_nhwc(background, full_res, mag = 'nearest', min = 'nearest')
        background = torch.cat((background, torch.zeros_like(background[..., 0:1])), dim = -1)
    else:
        background = torch.zeros(1, full_res[0], full_res[1], 4, dtype = torch.float32, device = 'cuda')

    # Composite layers front-to-back
    out_buffers = {}
    for key in layers[0][0].keys():
        if key == 'shaded':
            accum = composite_buffer(key, layers, background, True)
        else:
            accum = composite_buffer(key, layers, torch.zeros_like(layers[0][0][key]), False)

        # Downscale to framebuffer resolution. Use avg pooling
        out_buffers[key] = util.avg_pool_nhwc(accum, spp) if spp > 1 else accum

    return out_buffers


def interpolate(attr, rast, attr_idx, rast_db = None):
    return dr.interpolate(attr.contiguous(), rast, attr_idx, rast_db = rast_db,
                          diff_attrs = None if rast_db is None else 'all')


def render_layer_simple(
        rast,
        rast_deriv,
        mesh,
        view_pos,
        lgt,
        resolution,
        spp,
        msaa,
        bsdf,
        target,
        lighting = False, # It is better to set lighting = False.
):
    full_res = [resolution[0] * spp, resolution[1] * spp]

    ################################################################################
    # Rasterize
    ################################################################################

    # Scale down to shading resolution when MSAA is enabled, otherwise shade at full resolution
    if spp > 1 and msaa:
        rast_out_s = util.scale_img_nhwc(rast, resolution, mag = 'nearest', min = 'nearest')
        rast_out_deriv_s = util.scale_img_nhwc(rast_deriv, resolution, mag = 'nearest', min = 'nearest') * spp
    else:
        rast_out_s = rast
        rast_out_deriv_s = rast_deriv

    ################################################################################
    # Interpolate attributes
    ################################################################################

    # Interpolate world space position
    # gb_attr, _ = interpolate(torch.cat([mesh.v_pos, mesh.v_nrm, mesh.v_kd, mesh.v_ks], dim=-1), rast_out_s,
    #                          mesh.t_pos_idx.int())
    # gb_pos, gb_normal, kd, ks = gb_attr[..., :3], gb_attr[..., 3:6], gb_attr[..., 6:9], gb_attr[..., 9:]

    gb_attr, _ = interpolate(
        torch.cat([mesh.v_pos, mesh.v_nrm, mesh.base_model_verts.expand_as(mesh.v_pos), mesh.scale], dim = -1),
        rast_out_s,
        mesh.t_pos_idx.int())
    gb_pos, gb_normal, scale = gb_attr[..., :3], gb_attr[..., 3:6], gb_attr[..., -3:]
    cano_pos = gb_attr[..., 6:9]

    all_tex = mesh.material['kd_ks_normal'].sample(cano_pos, target['exp_conv'])
    kd, ks, perturbed_nrm = all_tex[..., :-6], all_tex[..., -6:-3], all_tex[..., -3:]

    gb_normal = util.safe_normalize(gb_normal)
    depth = gb_pos[..., [-1]]
    alpha = kd[..., 3:4] if kd.shape[-1] == 4 else torch.ones_like(kd[..., 0:1])

    assert 'bsdf' in mesh.material or bsdf is not None, "Material must specify a BSDF type"
    bsdf = mesh.material['bsdf'] if bsdf is None else bsdf
    ir = torch.zeros_like(gb_pos)
    if bsdf == 'diffuse':
        if isinstance(lgt, light.EnvironmentLight):
            shaded_col = lgt.shade(gb_pos, gb_normal, kd, ks, view_pos, specular = True, light = lighting)
        else:
            assert False, "Invalid light type"
    elif bsdf == 'normal':
        shaded_col = (gb_normal + 1.0) * 0.5
    elif bsdf == 'kd':
        shaded_col = kd
    elif bsdf == 'ks':
        shaded_col = ks
    else:
        assert False, "Invalid BSDF '%s'" % bsdf

    buffers = {
        'shaded': torch.cat((shaded_col, alpha), dim = -1),
        'depth': torch.cat((depth, alpha), dim = -1),
        'occlusion': torch.cat((ks[..., :1], alpha), dim = -1),
        'scale': torch.cat((scale, alpha), dim = -1)
    }
    ################################################################################
    # Prepare output
    ################################################################################

    # Scale back up to visibility resolution if using MSAA
    if spp > 1 and msaa:
        for key in buffers.keys():
            buffers[key] = util.scale_img_nhwc(buffers[key], full_res, mag = 'nearest', min = 'nearest')

    # Return buffers
    return buffers


def auto_normals(v_pos, t_pos_idx):
    batch_size, num_vertices, _ = v_pos.shape

    # Reshape t_pos_idx to match the batch size
    t_pos_idx = t_pos_idx.view(-1, 3)
    num_faces = len(t_pos_idx)
    # Calculate face normals for each batch and triangle
    v0 = v_pos[:, t_pos_idx[:, 0], :]
    v1 = v_pos[:, t_pos_idx[:, 1], :]
    v2 = v_pos[:, t_pos_idx[:, 2], :]

    face_normals = torch.cross(v1 - v0, v2 - v0)

    # Splat face normals to vertices
    v_nrm = torch.zeros_like(v_pos)
    v_nrm.scatter_add_(1, t_pos_idx[:, 0][None, :, None].expand(batch_size, num_faces, 3), face_normals)
    v_nrm.scatter_add_(1, t_pos_idx[:, 1][None, :, None].expand(batch_size, num_faces, 3), face_normals)
    v_nrm.scatter_add_(1, t_pos_idx[:, 2][None, :, None].expand(batch_size, num_faces, 3), face_normals)

    # Normalize, replace zero (degenerated) normals with some default value
    v_nrm = torch.where(torch.norm(v_nrm, dim = -1, keepdim = True) > 1e-20, v_nrm,
                        torch.tensor([0.0, 0.0, 1.0], dtype = torch.float32, device = v_pos.device))
    length = util.length(v_nrm)
    v_nrm = v_nrm / length

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(v_nrm))

    return v_nrm


def compute_tangents(v_pos, t_pos_idx, v_tex, t_tex_idx, v_nrm, t_nrm_idx):
    batch_size, num_vertices, _ = v_pos.shape

    vn_idx = [None] * 3
    pos = [None] * 3
    tex = [None] * 3
    for i in range(0, 3):
        pos[i] = v_pos[:, t_pos_idx[:, i], :]
        tex[i] = v_tex[:, t_tex_idx[:, i], :]
        vn_idx[i] = t_nrm_idx[:, i]

    tangents = torch.zeros_like(v_nrm)
    tansum = torch.zeros_like(v_nrm)

    # Compute tangent space for each triangle
    uve1 = tex[1] - tex[0]
    uve2 = tex[2] - tex[0]
    pe1 = pos[1] - pos[0]
    pe2 = pos[2] - pos[0]

    nom = (pe1 * uve2[..., 1:2] - pe2 * uve1[..., 1:2])
    denom = (uve1[..., 0:1] * uve2[..., 1:2] - uve1[..., 1:2] * uve2[..., 0:1])

    # Avoid division by zero for degenerated texture coordinates
    tang = nom / torch.where(denom > 0.0, torch.clamp(denom, min = 1e-6), torch.clamp(denom, max = -1e-6))

    # Update all 3 vertices
    for i in range(0, 3):
        idx = vn_idx[i][None, :, None].expand(batch_size, -1, 3)
        tangents.scatter_add_(1, idx, tang)  # tangents[n_i] = tangents[n_i] + tang
        tansum.scatter_add_(1, idx, torch.ones_like(tang))  # tansum[n_i] = tansum[n_i] + 1
    tangents = tangents / tansum

    # Normalize and make sure tangent is perpendicular to normal
    tangents = util.safe_normalize(tangents)
    tangents = util.safe_normalize(tangents - util.dot(tangents, v_nrm) * v_nrm)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(tangents))

    return tangents
