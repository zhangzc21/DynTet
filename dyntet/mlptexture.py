# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch
import tinycudann as tcnn
import numpy as np
from torch import nn
from render import material
#######################################################################################################################################################
# Small MLP using PyTorch primitives, internal helper class
#######################################################################################################################################################

class _MLP(torch.nn.Module):
    def __init__(self, cfg, loss_scale=1.0):
        super(_MLP, self).__init__()
        self.loss_scale = loss_scale
        net = (torch.nn.Linear(cfg['n_input_dims'], cfg['n_neurons'], bias=False), torch.nn.ReLU())
        for i in range(cfg['n_hidden_layers'] - 1):
            net = net + (torch.nn.Linear(cfg['n_neurons'], cfg['n_neurons'], bias=False), torch.nn.ReLU())
        net = net + (torch.nn.Linear(cfg['n_neurons'], cfg['n_output_dims'], bias=False),)
        self.net = torch.nn.Sequential(*net).cuda()

        self.net.apply(self._init_weights)

        if self.loss_scale != 1.0:
            self.net.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] * self.loss_scale,))

    def forward(self, x):
        return self.net(x.to(torch.float32))

    @staticmethod
    def _init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0.0)


#######################################################################################################################################################
# Outward visible MLP class
#######################################################################################################################################################

def params_num(net):
    return sum(param.numel() for param in net.parameters())


class MLPTexture3D_cond(torch.nn.Module):
    def __init__(self, AABB, channels=3, internal_dims=256, hidden=5, min_max=None):
        super(MLPTexture3D_cond, self).__init__()

        self.channels = channels
        self.internal_dims = internal_dims
        self.AABB = AABB
        self.min_max = min_max

        gradient_scaling = 1.0

        gradient_scaling = 128
        desired_resolution = 2048*2
        base_grid_resolution = 16
        num_levels = 16
        per_level_scale = np.exp(np.log(desired_resolution / base_grid_resolution) / (num_levels - 1))
        enc_cfg = {
            "otype": "TiledGrid",
            "n_levels": num_levels,
            "n_features_per_level": 2,
            "log2_hashmap_size": 19,
            "base_resolution": base_grid_resolution,
            "per_level_scale": per_level_scale
        }
        self.encoder = tcnn.Encoding(3, enc_cfg)
        self.encoder.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] / gradient_scaling,))

        self.cond_feature = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256)
        )

        self.vert_feature = nn.Sequential(
            nn.Linear(self.encoder.n_output_dims, 256),
        )
        # Setup MLP
        mlp_cfg = {
            "n_input_dims": 256,
            "n_output_dims": self.channels,
            "n_hidden_layers": hidden,
            "n_neurons": self.internal_dims
        }
        self.net = _MLP(mlp_cfg, gradient_scaling)
        print('rgb net params:', params_num(self))

    # Sample texture at a given location
    def sample(self, texc, cond, **kwargs):
        '''
        :param texc: [B,V,3] 3D coordinates
        :param cond: [B,C] 3DMM features
        '''
        cond = self.cond_feature(cond)
        _cond = cond[:, None].expand(cond.shape[0], texc.shape[1] if texc.ndim == 3 else texc.shape[1]*texc.shape[2],
                                     cond.shape[-1]).reshape(-1,cond.shape[-1])  # [BV, 32]

        _texc = (texc.view(-1, 3) - self.AABB[0][None, ...]) / (self.AABB[1][None, ...] - self.AABB[0][None, ...])
        _texc = torch.clamp(_texc, min=0, max=1)
        p_enc = self.vert_feature(self.encoder(_texc.contiguous()).float())
        p_enc = p_enc + _cond
        out = self.net.forward(p_enc)

        # Sigmoid limit and scale to the allowed range
        out = torch.sigmoid(out) * (self.min_max[1][None, :] - self.min_max[0][None, :]) + self.min_max[0][None, :]

        return out.view(*texc.shape[:-1], self.channels)  # Remap to [n, h, w, c]

    # In-place clamp with no derivative to make sure values are in valid range after training
    def clamp_(self):
        pass

    def cleanup(self):
        tcnn.free_temporary_memory()



def initial_guess_material(geometry, mlp, FLAGS, init_mat=None, internal_dims = 256):
    kd_min, kd_max = torch.tensor(FLAGS.kd_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.kd_max,
                                                                                                  dtype=torch.float32,
                                                                                                  device='cuda')
    ks_min, ks_max = torch.tensor(FLAGS.ks_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.ks_max,
                                                                                                  dtype=torch.float32,
                                                                                                  device='cuda')
    nrm_min, nrm_max = torch.tensor(FLAGS.nrm_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.nrm_max,
                                                                                                     dtype=torch.float32,
                                                                                                     device='cuda')
    if mlp:
        mlp_min = torch.cat((kd_min[0:3], ks_min, nrm_min), dim=0)
        mlp_max = torch.cat((kd_max[0:3], ks_max, nrm_max), dim=0)
        mlp_map_opt = MLPTexture3D_cond(geometry.getAABB(), channels=9, internal_dims = internal_dims, min_max=[mlp_min, mlp_max]).cuda()
        mat = material.Material({'kd_ks_normal': mlp_map_opt})
    else:
        assert 'error'

    if init_mat is not None:
        mat['bsdf'] = init_mat['bsdf']
    else:
        mat['bsdf'] = 'pbr'

    return mat