import torch
import torch.nn as nn
import numpy as np
import os


class Face_3DMM(nn.Module):
    def __init__(self, modelpath, id_dim, exp_dim, tex_dim, point_num):
        super(Face_3DMM, self).__init__()
        # id_dim = 100
        # exp_dim = 79
        # tex_dim = 100
        self.point_num = point_num
        DMM_info = np.load(
            os.path.join(modelpath, "3DMM_info.npy"), allow_pickle=True
        ).item()
        base_id = DMM_info["b_shape"][:id_dim, :]
        mu_id = DMM_info["mu_shape"]
        base_exp = DMM_info["b_exp"][:exp_dim, :]
        mu_exp = DMM_info["mu_exp"]
        mu = mu_id + mu_exp
        mu = mu.reshape(-1, 3)
        for i in range(3):
            mu[:, i] -= np.mean(mu[:, i])
        mu = mu.reshape(-1)
        self.base_id = torch.as_tensor(base_id).cuda() / 100000.0
        self.base_exp = torch.as_tensor(base_exp).cuda() / 100000.0
        self.mu = torch.as_tensor(mu).cuda() / 100000.0
        base_tex = DMM_info["b_tex"][:tex_dim, :]
        mu_tex = DMM_info["mu_tex"]
        self.base_tex = torch.as_tensor(base_tex).cuda()
        self.mu_tex = torch.as_tensor(mu_tex).cuda()
        sig_id = DMM_info["sig_shape"][:id_dim]
        sig_tex = DMM_info["sig_tex"][:tex_dim]
        sig_exp = DMM_info["sig_exp"][:exp_dim]
        self.sig_id = torch.as_tensor(sig_id).cuda()
        self.sig_tex = torch.as_tensor(sig_tex).cuda()
        self.sig_exp = torch.as_tensor(sig_exp).cuda()

        keys_info = np.load(
            os.path.join(modelpath, "keys_info.npy"), allow_pickle=True
        ).item()
        self.keyinds = torch.as_tensor(keys_info["keyinds"]).cuda()
        self.left_contours = torch.as_tensor(keys_info["left_contour"]).cuda()
        self.right_contours = torch.as_tensor(keys_info["right_contour"]).cuda()
        self.rigid_ids = torch.as_tensor(keys_info["rigid_ids"]).cuda()

        topo_info = np.load(os.path.join(modelpath, "topology_info.npy"), allow_pickle = True).item()

        self.tris = torch.as_tensor(topo_info["tris"]).cuda()
        self.vert_tris = torch.as_tensor(topo_info["vert_tris"]).cuda()


    def forward_geo(self, id_para, exp_para):
        id_para = id_para * self.sig_id
        exp_para = exp_para * self.sig_exp
        geometry = (
            torch.mm(id_para, self.base_id)
            + torch.mm(exp_para, self.base_exp)
            + self.mu
        )
        return geometry.reshape(-1, self.point_num, 3)

    def forward_tex(self, tex_para):
        tex_para = tex_para * self.sig_tex
        texture = torch.mm(tex_para, self.base_tex) + self.mu_tex
        return texture.reshape(-1, self.point_num, 3)

    def compute_normal(self, geometry):
        vert_1 = torch.index_select(geometry, 1, self.tris[:, 0])
        vert_2 = torch.index_select(geometry, 1, self.tris[:, 1])
        vert_3 = torch.index_select(geometry, 1, self.tris[:, 2])
        nnorm = torch.cross(vert_2 - vert_1, vert_3 - vert_1, 2)
        tri_normal = nn.functional.normalize(nnorm, dim=2)
        v_norm = tri_normal[:, self.vert_tris, :].sum(2)
        vert_normal = v_norm / v_norm.norm(dim=2).unsqueeze(2)
        return vert_normal
