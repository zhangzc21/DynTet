import trimesh
import open3d
import os
import open3d as o3d
import torch
import json
import shutil
import pathlib
import torchvision as tv
import time

def torch_to_trimesh_mesh(verts, faces):
    mesh = trimesh.Trimesh(vertices = verts.detach().cpu().numpy(), faces = faces.detach().cpu().numpy())
    return mesh


def save_mesh(verts, faces, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok = True)
    mesh = trimesh.Trimesh(vertices = verts.detach().cpu().numpy(), faces = faces.detach().cpu().numpy())
    trimesh.exchange.export.export_mesh(mesh, os.path.join(save_path))


def torch_to_open3d_mesh(verts, faces):
    verts_np = verts.cpu().detach().numpy()
    faces_np = faces.cpu().detach().numpy()
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts_np)
    mesh.triangles = o3d.utility.Vector3iVector(faces_np)
    return mesh


def open3d_mesh_to_image(mesh, path):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(mesh)
    vis.update_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(path)
    # depth = vis.capture_depth_float_buffer(do_render=False)
    vis.destroy_window()

@torch.no_grad()
def nearest_neighbors(point_3dmm, verts):
    def nn(a, b):
        distances = (a[:,None] - b[None,:]).square().sum(dim=-1)
        values, nearest_indices = torch.min(distances, dim = -1)

        return values, nearest_indices
    total_vert = len(point_3dmm)
    l = list(range(0, total_vert, 100))
    if l[-1] != total_vert:
        l.append(total_vert)

    indices = []
    values = []
    for s, e in zip(l[:-1], l[1:]):
        value, index = nn(point_3dmm[s:e], verts)
        values.append(value)
        indices.append(index)
    return torch.cat(values, dim = 0), torch.cat(indices, dim = 0)


@torch.no_grad()
def pairwise_min_distance(points):
    total_vert = len(points)
    l = list(range(0, total_vert, 100))
    if l[-1] != total_vert:
        l.append(total_vert)

    indices = []
    values = []
    for s,e in zip(l[:-1], l[1:]):
        dist = (points[s:e,None] - points[None,:]).square().sum(dim = -1)
        dist[:,s:e] += 10000 * torch.eye(e-s).to(dist.device)
        value, index = torch.min(dist, dim = -1)
        values.append(value)
        indices.append(index)
    return torch.cat(values), torch.cat(indices)


# from kaolin.ops.mesh.trianglemesh import _get_adj_verts, _get_alpha
# def subdivide_trianglemesh(vertices, faces, iterations, alpha=None):
#     """
#     code from kaolin, subtle modification for a batch of vertices
#     """
#     init_alpha = alpha
#     for i in range(iterations):
#         device = vertices.device
#         b, v, f = vertices.shape[0], vertices.shape[1], faces.shape[0]
#
#         edges_fx3x2 = faces[:, [[0, 1], [1, 2], [2, 0]]]
#         edges_fx3x2_sorted, _ = torch.sort(edges_fx3x2.reshape(edges_fx3x2.shape[0] * edges_fx3x2.shape[1], 2), -1)
#         all_edges_face_idx = torch.arange(edges_fx3x2.shape[0], device=device).unsqueeze(-1).expand(-1, 3).reshape(-1)
#         edges_ex2, inverse_indices, counts = torch.unique(
#             edges_fx3x2_sorted, dim=0, return_counts=True, return_inverse=True)
#
#         # To compute updated vertex positions, first compute alpha for each vertex
#         # TODO(cfujitsang): unify _get_adj_verts with adjacency_matrix
#         adj_sparse = _get_adj_verts(edges_ex2, v)
#         n = torch.sparse.sum(adj_sparse, 0).to_dense().view(-1, 1)
#         if init_alpha is None:
#             alpha = (_get_alpha(n) * n).unsqueeze(0)
#         if alpha.dim() == 2:
#             alpha = alpha.unsqueeze(-1)
#         alpha = alpha.expand(b, -1, -1)
#
#         adj_verts_sum = torch.bmm(torch.stack([adj_sparse]*b), vertices)
#         vertices_new = (1 - alpha) * vertices + alpha / n * adj_verts_sum
#
#         e = edges_ex2.shape[0]
#         edge_points = torch.zeros((b, e, 3), device=device)  # new point for every edge
#         edges_fx3 = inverse_indices.reshape(f, 3) + v
#         alpha_points = torch.zeros((b, e, 1), device=device)
#
#         mask_e = (counts == 2)
#
#         # edge points on boundary is computed as midpoint
#         if torch.sum(~mask_e) > 0:
#             edge_points[:, ~mask_e] += torch.mean(vertices[:,
#                                                   edges_ex2[~mask_e].reshape(-1), :].reshape(b, -1, 2, 3), 2)
#             alpha_points[:, ~mask_e] += torch.mean(alpha[:, edges_ex2[~mask_e].reshape(-1), :].reshape(b, -1, 2, 1), 2)
#
#         counts_f = counts[inverse_indices]
#         mask_f = (counts_f == 2)
#         group = inverse_indices[mask_f]
#         _, indices = torch.sort(group)
#         edges_grouped = all_edges_face_idx[mask_f][indices]
#         edges_face_idx = torch.stack([edges_grouped[::2], edges_grouped[1::2]], dim=-1)
#         e_ = edges_face_idx.shape[0]
#         edges_face = faces[edges_face_idx.reshape(-1), :].reshape(-1, 2, 3)
#         edges_vert = vertices[:, edges_face.reshape(-1), :].reshape(b, e_, 6, 3)
#         edges_vert = torch.cat([edges_vert, vertices[:, edges_ex2[mask_e].reshape(-1),
#                                             :].reshape(b, -1, 2, 3)], 2).mean(2)
#
#         alpha_vert = alpha[:, edges_face.reshape(-1), :].reshape(b, e_, 6, 1)
#         alpha_vert = torch.cat([alpha_vert, alpha[:, edges_ex2[mask_e].reshape(-1),
#                                             :].reshape(b, -1, 2, 1)], 2).mean(2)
#
#         edge_points[:, mask_e] += edges_vert
#         alpha_points[:, mask_e] += alpha_vert
#
#         alpha = torch.cat([alpha, alpha_points], 1)
#         vertices = torch.cat([vertices_new, edge_points], 1)
#         faces = torch.cat([faces, edges_fx3], 1)
#         faces = faces[:, [[1, 4, 3], [0, 3, 5], [2, 5, 4], [5, 3, 4]]].reshape(-1, 3)
#     return vertices, faces


def save_dict_to_json(data, file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    json_data = json.dumps(data, indent=4)

    # index = 0
    # while os.path.exists(file_path):
    #     index += 1
    #     base_name, ext = os.path.splitext(file_path)
    #     file_path = f"{base_name}_{index}{ext}"

    with open(file_path, 'w') as json_file:
        json_file.write(json_data)


def save_files_to_folder(file_list, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for file_path in file_list:
        file_name = os.path.basename(file_path)
        target_path = os.path.join(target_folder, file_name)
        shutil.copyfile(file_path, target_path)


def save_bchw_to_image(img, path, nrow = 1, range = '-1,1'):
    path = pathlib.Path(path)
    path.parent.mkdir(exist_ok = True, parents = True)
    assert range in ['0,1', '-1,1', '0,255']
    if range == '0,1':
        img = img
    if range == '-1,1':
        img = ((img + 1) / 2)
    if range == '0,255':
        img = img / 255
    img = img.clamp(0, 1)
    tv.utils.save_image(img, str(path), normalize = False, nrow = nrow)


class Timer:
    def __init__(self, name, show = True):
        self.name = name
        self.show = show

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        execution_time = end_time - self.start_time
        formatted_time = "{:.4f}".format(execution_time)
        if self.show:
            print(f"{self.name} costs {formatted_time}s")

