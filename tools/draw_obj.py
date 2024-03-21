import open3d as o3d
import uuid
import os

mesh_file_path = r'D:\cvpr2024\mesh\ernerf\english.obj'  # 替换为你的三维模型文件路径

def render_obj(mesh_file_path, json_file, save_name):
    mesh = o3d.io.read_triangle_mesh(mesh_file_path)
    mesh.compute_vertex_normals()
    # mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=1000)
    # mesh.paint_uniform_color([0.5, 0.5, 0.5])
    # line_set = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    # line_set.paint_uniform_color([0, 0, 0])  # 设置线条颜色为黑色
    vis = o3d.visualization.Visualizer()

    vis.create_window(visible=True)
    vis.add_geometry(mesh)
    # vis.add_geometry(line_set)
    if json_file is not None:
        ctr = vis.get_view_control()
        param = o3d.io.read_pinhole_camera_parameters(json_file)
        ctr.convert_from_pinhole_camera_parameters(param)
    # vis.get_render_option().load_from_json('render.json')
    # vis.run()

    vis.poll_events()
    vis.update_renderer()

    # random_filename = str(uuid.uuid4())

    vis.capture_screen_image(save_name)
    vis.destroy_window()
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(r'D:\cvpr2024\mesh\render3.json', param)
    # vis.update_renderer()


if __name__ == '__main__':
    import glob
    for file_path in (glob.glob(os.path.join(r'D:\cvpr2024\mesh\dyntet', '*.obj'))):
        if "may" not in file_path:
            continue
        for config in (glob.glob(os.path.join(r'D:\cvpr2024\mesh\render_config_3dmm', '*.json'))):
            config_name = os.path.basename(config)[:-4]
            save_path = file_path[:-4] + '_' + config_name + '.png'
            render_obj(file_path, config, save_path)
            # break



