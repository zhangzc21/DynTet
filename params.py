import os
import argparse
import torch
import json

def get_FLAGS():
    parser = argparse.ArgumentParser(description = 'DynTet')
    parser.add_argument('--config', type = str, default = f'configs/obama.json', help = 'Config file')
    parser.add_argument('-i', '--iter', type = int, default = 5000)
    parser.add_argument('-b', '--batch', type = int, default = 4)
    parser.add_argument('-s', '--spp', type = int, default = 1)
    parser.add_argument('-l', '--layers', type = int, default = 1)
    parser.add_argument('-r', '--train-res', nargs = 2, type = int, default = [512, 512])
    parser.add_argument('-dr', '--display-res', type = int, default = None)
    parser.add_argument('-di', '--display-interval', type = int, default = 0)
    parser.add_argument('-si', '--save-interval', type = int, default = 1000)
    parser.add_argument('-lr', '--learning-rate', type = float, default = 0.01)
    parser.add_argument('-bg', '--background', default = 'target',
                        choices = ['black', 'white', 'checker', 'reference', 'target'])
    parser.add_argument('-o', '--out-dir', type = str, default = None)
    parser.add_argument('-rm', '--ref_mesh', type = str)
    parser.add_argument('-bm', '--base-mesh', type = str, default = None)
    parser.add_argument('--validate', type = bool, default = True)
    parser.add_argument('--resume', type = bool, default = False)
    # test use
    parser.add_argument('--drive_3dmm', type = str, default = None)
    parser.add_argument('--audio', type = str, default = None)

    FLAGS = parser.parse_args()

    FLAGS.mtl_override = None  # Override material of model
    FLAGS.dmtet_grid = 64  # Resolution of initial tet grid. We provide 64 and 128 resolution grids. Other resolutions can be generated with https://github.com/crawforddoran/quartet
    FLAGS.mesh_scale = 2.1  # Scale of tet grid box. Adjust to cover the model
    FLAGS.env_scale = 1.0  # Env map intensity multiplier
    FLAGS.envmap = None  # HDR environment probe
    FLAGS.display = None  # Conf validation window/display. E.g. [{"relight" : <path to envlight>}]
    FLAGS.camera_space_light = False  # Fixed light in camera space. This is needed for setups like ethiopian head where the scanned object rotates on a stand.
    FLAGS.lock_light = False  # Disable light optimization in the second pass
    FLAGS.lock_pos = False  # Disable vertex position optimization in the second pass
    FLAGS.sdf_regularizer = 0.2  # Weight for sdf regularizer (see paper for details)
    FLAGS.kd_min = [0.0, 0.0, 0.0, 0.0]  # Limits for kd
    FLAGS.kd_max = [1.0, 1.0, 1.0, 1.0]
    FLAGS.ks_min = [0.0, 0.08, 0.0]  # Limits for ks
    FLAGS.ks_max = [1.0, 1.0, 1.0]
    FLAGS.nrm_min = [-1.0, -1.0, 0.0]  # Limits for normal map
    FLAGS.nrm_max = [1.0, 1.0, 1.0]
    FLAGS.cam_near_far = [0.1, 1000.0]
    FLAGS.learn_light = True
    FLAGS.use_elastic = False # Whether using elastic score or not.

    FLAGS.data_range = [0, -1]
    FLAGS.offset = [0, 0, 0]
    FLAGS.pre_load = False # Pre-load entire dataset into memory for faster training

    FLAGS.local_rank = 0
    FLAGS.multi_gpu = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1

    if FLAGS.multi_gpu:
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = 'localhost'
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = '23456'

        FLAGS.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(FLAGS.local_rank)
        torch.distributed.init_process_group(backend = "nccl", init_method = "env://")

    if FLAGS.config is not None:
        data = json.load(open(FLAGS.config, 'r'))
        for key in data:
            FLAGS.__dict__[key] = data[key]

    if FLAGS.display_res is None:
        FLAGS.display_res = FLAGS.train_res
    if FLAGS.out_dir is None:
        FLAGS.out_dir = 'out/cube_%d' % (FLAGS.train_res)
    else:
        FLAGS.out_dir = 'out/' + FLAGS.out_dir

    if FLAGS.local_rank == 0:
        print("Config / Flags:")
        print("---------")
        for key in FLAGS.__dict__.keys():
            print(key, FLAGS.__dict__[key])
        print("---------")

    return FLAGS