import os
import cv2
import numpy as np
import torch

from data_utils.face_tracking.facemodel import Face_3DMM
from data_utils.face_tracking.util import *
from data_utils.face_tracking.render_3dmm import Render_3DMM
import json
from tqdm import tqdm, trange
from torch.utils.data import Dataset

import time
import face_alignment
import open3d as o3d
from common_utils import Timer

import imageio


@torch.no_grad()
def visualize_lmk(video_file):
    fa2d = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input = False, device='cuda')
    fa3d = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input = False, device='cuda')

    os.makedirs('tools/viz_lmk', exist_ok = True)
    reader = imageio.get_reader(video_file)
    writer = imageio.get_writer(os.path.join('tools/viz_lmk', os.path.basename(video_file)), fps = 25, bitrate = '1M')

    preds = []
    for i, frame in tqdm(enumerate(reader)):

        pred = np.array(fa2d.get_landmarks(frame))[0,:,:2]
        # preds.append(pred)

        for point in pred:
            cv2.circle(frame, tuple(np.rint(point).astype(np.int).tolist()), 3,
                           (0, 0, 255), 1)

        pred = np.array(fa3d.get_landmarks(frame))[0,:,:2]
        # preds.append(pred)

        for point in pred:
            cv2.circle(frame, tuple(np.rint(point).astype(np.int).tolist()), 3,
                       (255, 0, 0), 1)

        writer.append_data(np.clip(frame, 0, 255).astype(np.uint8))  # BGR->RGB

        if i == 1024:
            break
    writer.close()


if __name__ == '__main__':
    visualize_lmk("/data2/zzc/nvdiffrec/data/video/May.mp4")
