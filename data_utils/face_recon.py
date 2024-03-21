import os
import cv2
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.io import savemat

import math
import torch

from models import create_model
from options.test_options import TestOptions
from util.preprocess import align_img
from util.load_mats import load_lm3d
from util.util import mkdirs, tensor2im, save_image

from skimage import transform as T
import glob
import natsort


def get_data_path(root):
    filenames = natsort.natsorted(glob.glob(os.path.join(root, '*.jpg')))
    keypoint_filenames = natsort.natsorted(glob.glob(os.path.join(root, '*.lms')))

    print(len(filenames), len(keypoint_filenames))
    assert len(filenames) == len(keypoint_filenames)

    return filenames, keypoint_filenames

class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, filenames, txt_filenames, bfm_folder):
        self.filenames = filenames
        self.txt_filenames = txt_filenames
        self.lm3d_std = load_lm3d(bfm_folder)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        txt_filename = self.txt_filenames[index]
        frame = self.read_image(filename)
        lm = np.loadtxt(txt_filename).astype(np.float32)
        lm = lm.reshape([-1, 2])
        out_img, _, out_trans_param, warp \
            = self.image_transform(frame, lm)
        return {
            'imgs': out_img,
            'trans_param': out_trans_param,
            'filename': filename,
            'warping': warp,
        }

    def read_image(self, filename):
        frame = cv2.imread(filename)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        return frame

    def image_transform(self, images, lm):
        W, H = images.size
        if np.mean(lm) == -1:
            lm = (self.lm3d_std[:, :2] + 1) / 2.
            lm = np.concatenate(
                [lm[:, :1] * W, lm[:, 1:2] * H], 1
            )
        else:
            lm[:, -1] = H - 1 - lm[:, -1]

        lm_old = lm.copy().astype(np.float64)
        trans_params, img, lm, _ = align_img(images, lm, self.lm3d_std)
        tform3 = T.ProjectiveTransform()
        assert tform3.estimate(src = lm, dst = lm_old)
        M = tform3.params[:2].copy()
        warp = np.eye(4)
        warp[:2, :2] = M[:2, :2]
        warp[:2:, -1:] = M[:2, -1:]
        warp = torch.tensor(warp.astype(np.float32))

        img = torch.tensor(np.array(img) / 255., dtype = torch.float32).permute(2, 0, 1)
        lm = torch.tensor(lm)
        trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)])
        trans_params = torch.tensor(trans_params.astype(np.float32))
        return img, lm, trans_params, warp


def main(opt, model):
    # import torch.multiprocessing
    # torch.multiprocessing.set_sharing_strategy('file_system')
    filenames, keypoint_filenames = get_data_path(opt.img_folder)
    dataset = ImagePathDataset(filenames, keypoint_filenames, opt.bfm_folder)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size = 1,  # can only set to one here!
        shuffle = False,
        drop_last = False,
        num_workers = 0,
    )

    pred_coeffs = []
    for data in tqdm(dataloader):

        # bypass existed files
        name = data['filename'][0].split('/')[-2:]

        data_input = {
            'imgs': data['imgs'],
        }
        model.set_input(data_input)
        model.test()
        pred_coeff = {key: model.pred_coeffs_dict[key].cpu().numpy() for key in model.pred_coeffs_dict}
        pred_coeff = np.concatenate([
            pred_coeff['id'],
            pred_coeff['exp'],
            pred_coeff['tex'],
            pred_coeff['angle'],
            pred_coeff['gamma'],
            pred_coeff['trans']], 1)
        pred_coeffs.append(pred_coeff)

    pred_coeffs = np.concatenate(pred_coeffs, 0)
    warping_params = data['warping'][0].cpu().numpy()
    pred_trans_params = data['trans_param'][0].cpu().numpy()
    name[-1] = os.path.splitext(name[-1])[0] + '.mat'

    # os.makedirs(os.path.join(opt.output_dir, name[-2]), exist_ok = True)
    save_path = os.path.join(os.path.dirname(opt.img_folder), 'face_recon3dmm.npy')
    np.save(save_path, pred_coeffs)


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    model = create_model(opt)
    model.setup(opt)
    model.device = 'cuda:0'
    model.parallelize()
    model.eval()

    main(opt, model)

