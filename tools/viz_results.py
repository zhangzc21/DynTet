import os
import numpy as np
import imageio
import glob
import natsort
from tqdm import tqdm

import imageio

def video_to_gif(video_path, gif_path):
    video_reader = imageio.get_reader(video_path)
    gif_writer = imageio.get_writer(gif_path, mode='I', fps = 25)
    for frame in video_reader:
        gif_writer.append_data(frame)
    video_reader.close()
    gif_writer.close()

    print("Convert VIDEO to GIF：", gif_path)

for name in [
    '/home/math/data/dyntet/out/obama_batch8/test',
]:
    video_dir = os.path.join(name)
    video_name = os.path.join(name, 'video.mp4')
    image_list = natsort.natsorted(glob.glob(os.path.join(name, '*_opt.png')))
    video_writer = imageio.get_writer(video_name, mode= 'I', fps=25, codec= 'libx264', bitrate= '2M')

    for image_path in tqdm(image_list):
        image1 = imageio.imread(image_path)
        # image2 = imageio.imread(image_path.replace('opt', 'ref'))
        # image = np.hstack([image2, image1])

        video_writer.append_data(image1.astype(np.uint8))

    video_writer.close()
    print(video_name)