import os

from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--video', type = str, default ="/data2/zzc/ER-NeRF/trial_may/results/sing.mp4")
parser.add_argument('--audio', type = str, default = "/data2/zzc/nvdiffrec/data/video/sing.wav")
args = parser.parse_args()

video_file = args.video
audio_file = args.audio
output_file = video_file.replace('.mp4', '_audio.mp4')

os.system(f"ffmpeg -y -i {video_file} -i {audio_file} -c:v copy -c:a aac -strict experimental {output_file}")



