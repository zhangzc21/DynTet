from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip
import json
import os
import cv2

def seconds_to_hhmmssms(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, remainder = divmod(remainder, 60)
    seconds, milliseconds = divmod(remainder, 1)
    milliseconds = int(milliseconds * 1000)

    time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.{milliseconds:03d}"

    return time_str


def ffmepg_cut(input_path, output_path, start_frame, end_frame, total_frame):
    cap = cv2.VideoCapture(input_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    assert frame_count == total_frame
    cap.release()

    total_time_seconds = frame_count / frame_rate

    st = seconds_to_hhmmssms(start_frame / total_frame * total_time_seconds)
    et = seconds_to_hhmmssms(end_frame / total_frame * total_time_seconds)
    cmd = f"ffmpeg -y -i {input_path} -ss {st} -to {et} -c copy {output_path}"
    print(cmd)
    os.system(cmd)

video_file = "data/video/obama.mp4"
base_name = os.path.basename(video_file).split('.')[0]
base_dir = os.path.dirname(video_file)
save_dir = os.path.join(base_dir,base_name)
train_cfg = os.path.join(save_dir, 'mv_transforms_train.json')
cfg = json.load(open(train_cfg, 'r'))
train_frames = len(cfg['frames'])
print(train_frames)
val_cfg = os.path.join(save_dir, 'mv_transforms_val.json')
cfg = json.load(open(val_cfg, 'r'))
test_frames = len(cfg['frames'])
print(test_frames)

video_clip = VideoFileClip(video_file)
fps = video_clip.fps


total_frames = train_frames + test_frames
start_frame_1 = 0
end_frame_1 = train_frames
output_file_1 = os.path.join(save_dir, 'train.mp4')
ffmepg_cut(video_file, output_file_1, start_frame_1, end_frame_1, total_frames)

start_frame_2 = train_frames
end_frame_2 = total_frames
output_file_2 = os.path.join(save_dir, 'val.mp4')
ffmepg_cut(video_file, output_file_2, start_frame_2, end_frame_2, total_frames)


video_clip = VideoFileClip(output_file_1)
frame_count = video_clip.reader.nframes
print('video1 frame', frame_count)
audio_clip = video_clip.audio
audio_clip.write_audiofile(os.path.join(save_dir, "audio_train.wav"))

video_clip = VideoFileClip(output_file_2)
frame_count = video_clip.reader.nframes
print('video2 frame', frame_count)
audio_clip = video_clip.audio
audio_clip.write_audiofile(os.path.join(save_dir, "audio_val.wav"))

video_clip.reader.close()
