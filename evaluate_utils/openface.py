import os


def find_mp4_files(root_dir):
    mp4_files = []

    for foldername, subfolders, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.mp4'):
                mp4_files.append(os.path.join(foldername, filename))

    return mp4_files


if __name__ == "__main__":
    root_directory = os.path.join(r"D:\cvpr2024videodata")
    mp4_files = find_mp4_files(root_directory)

    for mp4_file in mp4_files:
        cmd = "FeatureExtraction.exe -f {} -out_dir".format(mp4_file)
        print(cmd)
        os.system(cmd)
