import pandas as pd
import numpy as np
import glob
import os


def extract_au(file_path):
    df = pd.read_csv(file_path)
    au_columns = [col for col in df.columns if col.startswith(' AU') and col.endswith('_r')]
    au_data = df[au_columns].to_numpy()
    return au_data


prefix = 'ernerf'
main_dir = os.path.abspath(r'D:\cvpr2024\audio_driven')
file_list = glob.glob(os.path.join(main_dir, f"*.csv"))

res = {}
for file in file_list:
    au_data1 = extract_au(file)
    if 'val1' in file:
        au_data2 = extract_au(r"D:\cvpr2024\audio\val1.csv")
    elif 'val2' in file:
        au_data2 = extract_au(r"D:\cvpr2024\audio\val2.csv")
    else:
        assert 'error'
    min_len = min(len(au_data1), len(au_data2))
    diff = np.mean(np.sum(np.abs(au_data1[:min_len] - au_data2[:min_len]), axis = -1))
    basename = os.path.basename(file)[:-4]
    res[basename] = diff

print(res)
# print(np.mean(list(res.values())))
