export CUDA_VISIBLE_DEVICES=0

sudo env "PATH=$PATH" python data_utils/process.py --path "/mnt/sdc/math_data/obama.mp4" --save_dir "/mnt/sdc/math_data/obama" --task -1
