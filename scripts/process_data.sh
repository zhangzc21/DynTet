export CUDA_VISIBLE_DEVICES=0

sudo env "PATH=$PATH" python data_utils/process.py --path "data/vide/obama.mp4" --save_dir "data/vide/obama" --task -1
