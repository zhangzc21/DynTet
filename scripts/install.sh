conda create -n dyntet python=3.9
conda activate dyntet
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
pip install ninja imageio PyOpenGL glfw xatlas gdown
pip install git+https://github.com/NVlabs/nvdiffrast/
pip install git+https://github.com/facebookresearch/pytorch3d/
pip install --global-option="--no-networks" git+https://github.com/NVlabs/tiny-cuda-nn#subdirectory=bindings/torch
pip install scikit-learn configargparse face_alignment natsort matplotlib dominate tensorboard kornia trimesh open3d imageio-ffmpeg lpips easydict pysdf rich openpyxl gfpgan
