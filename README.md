# Learning Dynamic Tetrahedra for High-Quality Talking Head Synthesis (CVPR 2024)
This repository contains a PyTorch re-implementation of the paper: Learning Dynamic Tetrahedra for High-Quality Talking Head Synthesis.
### | [Arxiv](https://arxiv.org/pdf/2402.17364.pdf) | [Video](https://youtu.be/Hahv5jy2w_E) | 

## Abstract

Recent works in implicit representations, such as Neural Radiance Fields (NeRF), have advanced the generation of realistic and animatable head avatars from video sequences. These implicit methods are still confronted by visual artifacts and jitters, since the lack of explicit geometric constraints poses a fundamental challenge in accurately modeling complex facial deformations. In this paper, we introduce Dynamic Tetrahedra (DynTet), a novel hybrid representation that encodes explicit dynamic meshes by neural networks to ensure geometric consistency across various motions and viewpoints. DynTet is parameterized by the coordinate-based networks which learn signed distance, deformation, and material texture, anchoring the training data into a predefined tetrahedra grid. Leveraging Marching Tetrahedra, DynTet efficiently decodes textured meshes with a consistent topology, enabling fast rendering through a differentiable rasterizer and supervision via a pixel loss. To enhance training efficiency, we incorporate classical 3D Morphable Models to facilitate geometry learning and define a canonical space for simplifying texture learning. These advantages are readily achievable owing to the effective geometric representation employed in DynTet. Compared with prior works, DynTet demonstrates significant improvements in fidelity, lip synchronization, and real-time performance according to various metrics. Beyond producing stable and visually appealing synthesis videos, our method also outputs the dynamic meshes which is promising to enable many emerging applications. 

## Code
Code will be released soon.

## Citation
```
@InProceedings{zhang2024learning,
    title={Learning Dynamic Tetrahedra for High-Quality Talking Head Synthesis}, 
    author={Zicheng Zhang and Ruobing Zheng and Ziwen Liu and Congying Han and Tianqi Li and Meng Wang and Tiande Guo and Jingdong Chen and Bonan Li and Ming Yang},
    booktitle={CVPR},
    year={2024}
}
```
