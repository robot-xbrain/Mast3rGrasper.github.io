# MASt3RGrasper [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)]


## Overview
---
***Abstract:***
Constructing a 3D scene capable of handling open-ended language queries is a critical task in the field of robotics. This technology enables robots to perform object manipulation based on human language commands. To address this challenge, some research efforts have focused on developing language-embedded implicit fields. However, existing methods like NeRF (Neural Radiance Fields), 3D Gaussian Splatting, while excelling at high-quality 3D scene reconstruction, require numerous input views and incur high computational costs, making them unsuitable for real-time scenarios. To address these issues, we propose MASt3RGrasper, a framework that uses a small number of RGB images for rapid scene reconstruction and executes precise grasping operations based on language commands. The system generates a dense and accurate scene representation from a few RGB images, leveraging image segmentation and textual prompts to extract the target object’s point cloud. It then predicts the 6-DoF grasp poses using a grasp network. Furthermore, we introduce a synthetic data generation pipeline powered by large language models (LLM), which enhances the diversity of 3D training assets through various texture augmentations. Extensive experimental results show that our method enables accurate grasping of target objects based on textual prompts, while maintaining strong real-time performance.

### Framework

<img width="818" alt="image" src="https://github.com/user-attachments/assets/d730795a-aae8-4db1-a7e0-b95b1834aaef">


### Experimental Visualization


<iframe width="560" height="315" src="https://www.youtube.com/embed/ZFOOuQ-oRI0" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>



## Prerequisites & Installation
Install anaconda following the [anaconda installation documentation](https://docs.anaconda.com/anaconda/install/).
Create an environment with all required packages with the following command :
```bashscript
conda env create -f mast3rgrasper.yml
conda activate mast3rgrasper
```
then setup the segment-anything library:
```bashscript
cd segment-anything
pip install -e .
cd ..
```
download model checkpoints for SAM and place them in the segment-anything directory

### Start Demo

```bashscript
sh run.sh
```

---

### Copyright and license

It is under [the MIT license](/LICENSE).

