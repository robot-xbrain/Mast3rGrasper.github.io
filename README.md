# .X [![Build Status](https://travis-ci.org/nandomoreirame/dotX.svg?branch=master)](https://travis-ci.org/nandomoreirame/dotX)

> Simple & Beautiful Jekyll theme



### [Demo](https://nandomoreirame.github.io/dotX/)

![dotX - free Jekyll theme](/screenshot.png)


### Overview
---
***Abstract:***
Constructing a 3D scene capable of handling open-ended language queries is a critical task in the field of robotics. This technology enables robots to perform object manipulation based on human language commands. To address this challenge, some research efforts have focused on developing language-embedded implicit fields. However, existing methods like NeRF (Neural Radiance Fields), 3D Gaussian Splatting, while excelling at high-quality 3D scene reconstruction, require numerous input views and incur high computational costs, making them unsuitable for real-time scenarios. To address these issues, we propose MASt3RGrasper, a framework that uses a small number of RGB images for rapid scene reconstruction and executes precise grasping operations based on language commands. The system generates a dense and accurate scene representation from a few RGB images, leveraging image segmentation and textual prompts to extract the target object’s point cloud. It then predicts the 6-DoF grasp poses using a grasp network. Furthermore, we introduce a synthetic data generation pipeline powered by large language models (LLM), which enhances the diversity of 3D training assets through various texture augmentations. Extensive experimental results show that our method enables accurate grasping of target objects based on textual prompts, while maintaining strong real-time performance.

### Framework

<img width="818" alt="image" src="https://github.com/user-attachments/assets/d730795a-aae8-4db1-a7e0-b95b1834aaef">


### Experimental Visualization

https://github.com/user-attachments/assets/a2fffa0b-8e5a-4c54-8246-99c4b16e9ccd

https://github.com/user-attachments/assets/5c93d05d-09bb-480f-a505-7fe65ad4e687

https://github.com/user-attachments/assets/c4902d30-ee8b-4cf3-a5d2-0fbaee643d45

### Start in 4 steps

1. Download or clone repo `git clone https://github.com/robot-xbrain/Mast3rGrasper.github.io.git`
2. Enter the folder: `cd Mast3rGrasper/`
3. Installation: `pip install -r requirements.txt`
4. run demo: `sh run.sh`



### Using Rake tasks

* Create a new page: `rake page name="contact.md"`
* Create a new post: `rake post title="TITLE OF THE POST"`

---

### Copyright and license

It is under [the MIT license](/LICENSE).

Enjoy :yum:

by [nandomoreira.me](https://nandomoreira.me)
