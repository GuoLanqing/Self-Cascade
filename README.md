<div align="center">

<h2>Make a Cheap Scaling: A Self-Cascade Diffusion Model for Higher-Resolution Adaptation</h2> 

 <a href='https://arxiv.org/abs/2402.10491'><img src='https://img.shields.io/badge/ArXiv-2305.18247-red'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='https://guolanqing.github.io/Self-Cascade/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> 
  

_**[Lanqing Guo*](https://github.com/YingqingHe), [Yingqing He*](https://github.com/YingqingHe), Haoxin Chen, [Menghan Xia]((https://menghanxia.github.io/)), [Xiaodong Cun](http://vinthony.github.io/), [Yufei Wang](https://wyf0912.github.io), [Siyu Huang](https://siyuhuang.github.io),<br> 
[Yong Zhang<sup>#](https://yzhang2016.github.io), [Xintao Wang](https://xinntao.github.io/), [Qifeng Chen](https://cqf.io/), [Ying Shan](https://scholar.google.com/citations?hl=zh-CN&user=4oXBp9UAAAAJ) and [Bihan Wen<sup>#](https://personal.ntu.edu.sg/bihan.wen/)**_


(* first author, # corresponding author)

</div>

## 🥳 Demo
<p align="center"> <img src="docs/video_res/2048/0194_rank1-seed124.mp4" width="700px"> </p>

Please check more demo videos at the [project page](https://guolanqing.github.io/Self-Cascade/).

## 🔆 Abstract
<b>TL; DR: 🤗🤗🤗 **Self-cascade diffusion model** is a lightweight and efficient scale adaptation approach for higher-resolution image and video generation.</b>

> Diffusion models have proven to be highly effective in image and video generation; however, they still face composition challenges when generating images of varying sizes due to single-scale training data. Adapting large pre-trained diffusion models for higher resolution demands substantial computational and optimization resources, yet achieving a generation capability comparable to low-resolution models remains elusive. This paper proposes a novel self-cascade diffusion model that leverages the rich knowledge gained from a well-trained low-resolution model for rapid adaptation to higher-resolution image and video generation, employing either tuning-free or cheap upsampler tuning paradigms. Integrating a sequence of multi-scale upsampler modules, the self-cascade diffusion model can efficiently adapt to a higher resolution, preserving the original composition and generation capabilities. We further propose a pivot-guided noise re-schedule strategy to speed up the inference process and improve local structural details. Compared to full fine-tuning, our approach achieves a 5X training speed-up and requires only an additional 0.002M tuning parameters. Extensive experiments demonstrate that our approach can quickly adapt to higher resolution image and video synthesis by fine-tuning for just 10k steps, with virtually no additional inference time.


## 📭 Contact
If your have any comments or questions, feel free to contact  [Lanqing Guo](lanqing001@e.ntu.edu.sg) or [Yingqing He](yhebm@connect.ust.hk).
