# REGIS: Refining Generated Videos via Iterative Stylistic Redesigning

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/regis-refining-generated-videos-via-iterative/text-to-video-generation-on-ucf-101)](https://paperswithcode.com/sota/text-to-video-generation-on-ucf-101?p=regis-refining-generated-videos-via-iterative)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/regis-refining-generated-videos-via-iterative/video-generation-on-ucf-101)](https://paperswithcode.com/sota/video-generation-on-ucf-101?p=regis-refining-generated-videos-via-iterative)

This repository contains the code and supplementary materials for [REGIS: Refining Generated Videos via Iterative Stylistic Redesigning](https://doi.org/10.21203/rs.3.rs-3541408/v1). This research explores a novel approach to enhance the quality of videos generated from textual descriptions. The proposed method, known as REGIS, refines generated videos through iterative processes, resulting in improved video fidelity and reduced spatio-temporal noise.


## Abstract
In recent years, generative models have made impressive advancements towards realistic output; in particular, models working in the modalities of text and audio have reached a level of quality at which generated text or audio cannot be easily distinguished from real text or audio Despite these revolutionary advancements, the synthesis of realistic and temporally consistent videos is still in its inception. In this paper, we introduce a novel approach to the creation of realistic videos that focuses on improving the generated video in the latter steps of a video generation process. Specifically, we propose a framework for the iterative refinement of generated videos through repeated passes through a neural network trained to model the spatio-temporal dependencies found in real videos. Through our experiments, we demonstrate that our proposed approach significantly improves upon the generations of text-to-video models and achieves state-of-the-art results of the UCF-101 benchmark; removing the spatio-temporal artifacts and noise that make synthetic videos distinguishable from real videos. In addition, we discuss the ways in which one might augment this framework to achieve better performance.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
  - [Quantitative Analysis](#quantitative-analysis)
  - [Improved Fidelity for High-Movement Videos](#improved-fidelity-for-high-movement-videos)
  - [Comparison with Other Methods](#comparison-with-other-methods)

## Prerequisites

Before running the code, make sure you have the following prerequisites:

- Python 3.x
- PyTorch
- Torchsummary
- Opencv
- Tqdm
- Imageio
- Numpy
- Peft
- Matplotlib 

## Installation

1. Clone this repository to your local machine.

```bash
git clone https://github.com/Jaso1024/Refining-Generated-Videos.git
```

2. Download the [UCF101 dataset](https://www.crcv.ucf.edu/data/UCF101.php)

## Usage

All training files are located in the Training folder, and Models in the Models folder. To train one of the models, select a model to train and run to the corresponding training file in the Training folder.

## Results

The results obtained from the experiments are as follows:

### Quantitative Analysis

| Method           | FVD                                |
|------------------|------------------------------------|
| VideoFusion      | 130.19 ± 6                         |
| REGIS-VQ, I=2    | 129.39 ± 6                         |
| REGIS-U, I=2     | 115.36 ± 6                         |
| REGIS-Fuse, I=8  | 125.32 ± 6                         |


REGIS-U achieved the best performance, reducing the FVD to 115.36 ± 6, a noticeable improvement over the baseline VideoFusion model.

### Improved Fidelity for High-Movement Videos

REGIS-VQ demonstrates significant improvements in classes with large amounts of movement, such as "Pole Vault" and "High Jump." For example:

| Class           | VideoFusion FVD | REGIS-VQ FVD |
|-----------------|-----------------|--------------|
| Pole Vault      | 370.06          | 76.684       |
| High Jump       | 259.67          | 114.53       |
| Hammer Throw    | 193.21          | 120.92       |
| ...             | ...             | ...          |

REGIS-VQ achieved an FVD 179.3% better than VideoFusion's original FVD on the "Pole Vault" class.

### Comparison with Other Methods

REGIS outperforms other text-to-video generation methods on the UCF101 dataset:

| Method         | Resolution       | FVD↓   |
|----------------|------------------|--------|
| TGANv2         | 16 x 128 x 128   | 1209   |
| TATS           | 16 x 128 x 128   | 332    |
| VideoFusion    | 16 x 128 x 128   | 173    |
| REGIS-Fuse     | 16 x 128 x 128   | 141    |
