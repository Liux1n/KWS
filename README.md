# Extreme-Edge Experience Replay for Keyword Spotting

## Introduction

<!-- On-device Domain Adaptation (ODDA) for Noise-Robust Keyword Spotting is a methodology aimed at increasing the robustness to unseen noises for a keyword spotting system. The objective of keyword spotting (KWS) is to detect a set of predefined keywords within a stream of user utterances. The difficulty of the task increases in real environments with significant noise. To improve the performance of a KWS system in noise conditions unseen during training, we propose a methodology for tailoring a model to on-site noises through ODDA. -->

<!-- If you use our methodology in an academic context, please cite the following publication:

Paper: [Towards On-device Domain Adaptation for Noise-Robust Keyword Spotting](https://ieeexplore.ieee.org/document/9869990)

```
@INPROCEEDINGS{cioflan2022oddaAICAS,
  author={Cioflan, Cristian and Cavigelli, Lukas and Rusci, Manuele and De Prado, Miguel and Benini, Luca},
  booktitle={2022 IEEE 4th International Conference on Artificial Intelligence Circuits and Systems (AICAS)}, 
  title={Towards On-device Domain Adaptation for Noise-Robust Keyword Spotting}, 
  year={2022},
  volume={},
  number={},
  pages={82-85},
  doi={10.1109/AICAS54282.2022.9869990}}

``` -->


## Installation

To install the packages required to run the model's training and adaptation (in PyTorch), a conda environment can be created from `environment.yml` by running:
```
conda env create -f environment.yml
```
## Example

To change the preprocessing parameters (e.g., number of MFCCs) or the training parameters (e.g., noise factor, learning mode -nlkws, nakws, odda-), adapt ```utils.py``` according to your settings. 

To train a model from scratch in CIL, use the command:
```
python main.py --mode cil --noise_mode nlkws --method base_pretrain
```
To train a model from scratch in DIL, use the command:
```
python main.py --mode dil --noise_mode nakws --method base_pretrain
```
To perform training for continual learning e.g. in DIL, save a pre-trained model under the folder models/, and use the following command:
DA-ECBRS: 
```
python main.py --mode dil --noise_mode nakws --method DAECBRS --forgetting
```
LA-ECBRS:
```
python main.py --mode dil --noise_mode nakws --method LAECBRS --forgetting
```


## Contributor
Cristian Cioflan, ETH Zurich, [cioflanc@iis.ee.ethz.ch](cioflanc@iis.ee.ethz.ch)
Liuxin Qing, ETH Zurich, [liuqing@student.ethz.ch](liuqing@student.ethz.ch)


## License
The code is released under Apache 2.0, see the LICENSE file in the root of this repository for details.
