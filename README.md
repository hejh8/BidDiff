[(ACM MM 2025)] Degradation-Consistent Learning via Bidirectional Diffusion for Low-Light Image Enhancement[Paper](https://arxiv.org/abs/2507.18144)

## Over-all-Architecture
This is the official implementation code for [Degradation-Consistent Learning via Bidirectional Diffusion for Low-Light Image Enhancement](https://arxiv.org/abs/2507.18144).
![Over-all-Architecture](https://github.com/hejh8/BidDiff/blob/main/Fig1.png)

## How to Run the Code?

### Dependencies

* OS: Ubuntu 22.04
* nvidia:
	- cuda: 12.1
* python 3.9

### Install

 Clone Repo
 ```bash
 git clone https://github.com/hejh8/BidDiff.git
 cd BidDiff 
 ```
Create Conda Environment and Install Dependencies:
```bash
pip install -r requirements.txt
```
### Data Preparation

You can refer to the following links to download the datasets.

- [LOLv1](https://daooshee.github.io/BMVC2018website/)
- [LOLv2](https://github.com/flyywh/CVPR-2020-Semi-Low-Light)

## Test
Download the [pretrained model](https://drive.google.com/drive/folders/1IAGuhxQVhtRD-PdYeRkItzQ3n0szrz9U)
You need to modify ```dataset.py and config``` slightly for your environment, and then
```python test.py ```


## Acknowledgement
This repo is based on [WeatherDiff](https://github.com/IGITUGraz/WeatherDiffusion).

## Citation Information
If you find the project useful, please cite:  
```
@inproceedings{he2025degradation,
  title={Degradation-Consistent Learning via Bidirectional Diffusion for Low-Light Image Enhancement},
  author={He, Jinhong and Xue, Minglong and Liu, Zhipu and Zhou, Mingliang and Ning, Aoxiang and Shivakumara, Palaiahnakote},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
  pages={7152--7161},
  year={2025}
}
```
