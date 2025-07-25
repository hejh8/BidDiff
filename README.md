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
You need to modify ```dataset.py and config``` slightly for your environment, and then
```python test.py ```


## Acknowledgement
This repo is based on [WeatherDiff](https://github.com/IGITUGraz/WeatherDiffusion).

## Citation Information
If you find the project useful, please cite:  
```
@misc{he2025degradationconsistentlearningbidirectionaldiffusion,
      title={Degradation-Consistent Learning via Bidirectional Diffusion for Low-Light Image Enhancement}, 
      author={Jinhong He and Minglong Xue and Zhipu Liu and Mingliang Zhou and Aoxiang Ning and Palaiahnakote Shivakumara},
      year={2025},
      eprint={2507.18144},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.18144}, 
}
```
