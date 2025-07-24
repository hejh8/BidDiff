[(ACM MM 2025)] Degradation-Consistent Learning via Bidirectional Diffusion for Low-Light Image Enhancement[Paper]()

## Over-all-Architecture
This is the official implementation code for [Degradation-Consistent Learning via Bidirectional Diffusion for Low-Light Image Enhancement]().
![Over-all-Architecture](https://github.com/user-attachments/assets/e73e3d11-9b35-4363-a066-d399701414f2)

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

```bibtex  
