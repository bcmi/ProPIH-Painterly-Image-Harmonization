# ProPIH-Painterly-Image-Harmonization


We release the code used in the following paper:
> **Progressive Painterly Image Harmonization from Low-level Styles to High-level Styles**  [[arXiv]](https://arxiv.org/pdf/2312.10264.pdf)<br>
>
> Li Niu, Yan Hong, Junyan Cao, Liqing Zhang
>
> Accepted by AAAI 2024

Our method can harmonize a composite image from low-level styles to high-level styles. The results harmonized to the highest style level have sufficiently stylized foregrounds, but also take the risk of content distortion and artifacts. The users can select the result harmonized to the proper style level. 



## Prerequisites
- Linux
- Python 3.9
- PyTorch 1.10
- NVIDIA GPU + CUDA

## Getting Started
### Installation
- Clone this repo:

```bash
git clone https://github.com/bcmi/ProPIH-Painterly-Image-Harmonization.git
```

- Prepare the datasets as in [PHDNet](https://github.com/bcmi/PHDNet-Painterly-Image-Harmonization/).

- Install PyTorch and dependencies:

```bash
conda create -n ProPIH python=3.9
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

- Install python requirements:

```bash
pip install -r requirements.txt
```

- Download pre-trained VGG19 from [Baidu Cloud](https://pan.baidu.com/s/1HljOE-4Q2yUeeWmteu0nNA) (access code: pc9y) or [OneDrive](https://1drv.ms/u/s!AohNSvvkuxZmgSRYV1PSXQ6IrT_r?e=lFqivv). Put it in  `./<checkpoints_dir>/pretrained`

### ProPIH train/test
- Train ProPIH: 

Modify the `content_dir` and `style_dir` to the corresponding path of each dataset in `train.sh`.

```bash
cd scripts
bash train.sh
```

The trained model would be saved in `./<checkpoints_dir>/<name>/`. If you want to load a model and resume training, add `--continue_train` and set the `--epoch XX` in `train.sh`. It would load the model `./<checkpoints_dir>/<name>/<epoch>_net_G.pth`.
For example, if the model is saved in `./AA/BB/latest_net_G.pth`, the `checkpoints_dir` should be `../AA/`, the `name` should be `BB`, and the `epoch` should be `latest`.

- Test ProPIH:



Our pre-trained model is available in [Baidu Cloud](https://pan.baidu.com/s/1CDSnqzlcLKZGD7fzIFp5Qg) (access code: azir) or [OneDrive](https://1drv.ms/u/s!AohNSvvkuxZmgRSk6iTGEUsZdVfu?e=jHrMZG). Put it in `./<checkpoints_dir>/pretrained`. We provide some test examples in `./examples`. 

```bash
cd scripts
bash test.sh
```
The output results would be saved in `./output`. Some results are shown below. We can see that from stage 1 to stage 4, the composite images are harmonized progressively from low-level styles (color, simple texture) to high-level styles (complex texture). 

<div align="center">
	<img src="figures/result.jpg" alt="harmonization_results" width="800">
</div>

## Other Resources

+ [Awesome-Image-Harmonization](https://github.com/bcmi/Awesome-Image-Harmonization)
+ [Awesome-Image-Composition](https://github.com/bcmi/Awesome-Object-Insertion)
