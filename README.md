# DensePure: Understanding Diffusion Models towards Adversarial Robustness

## Requirements

- Python 3.8.5
- CUDA=11.1 
- Installation of PyTorch 1.8.0:
    ```bash
    conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
    ```
- Installation of required packages:
    ```bash
    pip install -r requirements.txt
    ```

Above Does note work, check Carlini's repo and make the conda env, pip install additional lmdb etc:
 ```
conda create -n denp python=3.8 -y
conda activate denp
conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch
pip install timm transformers statsmodels
pip install lmdb
pip install blobfile
 ```

For Toolkit/LambdaCloud, do not use `conda install` (will have C++ issues), instead do:
```
conda create -n denp python=3.8 -y
conda activate denp
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install timm transformers statsmodels
```

Then change model ckpt path in `runners/diffpure_guided_densepure.py` line 29. Change imagenet data path in `dataset.py`.

## Datasets, Pre-trained Diffusion Models and Classifiers
Before running our code, you need to first prepare two datasets CIFAR-10 and ImageNet. CIFAR-10 will be downloaded automatically.
For ImageNet, you need to download validation images of ILSVRC2012 from https://www.image-net.org/. And the images need to be preprocessed by running the scripts `valprep.sh` from https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
under validation directory.  

Please change IMAGENET_DIR to your own location of ImageNet dataset in `datasets.py` before running the code.  

For the pre-trained diffusion models, you need to first download them from the following links:  
- [Improved Diffusion](https://github.com/openai/improved-diffusion) for
  CIFAR-10: (`cifar10_uncond_50M_500K.pt`: [download link](https://openaipublic.blob.core.windows.net/diffusion/march-2021/cifar10_uncond_50M_500K.pt))
- [Guided Diffusion](https://github.com/openai/guided-diffusion) for
  ImageNet: (`256x256 diffusion unconditional`: [download link](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt))

For the pre-trained classifiers, ViT-B/16 model on CIFAR-10 will be automatically downloaded by `transformers`.  For ImageNet BEiT large model, you need to dwonload from the following links:
- [BEiT](https://github.com/microsoft/unilm/tree/master/beit) for
  ImageNet: (`beit_large_patch16_512_pt22k_ft22kto1k.pth`: [download link](https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_512_pt22k_ft22kto1k.pth))

Please place all the pretrained models in the `pretrained` directory. If you want to use your own classifiers, code need to be changed in `eval_certified_densepure.py`.

## Run Experiments of Carlini 2022
We provide our own code implementation for the paper [(Certified!!) Adversarial Robustness for Free!](https://arxiv.org/abs/2206.10550) to compare with DensePure.  

To gain the results in Table 1 about Carlini22, please run the following scripts using different noise levels `sigma`: 
```
cd run_scripts
bash carlini22_cifar10.sh [sigma] # For CIFAR-10
bash carlini22_imagenet.sh [sigma]  # For ImageNet
```

## Run Experiments of DensePure
To get certified accuracy under DensePure, please run the following scripts:
```
cd run_scripts
bash densepure_cifar10.sh [sigma] [steps] [reverse_seed] # For CIFAR-10
bash densepure_imagenet.sh 1 10 1 # For ImageNet
```

Note: `sigma` is the noise level of randomized smoothing. `steps` is the parameter for fast sampling steps in Section 5.2 and it must be larger than one and smaller than the total reverse steps. `reverse_seed` is a parameter which control majority vote process in Section 5.2. For example, you need to run `densepure_cifar10.sh` 10 times with 10 different `reverse_seed` to finish 10 majority vote numbers experiments. After running above scripts under one `reverse_seed`, you will gain a `.npy` file that contains labels of 100000 (for CIFAR-10) randomized smoothing sampling times. If you want to obtain the final results of 10 majority vote numbers, you need to run the following scripts in `results` directory:
```
cd results
bash merge_cifar10.sh [sigma] [steps] [majority_vote_numbers] # For CIFAR-10
bash merge_imagenet.sh [sigma] [steps] [majority_vote_numbers] # For ImageNet
```

