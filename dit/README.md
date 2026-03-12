## DiT

A classical model built in the paper “Scalable Diffusion Models with Transformers” by William Peebles and Saining Xie. see: [DiT](https://arxiv.org/abs/2212.09748).

### Quickstart

1. Download [vae](https://huggingface.co/hugfaceguy0001/SmallDiT/blob/main/vae.pt) and [dit](https://huggingface.co/hugfaceguy0001/SmallDiT/blob/main/dit.pt) pretrained weights to this folder.

2. Install python and pip, then use

   ```
   pip install torch torchvision numpy pillow tqdm
   ```

   to install necessary packages.

3. Infer.
   ```
   python inference_dit.py
   ```

### Downloading ImageNet

I trained DiT models using the classical ImageNet dataset. To download it, register huggingface and request  permission in [ImageNet](https://huggingface.co/datasets/ILSVRC/imagenet-1k).

Please download ImageNet into `dit/imagenet-dataset` folder. 

### VAE

Variational AutoEncoder. Using for image compression in DiT. 

**References:**

1. Diederik P Kingma, Max Welling, "Auto-Encoding Variational Bayes", arxiv: 1312.6114
2. Carl Doersch, "Tutorial on Variational Autoencoders", arxiv: 1606.05908

**Pretrained Weights:**

https://huggingface.co/hugfaceguy0001/SmallDiT/blob/main/vae.pt

**Start training:**

1. Download ImageNet and put it into `dit/imagenet-dataset` folder. 
2. Install python and the following packages: torch, torchvision, tqdm, datasets.
3. Train the model by `python train_vae_gan.py`. You can also use `python train_vae.py` to use L1 loss only, but that will make blur images.

**Inference:**

```cmd
python inference_vae.py --model_path "your vae model path" --image_path "your image path" --output_path "path to save reconstruction result" --device "cuda or cpu"
```

### DiT

Diffusion Transformer. 

**Pretrained Weights:**

https://huggingface.co/hugfaceguy0001/SmallDiT/blob/main/dit.pt

**Start training:**

1. Download ImageNet and put it into `dit/imagenet-dataset` folder. 
2. Install python and the following packages: torch, torchvision, tqdm, datasets, numpy, pillow.
3. Train or download vae checkpoint.
4. Train the model by `python train_dit.py`. 

**Inference:**

```cmd
python inference_dit.py --ckpt_path "your dit model path" --vae_path "your vae model path" --output_path "path to save reconstruction result" 
```