## DiT

A classical model built in the paper “Scalable Diffusion Models with Transformers” by William Peebles and Saining Xie. see: [DiT](https://arxiv.org/abs/2212.09748).

### Downloading ImageNet

I trained DiT models using the classical ImageNet dataset. To download it, register huggingface and request  permission in [ImageNet](https://huggingface.co/datasets/ILSVRC/imagenet-1k).

Please download ImageNet into `dit/`

### VAE

Variational AutoEncoder. Using for image compression in DiT. 

References:

Diederik P Kingma, Max Welling, "Auto-Encoding Variational Bayes", arxiv: 1312.6114
Carl Doersch, "Tutorial on Variational Autoencoders", arxiv: 1606.05908