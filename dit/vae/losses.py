import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class LPIPS(nn.Module):
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.channels = [64, 128, 256, 512, 512]
        
        # Use VGG16 pretrained
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        
        for x in range(4):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg[x])
        
        # Fix weights
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, input, target):
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        
        h0 = in0_input
        h1 = in1_input
        
        outs0 = []
        outs1 = []
        
        h0 = self.slice1(h0)
        h1 = self.slice1(h1)
        outs0.append(h0)
        outs1.append(h1)
        
        h0 = self.slice2(h0)
        h1 = self.slice2(h1)
        outs0.append(h0)
        outs1.append(h1)
        
        h0 = self.slice3(h0)
        h1 = self.slice3(h1)
        outs0.append(h0)
        outs1.append(h1)
        
        h0 = self.slice4(h0)
        h1 = self.slice4(h1)
        outs0.append(h0)
        outs1.append(h1)
        
        h0 = self.slice5(h0)
        h1 = self.slice5(h1)
        outs0.append(h0)
        outs1.append(h1)
        
        val = 0
        for i in range(len(outs0)):
            val += torch.mean(torch.abs(outs0[i] - outs1[i]))
            
        return val

class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

class GradientLoss(nn.Module):
    """
    Computes the gradient loss between predicted and target images.
    Useful for preserving edges and text structures.
    """
    def __init__(self, channels=3, device="cuda"):
        super(GradientLoss, self).__init__()
        # Sobel kernel for X direction
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        # Sobel kernel for Y direction
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Replicate for all channels
        self.kernel_x = kernel_x.repeat(channels, 1, 1, 1).to(device)
        self.kernel_y = kernel_y.repeat(channels, 1, 1, 1).to(device)
        self.channels = channels

    def forward(self, input, target):
        # Compute gradients for input
        grad_x_in = F.conv2d(input, self.kernel_x, padding=1, groups=self.channels)
        grad_y_in = F.conv2d(input, self.kernel_y, padding=1, groups=self.channels)
        grad_in = torch.abs(grad_x_in) + torch.abs(grad_y_in)
        
        # Compute gradients for target
        grad_x_tgt = F.conv2d(target, self.kernel_x, padding=1, groups=self.channels)
        grad_y_tgt = F.conv2d(target, self.kernel_y, padding=1, groups=self.channels)
        grad_tgt = torch.abs(grad_x_tgt) + torch.abs(grad_y_tgt)
        
        # L1 loss between gradients
        return F.l1_loss(grad_in, grad_tgt)

def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake))
    )
    return d_loss
