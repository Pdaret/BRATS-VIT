""" Full assembly of the parts to form the complete unet network with vit in its bottleneck"""
import torch.nn as nn
import torch.nn.functional as F
from unet import unet_parts
import vision_transformer as vt


class UNetViT(nn.Module):
    def __init__(self, n_channels, n_classes, image_size, patch_size, bilinear=False):
        super(UNetViT, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # UNet initial layers
        self.inc = unet_parts.DoubleConv(n_channels, 64)
        self.down1 = unet_parts.Down(64, 128)
        self.down2 = unet_parts.Down(128, 256)
        self.down3 = unet_parts.Down(256, 512)
        
        # Replace bottleneck with ViT
        self.vit = vt.ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=512,  # Adjust this according to the embedding dimension of ViT output
            dim=1024,         # Embedding dimension for ViT
            depth=6,          # Depth of transformer layers
            heads=8,          # Number of heads in multi-head attention
            mlp_dim=2048,     # Dimension of the MLP in ViT
            channels=512      # Number of input channels, matches the UNet down3 output
        )
        
        # Upsampling layers of UNet
        factor = 2 if bilinear else 1
        self.up1 = unet_parts.Up(1024, 512 // factor, bilinear)
        self.up2 = unet_parts.Up(512, 256 // factor, bilinear)
        self.up3 = unet_parts.Up(256, 128 // factor, bilinear)
        self.up4 = unet_parts.Up(128, 64, bilinear)
        self.outc = unet_parts.OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Pass through ViT bottleneck
        x_bottleneck = self.vit(x4)
        x_bottleneck = x_bottleneck.view(x_bottleneck.size(0), -1, 32, 32)  # Adjust the size according to the output of ViT

        x = self.up1(x_bottleneck, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits