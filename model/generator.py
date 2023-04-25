import torch
import torch.nn as nn
from .utils import Gup, Gdown, BRM


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Fully-connected layer-> z_randomvector(size=(batchsize, 8)) to image size
        self.noise_layer = nn.Linear(8, 3 * 256**2)

        # G_down(in_channels, out_channels, normalize=True)
        self.down1 = Gdown(3, 64, normalize=False)
        self.down2 = Gdown(64, 128)
        self.down3 = Gdown(128, 256)
        self.down4 = Gdown(256, 256)
        self.down5 = Gdown(256, 512)
        self.down6 = Gdown(512, 512, normalize=False)

        # G_up(in_channels, out_channels, normalize=True)
        self.up1 = Gup(512, 512)
        self.up2 = Gup(512, 256)
        self.up3 = Gup(256, 256)
        self.up4 = Gup(256, 128)
        self.up5 = Gup(128, 64)
        self.up6 = Gup(64, 3, normalize=False)

        # BRM(layer_number, out_channels)
        self.brm_down1 = BRM(1, 64)
        self.brm_down2 = BRM(2, 128)
        self.brm_down3 = BRM(3, 256)
        self.brm_down4 = BRM(4, 256)
        self.brm_down5 = BRM(5, 512)
        self.brm_up1 = BRM(4, 256)
        self.brm_up2 = BRM(3, 256)
        self.brm_up3 = BRM(2, 128)
        self.brm_up4 = BRM(1, 64)
        self.brm_up5 = BRM(0, 3)

    def forward(self, mask: torch.tensor, z: torch.tensor, pnb_img: torch.tensor)-> torch.tensor:
        # add noise on defect area of input image
        x_z = self.noise_layer(z)
        x_z = x_z.view(pnb_img.size(0), 3, 256, 256)
        noise_mask = mask.data
        noise_mask[noise_mask == -1] = 0
        x = (x_z * noise_mask) + pnb_img

        # downsampling
        d1 = self.brm_down1(self.down1(x), mask, pnb_img)
        d2 = self.brm_down2(self.down2(d1), mask, pnb_img)
        d3 = self.brm_down3(self.down3(d2), mask, pnb_img)
        d4 = self.brm_down4(self.down4(d3), mask, pnb_img)
        d5 = self.brm_down5(self.down5(d4), mask, pnb_img)
        d6 = self.down6(d5)

        # upsampling
        u1 = self.up1(d6, skip=False)
        u2 = self.brm_up1(self.up2(u1, d5), mask, pnb_img)
        u3 = self.brm_up2(self.up3(u2, d4), mask, pnb_img)
        u4 = self.brm_up3(self.up4(u3, d3), mask, pnb_img)
        u5 = self.brm_up4(self.up5(u4, d2), mask, pnb_img)
        u6 = self.up6(u5, d1)  # last input of BRM
        gen = self.brm_up5(u6, mask, pnb_img)

        return gen, u6
