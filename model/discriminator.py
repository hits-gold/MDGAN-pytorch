import torch
import torch.nn as nn
from .utils import Ddown, DDM


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # D_down(in_channels, out_channels, normalize=True)
        self.down1 = Ddown(4, 32, normalize=False)
        self.down2 = Ddown(64, 128)
        self.down3 = Ddown(256, 256)
        self.down4 = Ddown(512, 256)
        self.down5 = Ddown(512, 512)

        # DDM(layer_number)
        self.ddm1 = DDM(1)
        self.ddm2 = DDM(2)
        self.ddm3 = DDM(3)
        self.ddm4 = DDM(4)

        # 1-channel feature map (PatchGAN output)
        self.patch = nn.Conv2d(512, 1, 3, 1, 1, bias=False)

    def forward(self, x: torch.tensor, mask: torch.tensor)-> torch.tensor:
        # PatchGAN discriminator input
        x = torch.cat((x, mask), dim=1)

        d1 = self.ddm1(self.down1(x), mask)
        d2 = self.ddm2(self.down2(d1), mask)
        d3 = self.ddm3(self.down3(d2), mask)
        d4 = self.ddm4(self.down4(d3), mask)
        d5 = self.down5(d4)
        out = self.patch(d5)

        return out
