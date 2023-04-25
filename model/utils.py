import torch
import torch.nn as nn


# Generator Downsampling block
class Gdown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, normalize: bool = True):
        super(Gdown, self).__init__()
        if normalize:
            layers = [
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.AvgPool2d(2, 2),
                nn.InstanceNorm2d(in_channels),
                nn.LeakyReLU(inplace=True),
            ]
        else:
            layers = [
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.AvgPool2d(2, 2),
                nn.LeakyReLU(inplace=True),
            ]

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.tensor)-> torch.tensor:

        return self.block(x)


# Generator Upsampling block
class Gup(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, normalize: bool = True):
        super(Gup, self).__init__()
        if normalize:
            layers = [
                nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(inplace=True),
            ]
        else:
            layers = [
                nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                nn.Tanh(),
            ]

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.tensor, f: torch.tensor = 0, skip: bool = True)-> torch.tensor:
        if skip:
            x = x + f
        x = self.block(x)

        return x


# Discriminator Downsampling block
class Ddown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, normalize: bool = True):
        super(Ddown, self).__init__()
        if normalize:
            layers = [
                nn.Conv2d(in_channels, out_channels, 4, 2, 1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(),
            ]
        else:
            layers = [
                nn.Conv2d(in_channels, out_channels, 4, 2, 1),
                nn.ReLU(),
            ]

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.tensor)-> torch.tensor:

        return self.block(x)


# Background Replacement module
class BRM(nn.Module):
    def __init__(self, d: int, out_channels: int):
        super(BRM, self).__init__()
        self.k = 2**d
        self.mask_layer = nn.Sequential(
            nn.AvgPool2d(self.k, stride=self.k),
            nn.Conv2d(1, 1, 3, 1, 1),
            nn.Sigmoid(),
        )
        self.background_layer = nn.Sequential(
            nn.AvgPool2d(self.k, stride=self.k),
            nn.Conv2d(3, out_channels, 1, stride=1),
        )

    def forward(self, x: torch.tensor, mask: torch.tensor, pnb_img: torch.tensor)-> torch.tensor:
        f = self.mask_layer(mask)
        background = self.background_layer(pnb_img)

        f_ = (f * -1) + 1
        x = (background * f_) + (x * f)

        return x


# Double Discrimination module
class DDM(nn.Module):
    def __init__(self, d: int):
        super(DDM, self).__init__()
        self.k = 2**d
        self.mask_layer = nn.Sequential(
            nn.AvgPool2d(self.k, self.k),
            nn.Conv2d(1, 1, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.tensor, mask: torch.tensor)-> torch.tensor:
        mask = self.mask_layer(mask)
        local = x * mask
        x = torch.cat((x, local), dim=1)

        return x


# weight initailizing
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)