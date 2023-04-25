import torch
from torchvision.utils import save_image

import os
from dataset import get_loader
from model import Generator


def Tester(args):
    save_path = f"./result/{args.exp}/inference/"
    os.makedirs(save_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    netG = Generator().to(device)

    model_path = f"result/{args.exp}/model/{args.exp}_{args.epochs}.pth"
    checkpoint = torch.load(model_path)
    netG.load_state_dict(checkpoint["model_G_state_dict"])

    netG.eval()
    loader = get_loader(args)

    with torch.no_grad():
        for idx, sample in enumerate(loader):
            img = sample["image"].to(device)
            mask = sample["mask"].to(device)
            path = sample["file_name"]
            file_path = save_path + path[0]

            z = torch.normal(0, torch.var(img), (img.shape[0], 8)).to(device)  # random vector(latent dimension : 8)

            gen, no_brm = netG(mask, z, img)
            gen = gen.squeeze()
            no_brm = no_brm.squeeze()

            if idx == 0:
                save_image(mask, save_path + "mask.png")

            save_image([img[0], gen, no_brm], file_path, nrows=3)
