import torch
import torch.nn as nn
from torchvision.utils import save_image
from torch import optim
import torch
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional import image_gradients
from torch.autograd import Variable
import torch.autograd as autograd

import os
import time
import numpy as np
from datetime import datetime

from model import Generator, Discriminator, weights_init_normal
from dataset import get_loader


def compute_gradient_penalty(D, real_img, fake_img, mask):
    """Calculates the gradient penalty loss for WGAN GP"""
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(real_img.shape[0], 1, 1, 1)
    alpha = alpha.expand_as(real_img).cuda()
    # Get random interpolation between real and fake samples
    interpolates = (
        (alpha * real_img + ((1 - alpha) * fake_img)).requires_grad_(True).cuda()
    )
    d_interpolates = D(interpolates, mask)
    fake = Variable(Tensor(real_img.shape[0], 1, 8, 8).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def trainer(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    save_path = f"./result/{args.exp}"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path + "/img", exist_ok=True)
    os.makedirs(save_path + "/model", exist_ok=True)

    writer = SummaryWriter(f"./logs/{args.exp}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG = Generator().to(device)
    netD = Discriminator().to(device)

    # Loss
    L1_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()
    criterion_D = nn.MSELoss()

    # Optimizer
    optimizer_G = optim.Adam(netG.parameters(), lr=args.lr, betas=args.betas)
    optimizer_D = optim.Adam(netD.parameters(), lr=args.lr, betas=args.betas)

    # loss weight
    gamma_r = args.gammas[0]
    gamma_d = args.gammas[1]
    gamma_g = args.gammas[2]
    gamma_gp = args.gammas[3]

    # Initialize weights
    netG.apply(weights_init_normal)
    netD.apply(weights_init_normal)

    netG.train()
    netD.train()

    loader = get_loader(args)

    for epoch in range(args.epochs):
        epoch_start = time.time()
        for idx, sample in enumerate(loader):
            iter_start = time.time()

            pnb_img = sample["pnb_image"].to(device)
            img = sample["image"].to(device)
            mask = sample["mask"].to(device)  # mask for loss [0, 1]
            input_mask = mask.data
            input_mask[input_mask == 0] = -1
            input_mask.requires_grad_(True).to(device)  # mask for input [-1, 1]
            reverse_mask = 1 - mask

            z1 = torch.normal(0, torch.var(pnb_img), (img.shape[0], 8)).to(
                device
            )  # random vector(latent dimension : 8)
            z2 = torch.normal(0, torch.var(pnb_img), (img.shape[0], 8)).to(device)

            # -------------------------------- Train --------------------------------
            # PatchGAN Discriminator Label
            real_label = torch.ones(pnb_img.shape[0], 1, 8, 8, requires_grad=False).to(
                device
            )

            # -----------------
            #  Forward pass
            # -----------------
            # Generate
            gen1, dl1 = netG(input_mask, z1, pnb_img)  # gen -> generated image, dl -> input of last BRM
            gen2, dl2 = netG(input_mask, z2, pnb_img)  # gen2 -> for diversity

            # Discriminate
            out_dis1 = netD(gen1, input_mask)
            out_dis2 = netD(gen2, input_mask)
            out_real = netD(img, input_mask)
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Loss
            # reconstruction loss
            recon_loss = 5 * L1_loss(gen1 * mask, img * mask)  # reconstruction loss
            nb_loss1 = L1_loss(gen1 * reverse_mask, img * reverse_mask) + \
                       L1_loss(gen2 * reverse_mask, img * reverse_mask)  # normal background loss
            nb_loss2 = L1_loss(dl1 * reverse_mask, img * reverse_mask) + \
                       L1_loss(dl2 * reverse_mask, img * reverse_mask)  # normal background loss (using input of last BRM)
            loss_r = recon_loss + nb_loss1 + nb_loss2

            # diversity loss
            loss_div = -L1_loss(gen1 * mask, gen2 * mask)

            # gradient loss
            img_dy, img_dx = image_gradients(img)
            gen1_dy, gen1_dx = image_gradients(gen1)
            gen2_dy, gen2_dx = image_gradients(gen2)
            mask_dy, mask_dx = image_gradients(mask)
            mask_dy[mask_dy != 0] = 1  # non-zero elements to 1
            mask_dx[mask_dx != 0] = 1

            loss_grad1 = (mse_loss(img_dy * mask_dy, gen1_dy * mask_dy)
                          + mse_loss(img_dx * mask_dx, gen1_dx * mask_dx)) * 0.5
            loss_grad2 = (mse_loss(img_dy * mask_dy, gen2_dy * mask_dy)
                          + mse_loss(img_dx * mask_dx, gen2_dx * mask_dx)) * 0.5
            loss_grad = loss_grad1 + loss_grad2

            # adversarial loss
            loss_adv1 = criterion_D(out_dis1, real_label)
            loss_adv2 = criterion_D(out_dis2, real_label)
            loss_adv = loss_adv1 + loss_adv2

            # Total loss
            loss_G = loss_r * gamma_r + loss_div * gamma_d + loss_adv + loss_grad * gamma_g

            loss_G.backward()
            optimizer_G.step()

            # -----------------
            #  Train Discriminator
            # -----------------
            optimizer_D.zero_grad()

            # Loss
            # gradient-penalty loss
            loss_gp1 = compute_gradient_penalty(netD, img.data, gen1.data, mask.data)
            loss_gp2 = compute_gradient_penalty(netD, img.data, gen2.data, mask.data)
            loss_gp = loss_gp1 + loss_gp2

            out_real = netD(img, input_mask)
            loss_adv3 = 2 * criterion_D(out_real, real_label)

            # Total loss
            loss_D = loss_gp * gamma_gp + loss_adv3
            loss_adv = loss_adv + loss_adv3
            loss_D.backward()
            optimizer_D.step()

            # -------------------------------- save --------------------------------
            # -----------------
            #  Log print & save
            # -----------------
            # print log
            iter_end = time.time() - iter_start
            if idx + 1 == len(loader):
                epoch_end = time.time() - epoch_start
                loss_log = (
                    "[Epoch %d/%d] [D loss: %f] [G loss: %f, reconstruction: %f, adv: %f, diversity: %f, gradient: %f] Time: %s"
                    % (
                        epoch + 1,
                        args.epochs,
                        loss_G.item(),
                        loss_D.item(),
                        loss_r.item(),
                        loss_adv.item(),
                        loss_div.item(),
                        loss_grad.item(),
                        epoch_end,
                    )
                )
                # tensorboard
                writer.add_scalar("loss_G/train", loss_G.item(), epoch + 1)
                writer.add_scalar("loss_D/train", loss_D.item(), epoch + 1)
                writer.add_scalar("reconstruction_loss", loss_r.item(), epoch + 1)
                writer.add_scalar("adversarial_loss", loss_adv.item(), epoch + 1)
                writer.add_scalar("diversity_loss", loss_div.item(), epoch + 1)
                writer.add_scalar("gradient_loss", loss_grad.item(), epoch + 1)

            else:
                loss_log = (
                    "[Iter %d/%d] [D loss: %f] [G loss: %f, reconstruction: %f, adv: %f, diversity: %f, gradient: %f] Time: %s"
                    % (
                        idx + 1,
                        len(loader),
                        loss_G.item(),
                        loss_D.item(),
                        loss_r.item(),
                        loss_adv.item(),
                        loss_div.item(),
                        loss_grad.item(),
                        iter_end,
                    )
                )
            print(loss_log)

            # save log
            f = open(save_path + "/log.txt", "a")
            if idx == epoch == 0:
                f.write(f"{args.exp}  |  " + str(datetime.now()) + "\n")
                f.write(
                    f"image source : {args.root}, defect type : {args.defect_type}, num_epochs : {args.epochs}, batch_size : {args.batch_size}"
                )
                f.write("\n")
            f.write(loss_log + "\n")
            f.close()

            # -----------------
            #  Model & Image save
            # -----------------

            if (epoch + 1) % args.save_epoch == 0:
                torch.save(
                    {
                        "optimizer_G_state_dict": optimizer_G.state_dict(),
                        "model_G_state_dict": netG.state_dict(),
                        "optimizer_D_state_dict": optimizer_D.state_dict(),
                        "model_D_state_dict": netD.state_dict(),
                    },
                    save_path + f"/model/{args.exp}_{epoch+1}.pth",
                )

                save_image(
                    [img.data[0], pnb_img.data[0], gen1.data[0]],
                    save_path + f"/img/epoch_{epoch+1}.png",
                    nrows=3,
                )
