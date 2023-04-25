import os
import glob
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
import cv2


class TrainDataset(Dataset):
    def __init__(self, args):
        self.args = args
        img_paths = os.path.join(self.args.root, f"images/{self.args.defect_type}")
        mask_paths = os.path.join(self.args.root, f"ground_truth/{self.args.defect_type}")

        self.normal = cv2.cvtColor(cv2.imread(self.args.normal_path), cv2.COLOR_BGR2RGB)  # normal image
        self.img_paths = sorted(glob.glob(img_paths + "/*.*"))  # defect images path
        self.mask_paths = sorted(glob.glob(mask_paths + "/*.*"))  # masks path corresponding defect images

    # ---------- Pseudo-Normal background ----------
    # image transforms (defect image transformation)
    def trans(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(256, transforms.InterpolationMode("nearest")),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        return transform

    # image transforms (Normal image affine transformation)
    def normal_trans(self, center: tuple[int, int]):
        if self.args.affine_arg == 0:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(256, transforms.InterpolationMode("nearest")),
                ]
            )
        else:
            degrees = self.args.affine_arg[0]
            translate = self.args.affine_arg[1]
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.RandomAffine(
                        degrees=degrees, translate=translate, center=center
                    ),
                    transforms.Resize(256),
                ]
            )

        return transform

    # defect area center
    def center(self, mask: np.ndarray)-> tuple[int, int]:
        _, thresh = cv2.threshold(mask, 100, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        box = cv2.boxPoints(cv2.minAreaRect(cnt))
        center = tuple(np.int0(np.mean(box, axis=0)))
        return center

    # Pseudo-Normal background(PNB) construction
    def pnb(self, img: torch.tensor, mask: torch.tensor, normal: torch.tensor)-> torch.tensor:
        reverse_mask = torch.sub(1, mask)
        pnb = torch.add(torch.mul(img, reverse_mask), torch.mul(normal, mask))
        return pnb

    # --------------------------------------------------

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.img_paths[idx]), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[idx]), cv2.COLOR_BGR2GRAY)
        file_name = self.img_paths[idx].split("/")[-1]

        affine_center = self.center(mask)
        normal_trans = self.normal_trans(center=affine_center)
        trans = self.trans()

        img = trans(img)
        mask = trans(mask)
        normal = normal_trans(self.normal)
        pnb_img = self.pnb(img, mask, normal)

        sample = {
            "image": img,
            "pnb_image": pnb_img,
            "mask": mask,
            "file_name": file_name,
        }

        return sample

    def __len__(self):
        return len(self.img_paths)
