import os
import glob
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import cv2


class TestDataset(Dataset):
    def __init__(self, args):
        self.args = args
        img_paths = os.path.join(self.args.root, f"images/good")

        self.img_paths = sorted(glob.glob(img_paths + "/*.*"))
        self.mask = cv2.cvtColor(cv2.imread(args.mask_path), cv2.COLOR_BGR2GRAY)

    # data transforms (defect image transformation)
    def trans(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(256, transforms.InterpolationMode("nearest")),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        return transform

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.img_paths[idx]), cv2.COLOR_BGR2RGB)
        mask = self.mask
        file_name = self.img_paths[idx].split("/")[-1]

        trans = self.trans()
        img = trans(img)
        mask = trans(mask)

        sample = {"image": img, "mask": mask, "file_name": file_name}

        return sample

    def __len__(self):
        return len(self.img_paths)
