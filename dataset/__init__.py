from .traindataset import TrainDataset
from .testdataset import TestDataset
from torch.utils.data import DataLoader


def get_loader(args):
    if args.istrain == "train":
        dataset = TrainDataset(args)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    else:
        dataset = TestDataset(args)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    return loader
