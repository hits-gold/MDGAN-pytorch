import argparse
from datetime import datetime

from trainer import trainer
from tester import Tester

def get_arguments():
    parser = argparse.ArgumentParser(description='MDGAN: Mask guided Generation Method for Industrial Defect Images with Non-uniform Structures')
    
    parser.add_argument('--istrain', default='train', help='train or test')
    parser.add_argument('--root', default='./data/MVTec/cable', type=str, help='image source')
    parser.add_argument('--defect_type', default='bent_wire', type=str, help='defect type name(directory name in root)')
    parser.add_argument('--exp', default='exp1', type=str, help='result directory name(ex. MVTec_zipper_combined)')
    parser.add_argument('--epochs', default=400, type=int, help='num epochs')
    parser.add_argument('--save_epoch', default=10, type=int, help='save term')
    parser.add_argument('--batch_size', default=4, type=int, help='batch-size')
    parser.add_argument('--seed', default=1004, type=int, help='random seed')
    
    parser.add_argument('--betas', default=(0.5, 0.999), type=tuple, help='Adam optimizer params (beta1, beta2)')
    parser.add_argument('--lr', default=0.004, type=float, help='Learning rate')
    parser.add_argument('--gammas', default=[10, 15, 10, 10], type=list, help='Loss weights [r, d, g, gp]')

    # Pseudo-noraml bacground(PNB) 
    parser.add_argument('--affine_arg', default=0, help='affine transform arguments => [(degrees1, degrees2), (translate1, translate2)], 0 => no affine transform')
    parser.add_argument('--normal_path', default='./data/MVTec/zipper/test/good/000.png', help='image path of normal for constructing Pseudo-Normal backgroud ')

    # Test
    parser.add_argument('--mask_path', default='./data/MVTec/cable/ground_truth/bent_wire/001_mask.png', help='mask for inference')

    args = parser.parse_args()

    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if args.exp == '':
        args.exp = time_stamp

    return args

if __name__ == "__main__":
    args = get_arguments()
    if args.istrain=='train':
        print('---------------- Train Start ----------------')
        print(datetime.now())
        trainer(args)
    else:
        print('---------------- Test Start ----------------')
        print(datetime.now())
        Tester(args)