import argparse
import os
import numpy as np

import torchvision
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from condgenerators import ConGeneratorResnet
from utils import *
import torch.nn.functional as F
from tqdm import tqdm
from data_loader import CustomDataSet

parser = argparse.ArgumentParser(description='Conditional Adversarial Generator')
parser.add_argument('--data_dir', default='data/ImageNet1k', help='ImageNet Validation Data')
parser.add_argument('--test_dir', default='', help='Testing Data')
parser.add_argument('--test_type', type=str, default='png', help='test sample type')
parser.add_argument('--result_dir', default='', help='Result output')
parser.add_argument('--batch_size', type=int, default=10, help='Batch Size')
parser.add_argument('--model_t',type=str, default= 'res152',  help ='Model under attack : vgg16, vgg19, dense121' )
parser.add_argument('--model_t_ae',type=str, default= 'res152',  help ='Model under attack : vgg16, vgg19, dense121' )
parser.add_argument('--save_tae', type=int, default=0, help='if you want to save the transferable AEs')
args = parser.parse_args()
print(args)

# Normalize (0-1)

n_class = 1000
# GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# Input dimensions: Inception takes 3x299x299
if args.model_t in ['incv3', 'incv4']:
    img_size = 299
else:
    img_size = 224

model_t = load_model(args)
if str(device) != 'cpu':
    model_t = nn.DataParallel(model_t).cuda()
else:
    model_t = nn.DataParallel(model_t)
model_t.eval()

# Setup-Data
data_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
])

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
def normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]
    return t

class_ids = np.array([150, 507, 62, 843, 426, 590, 715, 952])

# Evaluation
sr = np.zeros(len(class_ids))
for idx in range(len(class_ids)):
    transferable_sample = []
    if args.test_type == 'png':
        test_dir = '{}_t{}'.format(args.test_dir, class_ids[idx])
        test_set = datasets.ImageFolder(test_dir, data_transform)
    elif args.test_type == 'npy':
        data_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_file = '{}_t{}_tae.npy'.format(args.model_t_ae, class_ids[idx], )
        test_set = CustomDataSet(os.path.join(args.test_dir, test_file), target=class_ids[idx],
                                 transform=data_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                              pin_memory=True)
    target_acc = 0.
    target_test_size = 0.
    for i, (img, _) in enumerate(test_loader):
        if str(device) != 'cpu':
            img = img.cuda()
        adv_out = model_t(normalize(img.clone().detach()))
        target_acc += torch.sum(adv_out.argmax(dim=-1) == (class_ids[idx])).item()
        target_test_size += img.size(0)
        _transferable_sample = img[adv_out.argmax(dim=-1) == (class_ids[idx])].cpu().detach().numpy()
        transferable_sample.extend(_transferable_sample)
    
    sr[idx] = target_acc / target_test_size
    print('sr: {}'.format(sr))
    print('number of transferable sample:{}'.format(len(transferable_sample)))
    if args.save_tae:
        np.save(os.path.join(args.result_dir, str(args.model_t) + '_t' + str(class_ids[idx]) + '_tae.npy'),
                transferable_sample)

print('target acc:{:.4%}\t target_test_size:{}'.format(sr.mean(), target_test_size))
