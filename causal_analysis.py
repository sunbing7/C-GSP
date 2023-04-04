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
import matplotlib.pyplot as plt
from data_loader import CustomDataSet

parser = argparse.ArgumentParser(description='Conditional Adversarial Generator')
parser.add_argument('--data_dir', default='data/ImageNet1k', help='ImageNet Validation Data')
parser.add_argument('--test_dir', default='', help='Testing Data')
parser.add_argument('--test_type', type=str, default='png', help='test sample type')
parser.add_argument('--result_dir', default='', help='Result output')
parser.add_argument('--batch_size', type=int, default=10, help='Batch Size')
parser.add_argument('--model_t', type=str, default='res152', help='Model under attack : vgg16, vgg19, dense121')
parser.add_argument('--layer', type=int, default=1, help='layer')
parser.add_argument('--option', type=str, default='causal', help='Run options')
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


def solve_causal(data_loader, model, arch, target_class, num_sample, normalize, split_layer=43, use_cuda=True):
    #split the model
    model1, model2 = split_model(model, arch, split_layer=split_layer)

    # switch to evaluate mode
    #model.eval()
    model1.eval()
    model2.eval()

    total_num_samples = 0
    out = []
    do_predict_avg = []
    for input, gt in data_loader:
        if total_num_samples >= num_sample:
            break
        if use_cuda:
            gt = gt.cuda()
            input = input.cuda()

        # compute output
        with torch.no_grad():
            dense_output = model1(normalize(input.clone().detach()))
            ori_output = model2(dense_output)
            #full_output = model(normalize(input.clone().detach()))
            dense_hidden_ = torch.clone(torch.reshape(dense_output, (dense_output.shape[0], -1)))
            # ori_output = filter_model(input)
            do_predict_neu = []
            do_predict = []
            # do convention for each neuron
            for i in range(0, len(dense_hidden_[0])):
                hidden_do = np.zeros(shape=dense_hidden_[:, i].shape)
                dense_output_ = torch.clone(dense_hidden_)
                dense_output_[:, i] = torch.from_numpy(hidden_do)
                dense_output_ = torch.reshape(dense_output_, dense_output.shape)
                if use_cuda:
                    dense_output_ = dense_output_.cuda()
                output_do = model2(dense_output_).cpu().detach().numpy()
                do_predict_neu.append(output_do)  # 4096x32x10
            do_predict_neu = np.array(do_predict_neu)
            do_predict_neu = np.abs(ori_output.cpu().detach().numpy() - do_predict_neu)
            do_predict = np.mean(np.array(do_predict_neu), axis=1)  # 4096x10

        do_predict_avg.append(do_predict)  # batchx4096x11
        total_num_samples += len(gt)
    # average of all baches
    do_predict_avg = np.mean(np.array(do_predict_avg), axis=0)  # 4096x10
    # insert neuron index
    idx = np.arange(0, len(do_predict_avg), 1, dtype=int)
    do_predict_avg = np.c_[idx, do_predict_avg]
    out = do_predict_avg[:, [0, (target_class + 1)]]

    return out


def solve_act(data_loader, model, arch, target_class, num_sample, split_layer=43, use_cuda=True):
    #split the model
    model1, model2 = split_model(model, arch, split_layer=split_layer)

    # switch to evaluate mode
    model1.eval()
    model2.eval()

    total_num_samples = 0
    dense_avg = []
    for input, gt in data_loader:
        if total_num_samples >= num_sample:
            break
        if use_cuda:
            gt = gt.cuda()
            input = input.cuda()

        # compute output
        with torch.no_grad():
            dense_output = model1(input)
            dense_this = np.mean(dense_output.cpu().detach().numpy(), axis=0)  # 4096

        dense_avg.append(dense_this) #batchx4096
        total_num_samples += len(gt)
    # average of all baches
    dense_avg = np.mean(np.array(dense_avg), axis=0)  # 4096x10
    # insert neuron index
    idx = np.arange(0, len(dense_avg), 1, dtype=int)
    dense_avg = np.c_[idx, dense_avg]

    return dense_avg

def split_model(ori_model, model_name, split_layer=43):
    '''
    split given model from the dense layer before logits
    Args:
        ori_model:
        model_name: model name
    Returns:
        splitted models
    '''
    if model_name == 'vgg19':
        if split_layer < 38:
            modules = list(ori_model.children())
            layers = list(modules[0]) + [modules[1]] + list(modules[2])
            module1 = layers[:split_layer]
            module2 = layers[split_layer:38]
            module3 = layers[38:]
            model_1st = nn.Sequential(*module1)
            model_2nd = nn.Sequential(*[*module2, Flatten(), *module3])
        else:
            modules = list(ori_model.children())
            layers = list(modules[0]) + [modules[1]] + list(modules[2])
            module1 = layers[:38]
            moduel2 = layers[38:split_layer]
            module3 = layers[split_layer:]
            model_1st = nn.Sequential(*[*module1, Flatten(), *moduel2])
            model_2nd = nn.Sequential(*module3)
    else:
        return None, None

    return model_1st, model_2nd


def outlier_detection(cmp_list, max_val, verbose=False):
    cmp_list = list(np.array(cmp_list) / max_val)
    consistency_constant = 1.4826  # if normal distribution
    median = np.median(cmp_list)
    mad = consistency_constant * np.median(np.abs(cmp_list - median))  # median of the deviation
    if mad != 0:

        #min_mad = np.abs(np.min(cmp_list) - median) / mad

        # print('median: %f, MAD: %f' % (median, mad))
        # print('anomaly index: %f' % min_mad)

        flag_list = []
        i = 0
        for cmp in cmp_list:
            if cmp_list[i] < median:
                i = i + 1
                continue
            if np.abs(cmp_list[i] - median) / mad > 2:
                flag_list.append((i, cmp_list[i]))
            i = i + 1

        if len(flag_list) > 0:
            flag_list = sorted(flag_list, key=lambda x: x[1])
            if verbose:
                print('flagged label list: %s' %
                      ', '.join(['%d: %2f' % (idx, val)
                                 for idx, val in flag_list]))
    else:

        flag_list = []
        i = 0
        for cmp in cmp_list:
            if cmp_list[i] <= median:
                i = i + 1
                continue
            flag_list.append((i, cmp_list[i]))
            i = i + 1

        if len(flag_list) > 0:
            flag_list = sorted(flag_list, key=lambda x: x[1])
            if verbose:
                print('flagged label list: %s' %
                      ', '.join(['%d: %2f' % (idx, val)
                                 for idx, val in flag_list]))
    return flag_list


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


def plot_multiple(causal, act, normalize=False, t=150):
    # plot the permutation of cmv img and test imgs
    plt_row = 1

    plt_col = 3
    fig, ax = plt.subplots(plt_row, plt_col, figsize=(7 * plt_col, 5 * plt_row), sharex=False, sharey=True)

    ax[0].set_title('causal')
    ax[1].set_title('act')
    ax[2].set_title('compare')

    if normalize:
        causal[:, 1] = causal[:, 1] / np.max(causal[:, 1])

    ax[0].scatter(causal[:, 0].astype(int), causal[:, 1],
                  color='b')
    ax[0].legend()

    if normalize:
        act[:, 1] = act[:, 1] / np.max(act[:, 1])

    ax[1].scatter(act[:, 0].astype(int), act[:, 1],
                  color='b')
    ax[1].legend()

    ax[2].scatter(causal[:, 0].astype(int), causal[:, 1],
                  color='b')
    ax[2].legend()

    ax[2].scatter(act[:, 0].astype(int), act[:, 1],
                  color='y')
    ax[2].legend()

    plt.savefig(args.result_dir + "/plt_" + t + ".png")
    # plt.show()



def plot_single(array, title):
    # plot the permutation of cmv img and test imgs
    plt_row = 1

    plt_col = 1
    fig, ax = plt.subplots(plt_row, plt_col, figsize=(7 * plt_col, 5 * plt_row), sharex=False, sharey=True)

    ax.set_title(title)

    ax.scatter(array[:, 0].astype(int), array[:, 1],
                         color='b')
    ax.legend()

    plt.savefig(args.result_dir + "/plt_" + title + ".png")

def causality_analysis():
    model_t = load_model(args)
    if str(device) != 'cpu':
        model_t = model_t.cuda()
    '''
    if str(device) == 'cuda:0':
        model_t = nn.DataParallel(model_t).cuda()
    else:
        model_t = nn.DataParallel(model_t)
    '''
    model_t.eval()

    # Setup-Data
    data_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]


    def normalize(t):
        t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
        t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
        t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]
        return t


    class_ids = np.array([150, 507, 62, 843, 426, 590, 715, 952])
    #class_ids = np.array([150])

    # Evaluation
    sr = np.zeros(len(class_ids))
    for idx in range(len(class_ids)):
        if args.test_type == 'png':
            test_dir = '{}_t{}'.format(args.test_dir, class_ids[idx])
            test_set = datasets.ImageFolder(test_dir, data_transform)
        elif args.test_type == 'npy':
            data_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            test_file = '{}_t{}_tae.npy'.format(args.model_t, class_ids[idx],)
            test_set = CustomDataSet(os.path.join(args.test_dir, test_file), target=class_ids[idx], transform=data_transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                                  pin_memory=True)
        neuron_ranking = solve_causal(test_loader, model_t, args.model_t, class_ids[idx], 1000, normalize, split_layer=43, use_cuda=(str(device) != 'cpu'))
        # find outstanding neuron neuron_ranking shape: 4096x2
        temp = neuron_ranking
        ind = np.argsort(temp[:, 1])[::-1]
        temp = temp[ind]
        top = outlier_detection(temp[:, 1], max(temp[:, 1]), verbose=False)
        outstanding_neuron = temp[0: len(top)][:, 0]

        print('total:{}, top:{}'.format(len(neuron_ranking), len(outstanding_neuron)))
        np.save(os.path.join(args.result_dir, str(args.model_t) + '_t' + str(class_ids[idx]) + '_outstanding.npy'),
                neuron_ranking)
        np.save(os.path.join(args.result_dir, str(args.model_t) + '_t' + str(class_ids[idx]) + '_causal_top.npy'),
                outstanding_neuron)


def activation_analysis():
    model_t = load_model(args)
    if str(device) != 'cpu':
        model_t = model_t.cuda()

    model_t.eval()

    # Setup-Data
    data_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def normalize(t):
        t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
        t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
        t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]
        return t

    class_ids = np.array([150, 507, 62, 843, 426, 590, 715, 952])
    #class_ids = np.array([150])

    # Evaluation
    sr = np.zeros(len(class_ids))
    for idx in range(len(class_ids)):
        test_dir = '{}_t{}'.format(args.test_dir, class_ids[idx])
        test_set = datasets.ImageFolder(test_dir, data_transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                                  pin_memory=True)
        neuron_ranking = solve_act(test_loader, model_t, args.model_t, class_ids[idx], 1000, split_layer=43,
                                      use_cuda=(str(device) != 'cpu'))
        # find outstanding neuron neuron_ranking shape: 4096x2
        temp = neuron_ranking
        ind = np.argsort(temp[:, 1])[::-1]
        temp = temp[ind]
        top = outlier_detection(temp[:, 1], max(temp[:, 1]), verbose=False)
        outstanding_neuron = temp[0: len(top)][:, 0]

        print('total:{}, top:{}'.format(len(neuron_ranking), len(outstanding_neuron)))
        np.save(os.path.join(args.result_dir, str(args.model_t) + '_t' + str(class_ids[idx]) + '_act.npy'),
                neuron_ranking)
        np.save(os.path.join(args.result_dir, str(args.model_t) + '_t' + str(class_ids[idx]) + '_act_top.npy'),
                outstanding_neuron)


def plot_compare():
    class_ids = np.array([150, 507, 62, 843, 426, 590, 715, 952])
    #class_ids = np.array([150])
    for idx in range(len(class_ids)):
        causal = np.load(os.path.join(args.result_dir, str(args.model_t) + '_t' + str(class_ids[idx]) + '_outstanding.npy'))
        act = np.load(os.path.join(args.result_dir, str(args.model_t) + '_t' + str(class_ids[idx]) + '_act.npy'))
        causal_top = np.load(os.path.join(args.result_dir, str(args.model_t) + '_t' + str(class_ids[idx]) + '_causal_top.npy'))
        act_top = np.load(os.path.join(args.result_dir, str(args.model_t) + '_t' + str(class_ids[idx]) + '_act_top.npy'))
        common = np.intersect1d(causal_top, act_top)
        print('common outstanding neuron: {}'.format(len(common) / len(causal[:, 0])))
        plot_multiple(causal, act, True, t=str(class_ids[idx]))


if __name__ == '__main__':
    if args.option == 'causal':
        causality_analysis()
    elif args.option == 'act':
        activation_analysis()
    elif args.option == 'plot':
        plot_compare()