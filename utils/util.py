import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.backends import cudnn
from random import *
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker
import scipy.misc
import cv2
import yaml
from torch.optim import lr_scheduler
import torch.nn.init as init
import random
from torchvision.utils import save_image, make_grid
import math
import seaborn as sns
sns.set(color_codes=True)


def show_output(x_A, x_B, x_AB, x_BA, m_A, m_B, x_fA, x_fB, x_bA, x_bB, label_A, label_B,
                attrs, num_print):
    batch_size = x_A.size(0)

    (x_A, x_B, x_AB, x_BA) = (var_to_numpy(x_A), var_to_numpy(x_B), var_to_numpy(x_AB), var_to_numpy(x_BA))

    x_fA, x_fB, x_bA, x_bB = (var_to_numpy(x_fA), var_to_numpy(x_fB), var_to_numpy(x_bA), var_to_numpy(x_bB))

    m_A = var_to_numpy(m_A, isReal=False)
    m_B = var_to_numpy(m_B, isReal=False)
    inverse_m_A = 1-m_A
    inverse_m_B = 1-m_B

    for x in range(batch_size):
        if x > (num_print-1) :
            break
        if label_A is not None:
            attributes = ''
            for idx, item in enumerate(label_A[x]):
                if int(item.data) == 1:
                    attributes += str(attrs[idx]) + '  '
            print('content attrs_%d: %s' % (x, attributes))

            attributes = ''
            for idx, item in enumerate(label_B[x]):
                if int(item.data) == 1:
                    attributes += str(attrs[idx]) + '  '
            print('style attrs_%d: %s' % (x, attributes))

        fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(20,15))
        axs[0][0].set_title('A')
        axs[0][0].imshow(x_A[x])
        axs[0][0].axis('off')
        axs[0][1].set_title('B')
        axs[0][1].imshow(x_B[x])
        axs[0][1].axis('off')
        axs[0][2].set_title('A >>> B')
        axs[0][2].imshow(x_AB[x])
        axs[0][2].axis('off')
        axs[0][3].set_title('B >>> A')
        axs[0][3].imshow(x_BA[x])
        axs[0][3].axis('off')


        axs[1][0].set_title('foreground of A')
        axs[1][0].imshow(x_fA[x])
        axs[1][0].axis('off')
        axs[1][1].set_title('foreground of B')
        axs[1][1].imshow(x_fB[x])
        axs[1][1].axis('off')
        axs[1][2].set_title('background of A')
        axs[1][2].imshow(x_bA[x])
        axs[1][2].axis('off')
        axs[1][3].set_title('background of B')
        axs[1][3].imshow(x_bB[x])
        axs[1][3].axis('off')
        

        resized_m_A = scipy.misc.imresize(m_A[x], (x_A.shape[1], x_A.shape[2]), interp='bilinear')

        axs[2][0].set_title('foreground mask for A')
        img = (x_A[x]*255).astype(np.uint8)
        heatmap_img = cv2.applyColorMap(resized_m_A, cv2.COLORMAP_JET)
        fin = cv2.addWeighted(heatmap_img, 0.75, img, 0.9, -20)
        fin = cv2.cvtColor(fin, cv2.COLOR_BGR2RGB)

        divider = make_axes_locatable(axs[2][0])
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        input_mask = axs[2][0].imshow(m_A[x], cmap='jet', interpolation='bilinear')
        tick_limit(plt.colorbar(input_mask, cax=cax, orientation='horizontal'))
        axs[2][0].axis('off')
        axs[2][0].imshow(fin)

        resized_m_B = scipy.misc.imresize(m_B[x], (x_A.shape[1], x_A.shape[2]), interp='bilinear')

        axs[2][1].set_title('foreground mask for B')
        img = (x_B[x]*255).astype(np.uint8)
        heatmap_img = cv2.applyColorMap(resized_m_B, cv2.COLORMAP_JET)
        fin = cv2.addWeighted(heatmap_img, 0.75, img, 0.9, -20)
        fin = cv2.cvtColor(fin, cv2.COLOR_BGR2RGB)

        divider = make_axes_locatable(axs[2][1])
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        input_mask = axs[2][1].imshow(m_B[x], cmap='jet', interpolation='bilinear')
        tick_limit(plt.colorbar(input_mask, cax=cax, orientation='horizontal'))
        axs[2][1].axis('off')
        axs[2][1].imshow(fin)

        resized_im_A = scipy.misc.imresize(inverse_m_A[x], (x_A.shape[1], x_A.shape[2]), interp='bilinear')

        axs[2][2].set_title('background mask for A')
        img = (x_A[x]*255).astype(np.uint8)
        heatmap_img = cv2.applyColorMap(resized_im_A, cv2.COLORMAP_JET)
        fin = cv2.addWeighted(heatmap_img, 0.75, img, 0.9, -20)
        fin = cv2.cvtColor(fin, cv2.COLOR_BGR2RGB)

        divider = make_axes_locatable(axs[2][2])
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        input_mask = axs[2][2].imshow(inverse_m_A[x], cmap='jet', interpolation='bilinear')
        tick_limit(plt.colorbar(input_mask, cax=cax, orientation='horizontal'))
        axs[2][2].axis('off')
        axs[2][2].imshow(fin)

        resized_im_B = scipy.misc.imresize(inverse_m_B[x], (x_A.shape[1], x_A.shape[2]), interp='bilinear')

        axs[2][3].set_title('background mask for B')
        img = (x_B[x]*255).astype(np.uint8)
        heatmap_img = cv2.applyColorMap(resized_im_B, cv2.COLORMAP_JET)
        fin = cv2.addWeighted(heatmap_img, 0.75, img, 0.9, -20)
        fin = cv2.cvtColor(fin, cv2.COLOR_BGR2RGB)

        divider = make_axes_locatable(axs[2][3])
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        input_mask = axs[2][3].imshow(inverse_m_B[x], cmap='jet', interpolation='bilinear')
        tick_limit(plt.colorbar(input_mask, cax=cax, orientation='horizontal'))
        axs[2][3].axis('off')
        axs[2][3].imshow(fin)

        plt.show()

def tick_limit(cb):
    tick_locator = ticker.MaxNLocator(nbins=5)
    cb.locator = tick_locator
    cb.update_ticks()

def var_to_numpy(obj, isReal=True):
    obj = obj.permute(0,2,3,1)

    if isReal:
        obj = (obj+1) / 2
    else:
        obj = obj.squeeze(3)
    obj = torch.clamp(obj, min=0, max=1)
    return obj.data.cpu().numpy()

def ges_Aonfig(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)

def get_scheduler(optimizer, config, iterations=-1):
    if 'LR_POLICY' not in config or config['LR_POLICY'] == 'constant':
        scheduler = None # constant scheduler
    elif config['LR_POLICY'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=config['STEP_SIZE'],
                                        gamma=config['GAMMA'], last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', config['LR_POLICY'])
    return scheduler

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun

def print_network(model, name):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print(name)
    print("The number of parameters: {}".format(num_params))

def concat_input(x, c):
    c = c.view(c.size(0), c.size(1), 1, 1)
    c = c.repeat(1, 1, x.size(2), x.size(3))
    x = torch.cat([x, c], dim=1)
    return x
