from __future__ import print_function

import argparse
# from cv2 import normalize
# from matplotlib import use

#################### Import Pytorch libraries
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
from torch import distributed as dist, nan_to_num_
import torch.utils.tensorboard as tfboard
from torchvision import utils
# import dct
import torch
#################### Import Python libraries
import math
from math import log10
import numpy as np
import random
import glob
from PIL import Image
# import matplotlib.pyplot as plt
#################### Import network, dataloader, loss function
from Network.Proposed_v1 import *
from utils.LoadData import Load_TestImagesDataset

from skimage.metrics import structural_similarity as ssim

# from skimage.metrics import structural_similarity as ssim
# from skimage.metrics import mean_squared_error

#################### Parameters
parser = argparse.ArgumentParser(description = 'Testing VCM MODEL')
parser.add_argument('--is_cuda', default = 'cuda:0', type = str)

#################### Main code
if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)

    opt = parser.parse_args()

    DEVICE = opt.is_cuda
   

    ### Test DATA path
 
    input_path = '/media/hdd/daole/VCM_Proposed/data/test_in_COCO/*.png'
    mask_path = '/media/hdd/daole/VCM_Proposed/data/test_mask_COCO/*.png'
    gt_path = '/media/hdd/daole/VCM_Proposed/data/test_gt_COCO/*.png'
    

    INPUT = sorted(glob.glob(input_path))
    GT = sorted(glob.glob(gt_path))
    MASK = sorted(glob.glob(mask_path))

    data_val = Load_TestImagesDataset(INPUT, GT, MASK)    
    data_val_loader = torch.utils.data.DataLoader(data_val,
                                                  batch_size = 1, shuffle = False, num_workers = 4)
    
    ### Define Generator Network
    BG_Generator = BGSNet_LaMa() 
    # BG_Decoder = BGSNet_Our_v1D()
    
    BG_Generator.to(DEVICE)
    # BG_Decoder.to(DEVICE)

    net_path_ = '/media/hdd/naeun/VCM_Proposed_2022/net_iter_291600_ep_99.pth'
    state_dict_ = torch.load(net_path_, map_location = lambda s, l: s)
    BG_Generator.load_state_dict(state_dict_['g'])
    # wbar = BG_Generator.decoder.mean_latent(int(1e5))
    #wbar = wbar[0].detach()
    BG_Generator.eval()

    net_path = '/media/hdd/naeun/VCM_Proposed_2022/net_iter_291600_ep_99.pth'
    state_dict = torch.load(net_path, map_location = lambda s, l: s)
    BG_Generator.load_state_dict(state_dict["g"])
    
    with torch.no_grad():
        BG_Generator.eval()
        for i, test in enumerate(data_val_loader, 0):
            # load images
            input_ = test[0].to(DEVICE)
            gt_ = test[1].to(DEVICE)
            mask_ = test[2].to(DEVICE)
           
            # ################################### #GLEAN-based_v1
            output_ = BG_Generator(input_, mask_)
        
            # Save images
            output_ = (output_ + 1) / 2
            
          
            utils.save_image(output_, f"/media/hdd/naeun/VCM_Proposed_2022/test_result_naeun/{str(i).zfill(6)}.png", nrow=1, normalize=False)






    
