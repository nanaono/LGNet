from __future__ import print_function

import argparse
from cv2 import normalize
import os

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

#################### Import Python libraries
import math
from math import log10
import numpy as np
import random
import glob
# import matplotlib.pyplot as plt

#################### Import network, dataloader, loss function
from Network.Proposed_v1 import LPIPS, WNormLoss
from Network.op import conv2d_gradfix
# from Network.LaMa import NLayerDiscriminator
from Network.Proposed_v1 import BGSNet_LGNet
from Network.LGNet.model.network import GlobalDis
from Network.losses import Style, Perceptual
from utils.LoadData import Load_ImagesDataset

#################### Parameters
parser = argparse.ArgumentParser(description = 'TRAINING VCM MODEL')
parser.add_argument('--num_epochs', default = 100, type = int)
parser.add_argument('--is_cuda', default = 'cuda', type = str)
parser.add_argument('--batch_size', default = 22, type = int) # default from StyleGAN2
# parser.add_argument("--iter", type=int, default=800000, help="total training iterations") # default from StyleGAN2
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate") # default from StyleGAN2
parser.add_argument("--d_reg_every", type=int, default=16, help="interval of the applying r1 regularization") # default from StyleGAN2
parser.add_argument("--g_reg_every", type=int, default=4, help="interval of the applying path length regularization") # default from StyleGAN2
parser.add_argument("--r1", type=float, default=10, help="weight of the r1 regularization") # default from StyleGAN2
parser.add_argument("--path_regularize", type=float, default=2, help="weight of the path length regularization") # default from StyleGAN2
parser.add_argument('--lr_decay_interval', type=int, default=50,
                        help='decay learning rate every N epochs(default: 100)')
parser.add_argument(
    "--local_rank", type=int, default=0, help="local rank for distributed training"
)

#################### Required Functions for Training
BCE = nn.BCELoss()

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // args.lr_decay_interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def our_nan_to_num(input, nan=0.0, posinf=None, neginf=None, *, out=None): # pylint: disable=redefined-builtin
        assert isinstance(input, torch.Tensor)
        if posinf is None:
            posinf = torch.finfo(input.dtype).max
        if neginf is None:
            neginf = torch.finfo(input.dtype).min
        assert nan == 0
        return torch.clamp(input.unsqueeze(0).nansum(0), min=neginf, max=posinf, out=out)

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False

    if not dist.is_initialized():
        return False

    return True

def get_rank():

    if not is_dist_avail_and_initialized():
        return 0

    return dist.get_rank()

def is_main_process():

    return get_rank() == 0
    
def get_world_size():
    if not dist.is_available():
        return 1

    if not dist.is_initialized():
        return 1

    return dist.get_world_size()

def reduce_sum(tensor):
    if not dist.is_available():
        return tensor

    if not dist.is_initialized():
        return tensor

    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    return tensor

def reduce_loss_dict(loss_dict):
    world_size = get_world_size()

    if world_size < 2:
        return loss_dict

    with torch.no_grad():
        keys = []
        losses = []

        for k in sorted(loss_dict.keys()):
            keys.append(k)
            losses.append(loss_dict[k])

        losses = torch.stack(losses, 0)
        dist.reduce(losses, dst=0)

        if dist.get_rank() == 0:
            losses /= world_size

        reduced_losses = {k: v for k, v in zip(keys, losses)}

    return reduced_losses

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()

def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty

def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss
def g_lsgan(real_img, label):
    errG = 0.5 * torch.mean((real_img - label)**2)
    return errG
def d_lsgan(real_pred, fake_pred, label_r, label_f):
    err_real = 0.5 * torch.mean((real_pred - label_r)**2)
    err_fake = 0.5 * torch.mean((fake_pred - label_f)**2)

    return err_real + err_fake

def lsgan_score(real_img, label):
    score = 0.5 * torch.mean((real_img - label)**2)
    return score

def g_path_regularize(fake_img, latents, mean_path_length, decay=0.001):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    # with conv2d_gradfix.no_weight_gradients():
    #     grad = torch.autograd.grad(outputs=[(fake_img * noise).sum()], inputs=[latents], create_graph=True, only_inputs=True)[0]

    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    ) 

    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths

def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
	return d_filt

# def Discriminator_backward(Dnet, real, fake, label):
#     logits_real = Dnet(real).view(-1)
#     errD_real = lsgan_score(logits_real, label)
#     errD_real.backward()

#     label.fill_(fake_label)
#     logits_fake = Dnet(fake.detach()).view(-1)
#     errD_fake = lsgan_score(logits_fake, label)
#     errD_fake.backward()

#     errD = errD_fake + errD_real

#     return logits_real, logits_fake, errD

def Loss_Adv(Dnet, fake, label):
    logits_fake_update = Dnet(fake).view(-1)
    err_adv = BCE(logits_fake_update, label)

    return logits_fake_update, err_adv


def dis_forward(netD, ground_truth, x_inpaint): # discriminator, gt, output
    assert ground_truth.size() == x_inpaint.size()
    batch_size = ground_truth.size(0)
    batch_data = torch.cat([ground_truth, x_inpaint], dim=0)
    batch_output = netD(batch_data)
    real_pred, fake_pred = torch.split(batch_output, batch_size, dim=0)

    return real_pred, fake_pred
#cmd : CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 train_LGNet.py

#################### Main code
if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)

    opt = parser.parse_args()

    NUM_EPOCHS = opt.num_epochs
    BATCH = opt.batch_size
    DEVICE = opt.is_cuda
    LR = opt.lr

    ### Distributed
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    # print(world_size)
    
    torch.distributed.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
    torch.cuda.set_device(local_rank)
    dist.barrier()

    ### Train DATA path
    input_path = '/media/hdd/daole/VCM_Proposed/data/New/synthesis_COCO/*.png'
    mask_path = '/media/hdd/daole/VCM_Proposed/data/New/mask_COCO/*.png'
    gt_path = '/media/hdd/daole/VCM_Proposed/data/New/gt_COCO/*.png'
    INPUT = sorted(glob.glob(input_path))
    GT = sorted(glob.glob(gt_path))
    MASK = sorted(glob.glob(mask_path))

    data_train = Load_ImagesDataset(INPUT, GT, MASK, is_trained=True)

    data_train_loader = torch.utils.data.DataLoader(data_train,
                                                    batch_size = BATCH,
                                                    sampler=torch.utils.data.distributed.DistributedSampler(data_train, shuffle=True),
                                                    drop_last=True,
                                                    num_workers=8)

    ### Define Generator Network
    BG_Generator = BGSNet_LGNet()#BGSNet_LGNet 
    BG_Generator.cuda()
    ### Loading average style vectors of pre-trained styleGAN
    discriminator = GlobalDis() #original globalD
    discriminator.cuda()
    
    # ### Continue training ...
    # discriminator.load_state_dict(state_dict1["d"])

    ### Distribute
    BG_Generator = nn.parallel.DistributedDataParallel(BG_Generator, broadcast_buffers=False, find_unused_parameters=True)
    discriminator = nn.parallel.DistributedDataParallel(discriminator, broadcast_buffers=False)

    ### Loss function
    L1_loss = nn.L1Loss()
    # L2_loss = nn.MSELoss()
    # w_loss = WNormLoss()
    style_loss = Style()
    lpips_loss = Perceptual()
    # lpips_loss = LPIPS(net_type='vgg').cuda().eval()
    # hrf_loss = ResNetPL().cuda().eval()
    ### Define Optimizers
    # g_reg_ratio = opt.g_reg_every / (opt.g_reg_every + 1)
    # d_reg_ratio = opt.d_reg_every / (opt.d_reg_every + 1)

    # params = list(BG_Generator.encoder.parameters())
    # params += list(BG_Generator.decoder.parameters())
    g_optim = optim.Adam(BG_Generator.parameters(), lr = LR, betas=(0.5, 0.999) )
    d_optim = optim.Adam(discriminator.parameters(), lr = LR, betas=(0.5, 0.999))
    
    ### Continue training ...
    # g_optim.load_state_dict(state_dict1["g_optim"])
    # d_optim.load_state_dict(state_dict1["d_optim"])

    ### Training and Testing section
    mean_path_length = 0
    mean_path_length_avg = 0
    # accum = 0.5 ** (32 / (10 * 1000))

    r1_loss = torch.tensor(0.0).cuda()
    path_loss = torch.tensor(0.0).cuda()
    path_lengths = torch.tensor(0.0).cuda()

    loss_dict = {}
    iteration = 0

    for epoch in range(NUM_EPOCHS):
        for phase in ['train', 'val']:
            if phase == 'train':
                
                BG_Generator.train()
                discriminator.train()

                # adjust_learning_rate(opt, g_optim, epoch)
                count = 0

                data_train_loader.sampler.set_epoch(epoch)
                # torch.autograd.set_detect_anomaly(True)

                for i, data in enumerate(data_train_loader, 0):
                    # Load images
                    input = data[0].cuda()
                    gt = data[1].cuda()
                    mask = data[2].cuda()
                    inv_mask = 1. - mask
                        


                    ###########################
                    ### Train Discriminator ###
                    ###########################
                    requires_grad(discriminator, True)
                    requires_grad(BG_Generator, False)
                    # d_optim.zero_grad()

                    fake_img = BG_Generator(input, mask) 

                    # fake_pred = discriminator(fake_img) #.detach -> no backpropogation
                    # real_pred = discriminator(gt * inv_mask + input*mask)
                    refine_real, refine_fake = dis_forward(discriminator, gt, fake_img)
                    
                    d_loss_rel=torch.mean(torch.log(1.0 + torch.abs(refine_real - refine_fake)))
                    d_loss_loren=(torch.mean(
                    torch.nn.ReLU()(1.0 - (refine_real - torch.mean(refine_fake)))) + torch.mean(
                    torch.nn.ReLU()(1.0 + (refine_fake - torch.mean(refine_real))))) / 2
                    # Discriminator loss
                    d_loss = d_loss_rel+d_loss_loren
                    
                    # Save loss
                    # loss_dict["d"] = d_loss # discrimator loss

                    # Backpropagation (Optimizing)
                    discriminator.zero_grad()
                    d_loss.backward()
                    d_optim.step()

                    ###########################
                    ##### Train Generator #####
                    ###########################
                    requires_grad(BG_Generator, True)
                    requires_grad(discriminator, False)
                    # g_optim.zero_grad()

                    fake_img = BG_Generator(input, mask) 
                    fake_pred = discriminator(fake_img)
                    g_loss = g_nonsaturating_loss(fake_pred)

                    l1_loss = L1_loss(fake_img * inv_mask, gt * inv_mask)
                    # cycle_loss = L1_loss(compressed_img * inv_mask, input * inv_mask)
                    # lpips_loss_ = lpips_loss(fake_img* inv_mask + input*mask, gt * inv_mask + input*mask)
                    # style_loss_ = style_loss(fake_img* inv_mask + input*mask, gt * inv_mask + input*mask)
                    g_loss_loren = torch.mean(torch.log(1.0 + torch.abs(refine_fake - refine_real)))
                    g_loss_rel = (torch.mean(
                        torch.nn.ReLU()(1.0 + (refine_real - torch.mean(refine_fake)))) + torch.mean(
                        torch.nn.ReLU()(1.0 - (refine_fake - torch.mean(refine_real))))) 
                    tot_g_loss =  l1_loss*1.2+g_loss_rel+g_loss_loren
                    

                    
                    # # Save loss
                    # loss_dict["pixel"] = pixel_loss
                    # loss_dict["g"] = g_loss
                    # loss_dict["g_total"] = tot_loss

                    # Backpropagation (Optimizing)
                    BG_Generator.zero_grad()
                    tot_g_loss.backward()
                    g_optim.step()

                    ###########################
                    ###########################
                    ###########################

                    loss_reduced = reduce_loss_dict(loss_dict)

                    d_loss_ = loss_reduced["d"].mean().item()
                    g_loss_ = loss_reduced["g"].mean().item()
                    p_loss_ = loss_reduced["pixel"].mean().item()
                    # r1_ = loss_reduced["r1"].mean().item()
                    real_score_ = loss_reduced["real_score"].mean().item()
                    fake_score_ = loss_reduced["fake_score"].mean().item()


                    if is_main_process():
                        if count % 50 == 0:
                            print('[%d/%d][%d/%d]---------------------------------------------' %(epoch, NUM_EPOCHS, count, len(data_train_loader)))
                            print('-----Discriminator-----')
                            print('D loss: %.4f\t Real score: %.4f\t Fake score: %.4f' %(
                                    d_loss_, real_score_, fake_score_))

                            print('-----Generator-----')
                            print('Tot_g_loss: %.4f \t   ' %(
                                    tot_g_loss.mean().item(),))
                            
                            # Save images
                            BG_Generator.eval()
                            with torch.no_grad():
                                inv_mask_ = 1 - mask[0:4,:,:,:]
                                
                                fake_img = BG_Generator(input[0:4,:,:,:], mask[0:4,:,:,:])
                                # fake_img, _ = BG_Generator(input[0:BATCH,:,:,:], mask[0:BATCH,:,:,:], wbar[0:BATCH,:,:])
                                # fake_img = fake_img * inv_mask_ + input[0:1,:,:,:] * mask[0:1,:,:,:]
                                # fake_img, _ = BG_Generator(input[0:1,:,:,:], mask[0:1,:,:,:])

                                fake_img_save = (fake_img + 1) / 2
                                input_save = (input[0:4,:,:,:] + 1) / 2
                                
                            utils.save_image(mask[0:4,:,:,:], f"mask_LGNet.png", nrow=2, normalize=False)
                            utils.save_image(input_save, f"input_LGNet.png", nrow=2, normalize=False)
                            utils.save_image(fake_img_save, f"out_LGNet.png", nrow=2, normalize=False)

                            BG_Generator.train()
                        
                    ###########################
                    ###########################
                    ###########################
                    iteration = iteration + 1
                    count = count + 1
                
                if is_main_process():
                    # Save after each epoch
                    torch.save(
                        {
                        "g": BG_Generator.module.state_dict(),
                        "d": discriminator.module.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict() 
                        }, 
                        "Weights/Weights_LGNet/net_iter_%d_ep_%d.pth" %(iteration, epoch)
                    )





    
