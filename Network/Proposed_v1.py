import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
import numpy as np
from collections import namedtuple
from Network.StyleGAN2 import Generator_StyleOp, Generator
from Network.lpips_network import get_network, LinLayers
from Network.lpips_utils import get_state_dict
from Network.AOTGAN_network import *
from Network.MADF_network import *
from Network.LaMa import *
from Network.FCF_network import *
from Network.LGNet.model.network import *
class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
	""" A named tuple describing a ResNet block. """


def get_block(in_channel, depth, num_units, stride=2):
	return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
	if num_layers == 50:
		blocks = [
			get_block(in_channel=64, depth=64, num_units=3),
			get_block(in_channel=64, depth=128, num_units=4),
			get_block(in_channel=128, depth=256, num_units=14),
			get_block(in_channel=256, depth=512, num_units=3)
		]
	elif num_layers == 100:
		blocks = [
			get_block(in_channel=64, depth=64, num_units=3),
			get_block(in_channel=64, depth=128, num_units=13),
			get_block(in_channel=128, depth=256, num_units=30),
			get_block(in_channel=256, depth=512, num_units=3)
		]
	elif num_layers == 152:
		blocks = [
			get_block(in_channel=64, depth=64, num_units=3),
			get_block(in_channel=64, depth=128, num_units=8),
			get_block(in_channel=128, depth=256, num_units=36),
			get_block(in_channel=256, depth=512, num_units=3)
		]
	else:
		raise ValueError("Invalid number of layers: {}. Must be one of [50, 100, 152]".format(num_layers))
	return blocks
class SEModule(nn.Module):
	def __init__(self, channels, reduction):
		super(SEModule, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
		self.relu = nn.ReLU(inplace=True)
		self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		module_input = x
		x = self.avg_pool(x)
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		x = self.sigmoid(x)
		return module_input * x


class bottleneck_IR(nn.Module):
	def __init__(self, in_channel, depth, stride):
		super(bottleneck_IR, self).__init__()
		if in_channel == depth:
			self.shortcut_layer = nn.MaxPool2d(1, stride)
		else:
			self.shortcut_layer = nn.Sequential(
				nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
				nn.BatchNorm2d(depth)
			)
		self.res_layer = nn.Sequential(
			nn.BatchNorm2d(in_channel),
			nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), nn.PReLU(depth),
			nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False), nn.BatchNorm2d(depth)
		)

	def forward(self, x):
		shortcut = self.shortcut_layer(x)
		res = self.res_layer(x)
		return res + shortcut


class bottleneck_IR_SE(nn.Module):
	def __init__(self, in_channel, depth, stride):
		super(bottleneck_IR_SE, self).__init__()
		if in_channel == depth:
			self.shortcut_layer = nn.MaxPool2d(1, stride)
		else:
			self.shortcut_layer = nn.Sequential(
				nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
				nn.BatchNorm2d(depth)
			)
		self.res_layer = nn.Sequential(
			nn.BatchNorm2d(in_channel),
			nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
			nn.PReLU(depth),
			nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
			nn.BatchNorm2d(depth),
			SEModule(depth, 16)
		)

	def forward(self, x):
		shortcut = self.shortcut_layer(x)
		res = self.res_layer(x)
		return res + shortcut

class GradualStyleBlock(nn.Module):
    def __init__(self, in_c, out_c, spatial):
        super(GradualStyleBlock, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                nn.Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x
    

class WNormLoss(nn.Module):

	def __init__(self, start_from_latent_avg=True):
		super(WNormLoss, self).__init__()
		self.start_from_latent_avg = start_from_latent_avg

	def forward(self, latent, latent_avg=None):
		if self.start_from_latent_avg:
			latent = latent - latent_avg
		return torch.sum(latent.norm(2, dim=(1, 2))) / latent.shape[0]


class GradualStyleEncoder(nn.Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(GradualStyleEncoder, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = nn.Sequential(nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),#input_nc =3
                                      nn.BatchNorm2d(64),
                                      nn.PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = nn.Sequential(*modules)

        self.styles = nn.ModuleList()
        # self.style_count = opts.n_styles
        self.style_count = int(math.log(256, 2)) * 2 - 2#256: output_size
        self.coarse_ind = 3
        self.middle_ind = 7
        # print(self.style_count)
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y
      
    def forward(self, x):
        x = self.input_layer(x)
        # print('x256', x.size())
        feature256 = x
        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            # print('i',i, 'x', x.size())
            if i == 2:
                feature128 = x
            elif i == 6:
                c1 = x
                # print('c1', c1.size())
                feature64 = c1
            elif i == 20:
                c2 = x
                # print('c2', c2.size())
            elif i == 23:
                c3 = x
                # print('c3', c3.size())

        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))

        p2 = self._upsample_add(c3, self.latlayer1(c2))
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))

        p1 = self._upsample_add(p2, self.latlayer2(c1))
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))
        
        out = torch.stack(latents, dim=1)

        return out, feature64, feature128, feature256
        # return out
##############################################################################################
##############################################################################################

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):

        out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )

class BGSNet_StyleOp(nn.Module):
    def __init__(self):
        super(BGSNet_StyleOp, self).__init__()
        self.encoder = GradualStyleEncoder(50 ,'ir_se')
        self.decoder = Generator_StyleOp()

    def forward(self, input, mask, wbar):
        codes, feat64_enc, feat128_enc, feat256_enc  = self.encoder(input)
        output, feat64, feat128, feat256 = self.decoder([codes+wbar],input, input_is_latent = True, return_features=True) 
        # output = roi + output * inv_mask
        return output, codes, feat64, feat128, feat256, feat64_enc

class LPIPS(nn.Module):
    r"""Creates a criterion that measures
    Learned Perceptual Image Patch Similarity (LPIPS).
    Arguments:
        net_type (str): the network type to compare the features:
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """
    def __init__(self, net_type: str = 'alex', version: str = '0.1'):

        assert version in ['0.1'], 'v0.1 is only supported now'

        super(LPIPS, self).__init__()

        # pretrained network
        self.net = get_network(net_type).to("cuda")

        # linear layers
        self.lin = LinLayers(self.net.n_channels_list).to("cuda")
        self.lin.load_state_dict(get_state_dict(net_type, version))

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        feat_x, feat_y = self.net(x), self.net(y)

        diff = [(fx - fy) ** 2 for fx, fy in zip(feat_x, feat_y)]
        res = [l(d).mean((2, 3), True) for d, l in zip(diff, self.lin)]

        return torch.sum(torch.cat(res, 0)) / x.shape[0] 

class Mask_Encoder(nn.Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(Mask_Encoder, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        # assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        
        unit_module = bottleneck_IR_SE_Mask
        self.input_layer = nn.Sequential(nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),#input_nc =3
                                      nn.BatchNorm2d(64),
                                      MaskUpdate(0.8))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = nn.Sequential(*modules)

        self.styles = nn.ModuleList()
        # self.style_count = opts.n_styles
        self.style_count = int(math.log(256, 2)) * 2 - 2#256: output_size
        self.coarse_ind = 3
        self.middle_ind = 7
        # print(self.style_count)
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock_Mask(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock_Mask(512, 512, 32)
            else:
                style = GradualStyleBlock_Mask(512, 512, 64)
            self.styles.append(style)

        self.latlayer1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0),
            MaskUpdate(0.8)
            # GaussActivationv2(1.1, 2.0, 1.0, 1.0)
        )
        self.latlayer2 = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0),
            MaskUpdate(0.8)
            # GaussActivationv2(1.1, 2.0, 1.0, 1.0)
        )

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        x = self.input_layer(x)
        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))

        p2 = self._upsample_add(c3, self.latlayer1(c2))
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))

        p1 = self._upsample_add(p2, self.latlayer2(c1))
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))
        
        out = torch.stack(latents, dim=1)

        return out

class BGSNet_StyleOp_Mask(nn.Module):#2022.09.17; psp encoder + pretrained + proposed StyleOp
    def __init__(self):
        super(BGSNet_StyleOp_Mask, self).__init__()
        self.Generator = BGSNet_StyleOp_withWMask()
        self.encoder_mask = Mask_Encoder(50 ,'ir_se')

    def forward(self, input, mask, wbar):
        inv_mask = 1. - mask

        roi = input * mask
        
        w_m = self.encoder_mask(inv_mask)
        fake_img, latent, win = self.Generator(input, mask, wbar, w_m)
        output = roi + fake_img * inv_mask
        return output, latent, win

class BGSNet_StyleOp_withWMask(nn.Module):#2022.09.17; psp encoder + pretrained + proposed StyleOp
    def __init__(self):
        super(BGSNet_StyleOp_withWMask, self).__init__()
        self.encoder = GradualStyleEncoder(50 ,'ir_se')
        self.decoder = Generator_StyleOp()
        # self.load_weights()

    # def load_weights(self):
    #         net_path = 'StyleGAN_Weights/net_iter_221288_ep_51.pth'
    #         state_dict = torch.load(net_path, map_location = lambda s, l: s)
    #         self.decoder.load_state_dict(state_dict["g"])

    def forward(self, input, mask, wbar, wmask):
        inv_mask = 1. - mask

        roi = input * mask
        
        win = self.encoder(input)
        codes = win * wmask
        fake_img, latent = self.decoder([codes + wbar], input, input_is_latent = True, return_latents=True) 
        
        output = roi + fake_img * inv_mask
        return output, latent, win

class BGSNet_StyleOp_MaskInStyleOp(nn.Module):#2022.09.17; psp encoder + pretrained (mask in StyleOp) + proposed StyleOp 
    def __init__(self):
        super(BGSNet_StyleOp_MaskInStyleOp, self).__init__()
        self.encoder = GradualStyleEncoder(50 ,'ir_se')
        self.decoder = Generator_StyleOp_Mask() #input should be invert mask

    def forward(self, input, mask, wbar):
        inv_mask = 1. - mask

        roi = input * mask
        
        codes, feat64_enc, feat128_enc, feat256_enc  = self.encoder(input)
        fake_img, feat64, feat128, feat256 = self.decoder([codes+wbar],input,inv_mask, input_is_latent = True, return_features=True) 
        fake_img = roi + fake_img * inv_mask
        return fake_img, codes, feat64, feat128, feat256, feat64_enc

class BGSNet_Proposed(nn.Module):#output features from v1 algorithm, GLEAN decoder,
    def __init__(self):
        super(BGSNet_Proposed, self).__init__()
        self.decoder = Decoder(256, 256) 

    def forward(self, input, mask, feat64_enc, generator_features):
        inv_mask = 1. - mask

        roi = input * mask
        output= self.decoder(feat64_enc, generator_features)
        output = roi + output * inv_mask
        return output

class Decoder(nn.Module):#adding conv_fused
    def __init__(self,
                 in_size,
                 out_size,
                 img_channels=3
                ):

        super().__init__()
        self.in_size = in_size
        self.style_channels = 512

        channels = {
            64: 512,
            128: 256,
            256: 128,
        }

        # decoder
        decoder_res = [
            2**i
            for i in range(int(np.log2(64)), int(np.log2(out_size) + 1))
        ]
        # print(decoder_res)
        self.decoder = nn.ModuleList()
        for res in decoder_res:
            if res == 64:
                in_channels = 256
            else:
                in_channels = channels[res]//2

            if res < out_size:
                out_channels = channels[res * 2]
                self.decoder.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(in_channels, out_channels,
                                        kernel_size=4, stride=2,
                                        padding=1),
                        # nn.LeakyReLU(negative_slope=0.2, inplace=True)
                        nn.PReLU(out_channels)
                        ))
            else:
                self.decoder.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, 64, 3, 1, 1),
                        # nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.PReLU(64),
                        nn.Conv2d(64, img_channels, 3, 1, 1),
                        nn.Tanh()))   
        self.conv_fused1 = nn.Sequential(
            nn.Conv2d(512+128, 256, 3, 1, 1),
            nn.PReLU()
            )                         
        self.conv_fused2 = nn.Sequential(
            nn.Conv2d(256 *2, 128, 3, 1, 1),
            nn.PReLU()
            )     
        self.conv_fused3 = nn.Sequential(
            nn.Conv2d(128*2, 64, 3, 1, 1),
            nn.PReLU()
            ) 
                             
    def forward(self,encoder_features, generator_features ):
        """Forward function.
        Args:
            lq (Tensor): Input LR image with shape (n, c, h, w).
        Returns:
            Tensor: Output HR image.
        """
        hr = torch.cat([encoder_features[0], generator_features[0]], dim=1)

        hr = self.conv_fused1(hr)
        for i, block in enumerate(self.decoder):
            if i == 1:
                hr = torch.cat([hr, generator_features[i]], dim=1)
                
                hr = self.conv_fused2(hr)

            elif i == 2:
                hr = torch.cat([hr, generator_features[i]], dim=1)
                hr = self.conv_fused3(hr)
            hr = block(hr)
        return hr         

class BGSNet_psp_UsingDecoder(nn.Module):#follows psp encoder,  using pretrained styleGAN
    def __init__(self):
        super(BGSNet_psp_UsingDecoder, self).__init__()
        self.encoder = GradualStyleEncoder(50 ,'ir_se')
        self.decoder = Generator(256,512,8)

    def forward(self, input, mask, wbar):
        inv_mask = 1. - mask

        roi = input * mask
        
        codes, feat64_enc, feat128_enc, feat256_enc = self.encoder(input)
        fake_img, feat64, feat128, feat256 = self.decoder([codes+ wbar],input_is_latent = True, return_features=True) 
        output = roi + fake_img * inv_mask
        return output, codes, feat64, feat128, feat256, feat64_enc, feat128_enc, feat256_enc 

class BGSNet_MADF(nn.Module):
    def __init__(self):
        super(BGSNet_MADF, self).__init__()
        self.generator = MADFNet()

    def forward(self, input, mask):
        # input = (input + 1)/2
        inv_mask = 1. - mask

        roi = input * mask

        # x = torch.cat((input,mask[:,0,:,:]),1)
        # x = mask[:,0:1,:,:]
        # print(x.size())

        fake_img = self.generator(input, mask) 
        # print(fake_img[0])
        # print('roi',roi.size())
        output = roi + fake_img[0] * inv_mask
        # utils.save_image(fake_img[0:4,:,:,:], f"masked_img.png", nrow=1, normalize=False)
        return output    


class BGSNet_LaMa(nn.Module):
    def __init__(self):
        super(BGSNet_LaMa, self).__init__()
        self.generator = LaMa_model()

    def forward(self, input, mask):
        inv_mask = 1. - mask

        roi = input * mask

        # x = torch.cat((input,mask[:,0,:,:]),1)
        # x = mask[:,0:1,:,:]
        # print(x.size())
        # masked_img = input*mask
        # utils.save_image(masked_img, f"masked_img.png", nrow=1, normalize=True)
        fake_img = self.generator(torch.cat((input,inv_mask[:,0:1,:,:]),1)) 
        # print(fake_img.size())
        # print('roi',roi.size())
        output = roi + fake_img * inv_mask
        return output  
    
    # FCF_train
class BGSNet_FCF(nn.Module):
    def __init__(self, c_dim=0):
        super(BGSNet_FCF, self).__init__()
        self.generator = Generator_FCF(z_dim=512, w_dim=512, c_dim = c_dim, img_resolution = 256)

    def forward(self, input, mask):
        inv_mask = 1. - mask

        roi = input * mask

        # x = torch.cat((input,mask[:,0,:,:]),1)
        # x = mask[:,0:1,:,:]
        # print(x.size())
        # masked_img = input*mask
        # utils.save_image(masked_img, f"masked_img.png", nrow=1, normalize=True)
        
        # print(input.size())
        # print(inv_mask.size())
        fake_img = self.generator(torch.cat([input,mask[:,0:1,:,:]], dim=1), None) 
        # print(fake_img.size())
        # print('roi',roi.size())
        output = roi + fake_img * inv_mask

        return output 

    #FCF_test
class BGNet_FCF_inpainting(nn.Module):
    def __init__(self):
        super(BGNet_FCF_inpainting, self).__init__()
        self.generator = Generator_FCF()
        self.encoder = EncoderNetwork()
        #Encoder_Block from FCF

    def forward(self, input, mask):
        inv_mask = 1. - mask

        roi = input * mask

        # x = torch.cat((input,mask[:,0,:,:]),1)
        # x = mask[:,0:1,:,:]
        # print(x.size())
        # masked_img = input*mask
        # utils.save_image(masked_img, f"masked_img.png", nrow=1, normalize=True)
        z_enc = self.encoder(torch.cat((input,inv_mask[:,0:1,:,:]),1)) 
        output = self.generator() 
        fake_img = self.generator(torch.cat((input,inv_mask[:,0:1,:,:]),1))
        # print(fake_img.size())
        # print('roi',roi.size())
        output = roi + fake_img * inv_mask
        return output  
   
    # train_LGNet
    class BGSNet_LGNet(nn.Module):
        def __init__(self):
            super(BGSNet_LGNet, self).__init__()
            self.generator = Generator()

        def forward(self, input, mask):
            # input = (input + 1)/2
            inv_mask = 1. - mask

            roi = input * mask

            # x = torch.cat((input,mask[:,0,:,:]),1)
            # x = mask[:,0:1,:,:]
            # print(x.size())

            fake_img = self.generator(input, mask) 
            # print(fake_img[0])
            # print('roi',roi.size())
            output = roi + fake_img[0] * inv_mask
            # utils.save_image(fake_img[0:4,:,:,:], f"masked_img.png", nrow=1, normalize=False)
            return output  
    
    # def forward(self, input, mask):
    #     inv_mask = 1. - mask
    #     feat64_enc = self.generator(torch.cat((input,inv_mask[:,0:1,:,:]),1)) 
    #     return feat64_enc    



class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).' % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)

class InpaintGenerator(BaseNetwork):
    def __init__(self):  # 1046
        super(InpaintGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(4, 64, 7),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(True)
        )
        rates = '1+2+4+8'
        rates = list(map(int, list(rates.split('+'))))
        self.middle = nn.Sequential(*[AOTBlock(256, rates) for _ in range(8)])

        self.decoder = nn.Sequential(
            UpConv(256, 128),
            nn.ReLU(True),
            UpConv(128, 64),
            nn.ReLU(True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1)
        )

        self.init_weights()

    def forward(self, x, mask):
        x = torch.cat([x, mask], dim=1)
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = torch.tanh(x)
        return x

class UpConv(nn.Module):
    def __init__(self, inc, outc, scale=2):
        super(UpConv, self).__init__()
        self.scale = scale
        self.conv = nn.Conv2d(inc, outc, 3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True))


class AOTBlock(nn.Module):
    def __init__(self, dim, rates):
        super(AOTBlock, self).__init__()
        self.rates = rates
        # print('rates', rates)
        for i, rate in enumerate(rates):
            # print(i, ': rates', rates)
            self.__setattr__(
                'block{}'.format(str(i).zfill(2)), 
                nn.Sequential(
                    nn.ReflectionPad2d(rate),
                    nn.Conv2d(dim, dim//4, 3, padding=0, dilation=rate),
                    nn.ReLU(True)))
        self.fuse = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, dilation=1))
        self.gate = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, dilation=1))

    def forward(self, x):
        out = [self.__getattr__(f'block{str(i).zfill(2)}')(x) for i in range(len(self.rates))]
        out = torch.cat(out, 1)
        out = self.fuse(out)
        mask = my_layer_norm(self.gate(x))
        mask = torch.sigmoid(mask)
        return x * (1 - mask) + out * mask
        # return x * mask + out * (1 - mask)
        # return out 


def my_layer_norm(feat):
    mean = feat.mean((2, 3), keepdim=True)
    std = feat.std((2, 3), keepdim=True) + 1e-9
    feat = 2 * (feat - mean) / std - 1
    feat = 5 * feat
    return feat

class BGSNet_AOTGAN(nn.Module):
    def __init__(self):
        super(BGSNet_AOTGAN, self).__init__()
        self.generator = InpaintGenerator()

    def forward(self, input, mask):
        # input = (input + 1)/2
        inv_mask = 1. - mask

        roi = input * mask

        # x = torch.cat((input,mask[:,0,:,:]),1)
        # x = mask[:,0:1,:,:]
        # print(x.size())

        fake_img = self.generator(input,inv_mask[:,0:1,:,:]) 
        # print(fake_img.size())
        # print('roi',roi.size())
        output = roi + fake_img * inv_mask
        # utils.save_image(fake_img[0:4,:,:,:], f"masked_img.png", nrow=1, normalize=False)
        return output    
    
class BGSNet_LGNet(nn.Module):
    def __init__(self):
        super(BGSNet_LGNet, self).__init__()
        self.generator = Generator(use_cuda=True,device_ids=0) #Generator from LGNet

    def forward(self, input, mask):
        # input = (input + 1)/2
        inv_mask = 1. - mask

        roi = input * mask

        # x = torch.cat((input,mask[:,0,:,:]),1)
        # x = mask[:,0:1,:,:]
        # print(x.size())

        fake_img = self.generator(input,inv_mask[:,0:1,:,:]) 
        # print(fake_img.size())
        # print('roi',roi.size())
        output = roi + fake_img * inv_mask
        # utils.save_image(fake_img[0:4,:,:,:], f"masked_img.png", nrow=1, normalize=False)
        return output          