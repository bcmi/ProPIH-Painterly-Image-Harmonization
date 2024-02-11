import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import functools
from torch.nn import init
from torch.optim import lr_scheduler

import cv2
import os
import time

## modules for painterly image harmonization

class TVLoss(nn.Module):
    def __init__(self, strength):
        super(TVLoss, self).__init__()
        self.strength = strength
        self.x_diff = torch.Tensor()
        self.y_diff = torch.Tensor()

    def forward(self, input):
        self.x_diff = input[:, :, 1:, :] - input[:, :, :-1, :]
        self.y_diff = input[:, :, :, 1:] - input[:, :, :, :-1]
        self.loss = self.strength * (torch.sum(torch.abs(self.x_diff)) + torch.sum(torch.abs(self.y_diff)))
        # return input
        return self.loss

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.contiguous().view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.contiguous().view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adain_fg(comp_feat, style_feat, mask):
    #assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = comp_feat.size()
    #style_mean, style_std = calc_mean_std(style_feat)  # the style features
    downsample_mask_style = torch.ones(mask.size()).to(style_feat.device)
    # downsample_mask_style = 1 - mask
    style_mean, style_std = get_foreground_mean_std(style_feat, downsample_mask_style)  # the style features
    fore_mean, fore_std = get_foreground_mean_std(comp_feat, mask)  # the foreground features

    normalized_feat = (comp_feat - fore_mean.expand(size)) / fore_std.expand(size)
    return (normalized_feat * style_std.expand(size) + style_mean.expand(size)) * mask + (comp_feat * (1 - mask))

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(), # relu1-1
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)), 
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(), # relu2-1
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    # nn.Conv2d(64, 1, (3, 3),padding=0,stride=1), ##matting layer
    nn.Conv2d(65, 1, (1, 1),padding=0,stride=1), ##matting layer
    nn.ReflectionPad2d((1, 1, 1, 1)), # 24
    # nn.ReflectionPad2d((1, 1, 1, 1)), ##matting layer
    nn.Conv2d(64, 3, (3, 3)),
)


vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)), 
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)


encoder = nn.Sequential(
    nn.Conv2d(4, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
)



class Identity(nn.Module):
    def forward(self, x):
        return x

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    elif norm_type.startswith('rain'):
        norm_layer = RAIN
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_G(input_nc, output_nc, ngf, netG,opt,  norm='batch', use_dropout=False,
             init_type='normal', init_gain=0.02, gpu_ids=[],encoder=None,decoder=None):
    """load a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: rainnet
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    """
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'rainnet':
        net = RainNet(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, use_attention=True)
    elif netG == 'adain':
        print('loading vgg from {}'.format(opt.vgg))
        encoder.load_state_dict(torch.load(opt.vgg))
        encoder = nn.Sequential(*list(encoder.children())[:31])
        net = AdainNet(encoder, decoder)
    elif netG == 'RegRainNet':
        print('loading vgg from {}'.format(opt.vgg))
        encoder.load_state_dict(torch.load(opt.vgg))
        encoder = nn.Sequential(*list(encoder.children())[:31])
        net = RegRainNet(encoder, decoder)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    """
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
            self.relu = nn.ReLU()
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean() # self.relu(1-prediction.mean())
            else:
                loss = prediction.mean() # self.relu(1+prediction.mean())
        return loss


def get_foreground_mean_std(features, mask, eps=1e-5):
    region = features * mask 
    sum = torch.sum(region, dim=[2, 3])     # (B, C)
    num = torch.sum(mask, dim=[2, 3])       # (B, C)
    mu = sum / (num + eps)
    mean = mu[:, :, None, None]
    var = torch.sum((region + (1 - mask)*mean - mean) ** 2, dim=[2, 3]) / (num + eps)
    var = var[:, :, None, None]
    std = torch.sqrt(var+eps)
    return mean, std


class UpsampleDecoder(nn.Module):
    def __init__(self, k, dim):
        super().__init__()
        self.conv1 = nn.Sequential(nn.ReflectionPad2d((1, 1, 1, 1)),
                                   nn.Conv2d(dim, dim//2, (3, 3)),
                                   nn.ReLU())
        upconvs = [] 
        for _ in range(k):
            dim = dim//2 
            upconv = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                   nn.ReflectionPad2d((1, 1, 1, 1)),
                                   nn.Conv2d(dim, dim//2, (3, 3)),
                                   nn.ReLU())
            upconvs.append(upconv)
            
        self.upconvs = nn.Sequential(*upconvs) 
            
    def forward(self, x):
        output = self.conv1(x)
        output = self.upconvs(output)
        return output
        
class FuseBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fuse_block1 = nn.Sequential(nn.ReflectionPad2d((1, 1, 1, 1)),
                                        nn.Conv2d(64, 32, (3, 3)),
                                        nn.ReLU(),  
                                        nn.ReflectionPad2d((1, 1, 1, 1)),
                                        nn.Conv2d(32, 32, (3, 3)),
                                        nn.ReLU())    
        
        self.fuse_block2 = nn.Sequential(nn.ReflectionPad2d((1, 1, 1, 1)),
                                        nn.Conv2d(32, 32, (3, 3)),
                                        nn.ReLU(),  
                                        nn.ReflectionPad2d((1, 1, 1, 1)),
                                        nn.Conv2d(32, 32, (3, 3)),
                                        nn.ReLU())    
                    
    def forward(self, x1, x2):
        output = torch.cat((x1,x2), dim=1)
        output = self.fuse_block1(output)
        output = self.fuse_block2(output)
        return output
    
class VGG19HRNet(nn.Module):
    def __init__(self,vgg):
        super().__init__()
        # load the pretrained VGG encoder
        vgg_layers = list(vgg.children())
        self.enc_1 = nn.Sequential(*vgg_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*vgg_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*vgg_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*vgg_layers[18:31])  # relu3_1 -> relu4_1

        # fix the VGG encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False
        
        self.updec_1 = UpsampleDecoder(0,64)
        self.updec_2 = UpsampleDecoder(1,128)
        self.updec_3 = UpsampleDecoder(2,256)
        self.updec_4 = UpsampleDecoder(3,512)

        self.final_conv_1 = nn.Conv2d(32, 3, (1, 1))
        self.fuseblk_2 =  FuseBlock()
        self.final_conv_2 = nn.Conv2d(32, 3, (1, 1))
        self.fuseblk_3 =  FuseBlock()
        self.final_conv_3 = nn.Conv2d(32, 3, (1, 1))
        self.fuseblk_4 =  FuseBlock()
        self.final_conv_4 = nn.Conv2d(32, 3, (1, 1))


        self.mse_loss = nn.MSELoss()
    
    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]
    
    def calc_content_loss(self, gen, comb):
        loss = self.mse_loss(gen, comb) 
        return loss
    

    def downsample(self, image_tensor, width, height):
        image_upsample_tensor = torch.nn.functional.interpolate(image_tensor, size=[width, height])
        image_upsample_tensor = image_upsample_tensor.clamp(0, 1)
        return image_upsample_tensor


    def calc_style_loss_mulitple_fg_layer(self, combs, styles, mask, layer):
        loss = torch.zeros(1).to(mask.device)
        for i in range(0, layer):
            width = height = combs[i].size(-1)
            downsample_mask = self.downsample(mask, width, height)
            downsample_mask_style = torch.ones(downsample_mask.size()).to(mask.device)  #1-downsample_mask 

            mu_cs,sigma_cs = get_foreground_mean_std(combs[i], downsample_mask)
            mu_target,sigma_target = get_foreground_mean_std(styles[i], downsample_mask_style)
            loss_i = self.mse_loss(mu_cs, mu_target) + self.mse_loss(sigma_cs, sigma_target)
            loss += loss_i
        return loss
    
    def cal_loss(self,final_output,comb_feats,style_feats,mask, layer):
        fine_feats = self.encode_with_intermediate(final_output)
        # calculate content loss
        loss_c = self.calc_content_loss(fine_feats[-1], comb_feats[-1])
        # calculate style loss
        loss_s = self.calc_style_loss_mulitple_fg_layer(fine_feats, style_feats, mask,layer)
        return loss_c,loss_s
    
        
    def forward(self, comp, style, mask):
        style_feats = self.encode_with_intermediate(style)
        comb_feats = self.encode_with_intermediate(comp)
        
        output1 = self.enc_1(comp)
        output2 = self.enc_2(output1)
        output3 = self.enc_3(output2)
        output4 = self.enc_4(output3)

        width = height = output1.size(-1)
        downsample_mask1 = self.downsample(mask, width, height)
        t1 = adain_fg(output1, style_feats[0], downsample_mask1)

        width = height = output2.size(-1)
        downsample_mask2 = self.downsample(mask, width, height)
        t2 = adain_fg(output2, style_feats[1], downsample_mask2)

        width = height = output3.size(-1)
        downsample_mask3 = self.downsample(mask, width, height)
        t3 = adain_fg(output3, style_feats[2], downsample_mask3)
        
        width = height = output4.size(-1)
        downsample_mask4 = self.downsample(mask, width, height)
        t4 = adain_fg(output4, style_feats[3], downsample_mask4)

        output1 = self.updec_1(t1)
        output2 = self.updec_2(t2)
        output3 = self.updec_3(t3)
        output4 = self.updec_4(t4)
        

        coarse_output1 = self.final_conv_1(output1)
        # final_output1 = coarse_output1
        blend_mask1 = mask
        final_output1 = coarse_output1 * blend_mask1 + style * (1 - blend_mask1)

        output2 = self.fuseblk_2(output1.detach(),output2)
        coarse_output2 = self.final_conv_2(output2)
        # final_output2 = coarse_output2
        blend_mask2 = mask
        final_output2 = coarse_output2 * blend_mask2 + style * (1 - blend_mask2)

        output3 = self.fuseblk_3(output2.detach(),output3)
        coarse_output3 = self.final_conv_3(output3)
        # final_output3 = coarse_output3
        blend_mask3 = mask
        final_output3 = coarse_output3 * blend_mask3 + style * (1 - blend_mask3)

        output4 = self.fuseblk_4(output3.detach(),output4)
        coarse_output4 = self.final_conv_4(output4)
        # final_output4 = coarse_output4
        blend_mask4 = mask
        final_output4 = coarse_output4 * blend_mask4 + style * (1 - blend_mask4)


        loss_c_1,loss_s_1 = self.cal_loss(coarse_output1,comb_feats,style_feats,mask, 1)
        loss_c_2,loss_s_2 = self.cal_loss(coarse_output2,comb_feats,style_feats,mask, 2)
        loss_c_3,loss_s_3 = self.cal_loss(coarse_output3,comb_feats,style_feats,mask, 3)
        loss_c_4,loss_s_4 = self.cal_loss(coarse_output4,comb_feats,style_feats,mask, 4)

        loss_c = loss_c_1 + loss_c_2 + loss_c_3 + loss_c_4
        loss_s = loss_s_1 + loss_s_2 + loss_s_3 + loss_s_4
            

        loss_c_1,loss_s_1 = self.cal_loss(final_output1,comb_feats,style_feats,mask, 1)
        loss_c_2,loss_s_2 = self.cal_loss(final_output2,comb_feats,style_feats,mask, 2)
        loss_c_3,loss_s_3 = self.cal_loss(final_output3,comb_feats,style_feats,mask, 3)
        loss_c_4,loss_s_4 = self.cal_loss(final_output4,comb_feats,style_feats,mask, 4)

        loss_c += loss_c_1 + loss_c_2 + loss_c_3 + loss_c_4
        loss_s += loss_s_1 + loss_s_2 + loss_s_3 + loss_s_4

        
        # print(output1.shape, output2.shape, output3.shape, output4.shape)
        return final_output1, final_output2, final_output3, final_output4, \
               coarse_output1, coarse_output2, coarse_output3, coarse_output4, \
               blend_mask1*2-1, blend_mask2*2-1, blend_mask3*2-1, blend_mask4*2-1, \
               loss_c, loss_s


"""
discriminator
"""

class ConvBlock_D(nn.Module):
    def __init__(
        self,
        in_channels, out_channels,
        kernel_size=4, stride=2, padding=1,
        norm_layer=nn.BatchNorm2d, activation=nn.ELU,
        bias=True,
    ):
        super(ConvBlock_D, self).__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            norm_layer(out_channels) if norm_layer is not None else nn.Identity(),
            nn.LeakyReLU(0.2, True),
        )
    def forward(self, x):
        return self.block(x)


class ConvEncoder_D(nn.Module):
    def __init__(
        self,
        depth, ch, patch_number,
        norm_layer, batchnorm_from, max_channels
    ):
        super(ConvEncoder_D, self).__init__()
        self.depth = depth
        self.patch_number = patch_number

        in_channels = 3
        out_channels = ch

        self.block0 = ConvBlock_D(in_channels, out_channels, norm_layer=norm_layer if batchnorm_from == 0 else None)
        self.block1 = ConvBlock_D(out_channels, out_channels, norm_layer=norm_layer if 0 <= batchnorm_from <= 1 else None)
        self.blocks_enc = nn.ModuleDict()
        for block_i in range(2, depth-2):
            if block_i % 2:
                in_channels = out_channels
            else:
                in_channels, out_channels = out_channels, min(2 * out_channels, max_channels)
            self.blocks_enc[f'block{block_i}'] = ConvBlock_D(
                in_channels, out_channels,
                norm_layer=norm_layer if 0 <= batchnorm_from <= block_i else None,
                padding=1 #int(block_i < depth - 1)
            )

        self.blocks_connected = nn.ModuleDict()
    
        for block_i in range(depth - 2, depth):
            if block_i % 2:
                in_channels = out_channels
            else:
                in_channels, out_channels = out_channels, min(2 * out_channels, max_channels)
            self.blocks_connected[f'block{block_i}'] = ConvBlock_D(
                in_channels, out_channels,
                norm_layer=norm_layer if 0 <= batchnorm_from <= block_i else None,
                kernel_size=3, stride=1, padding=int(block_i < depth - 1)
            )
        self.inner_channels = out_channels
        
    def forward(self, x):
        x = self.block0(x)
        x = self.block1(x)

        for block_i in range(2, self.depth - 2):
            block = self.blocks_enc[f'block{block_i}']
            x = block(x)

        output = x

        for block_i in range(self.depth - 2, self.depth):
            block = self.blocks_connected[f'block{block_i}']
            output = block(output)

        return output


class DeconvDecoder_D(nn.Module):
    def __init__(self, depth, encoder_innner_channels, norm_layer):
        super(DeconvDecoder_D, self).__init__()
        self.deconv_blocks = nn.ModuleList()

        in_channels = encoder_innner_channels
        self.deconv_block0 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            #nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1),
            norm_layer(in_channels // 2) if norm_layer is not None else nn.Identity(),
            nn.ReLU(True),
        )
        self.deconv_block1 = nn.Sequential(
            #nn.UpsamplingNearest2d(scale_factor=2),
            #nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, stride=1, padding=1),
            norm_layer(in_channels // 4) if norm_layer is not None else nn.Identity(),
            nn.ReLU(True),
        )

        self.to_binary = nn.Conv2d(in_channels // 4, 1, kernel_size=1)

    def forward(self, encoder_outputs, image):

        output = self.deconv_block0(encoder_outputs)
        output = self.deconv_block1(output)

        output = self.to_binary(output)

        return output


class ConvDiscriminator(nn.Module):
    def __init__(
        self,
        depth, patch_number,
        norm_layer=nn.BatchNorm2d, batchnorm_from=0,
        ch=64, max_channels=512
    ):
        super(ConvDiscriminator, self).__init__()
        self.depth = depth
        self.patch_number = patch_number
        self.encoder = ConvEncoder_D(depth, ch, patch_number, norm_layer, batchnorm_from, max_channels)
        self.decoder = DeconvDecoder_D(2, self.encoder.inner_channels, norm_layer)

    def forward(self, image):
        intermediates = self.encoder(image)
        output = self.decoder(intermediates, image)
        return output