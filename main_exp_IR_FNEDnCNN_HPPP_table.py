import os
import numpy as np
import hdf5storage
import argparse
import cv2
import sys
import torch
import scipy.io as scio
import torch.nn as nn
from PnP_restoration.utils.utils_restoration import matlab_style_gauss2D, imread_uint
from argparse import ArgumentParser
from scipy import ndimage
from PnP_restoration.utils.utils_restoration import array2tensor
from PnP_restoration.utils import utils_sr
######RGB2YCbCr, epsilon=0.82
#python main_deblurring_SR_PLM_table.py --task Super-resolution --ycbcr True --algo PLM
# python main_deblurring_SR_PLM_table.py --task Gaussian-deblurring --ycbcr True --algo RED
# python main_deblurring_SR_PLM_table.py --task Gaussian-deblurring --ycbcr True --algo REDPRO
# python main_deblurring_SR_PLM_table.py --task Gaussian-deblurring --ycbcr True --algo PnP_FBS
####灰度图像,delta-0.2,高斯模糊 sigma=5
# python main_deblurring_SR_comparison_PLM_CRED_table.py --task Gaussian-deblurring --ycbcr False --noise_level 5 --algo CRED --dataset set12
# python main_deblurring_SR_comparison_PLM_CRED_table.py --task Gaussian-deblurring --ycbcr False --noise_level 5 --algo CRED --dataset BSD8
# python main_deblurring_SR_comparison_PLM_CRED_table.py --task Gaussian-deblurring --ycbcr False --noise_level 5 --algo PLM --dataset set12
# python main_deblurring_SR_comparison_PLM_CRED_table.py --task Gaussian-deblurring --ycbcr False --noise_level 5 --algo PLM --dataset BSD8
###################HPPP图像复原实验#############################
# python main_exp_IR_FNEDnCNN_HPPP_table.py --task Gaussian-deblurring --ycbcr False --noise_level 2.55 --algo RED
# python main_exp_IR_FNEDnCNN_HPPP_table.py --task Gaussian-deblurring --ycbcr False --noise_level 2.55 --algo REDPRO
# python main_exp_IR_FNEDnCNN_HPPP_table.py --task Gaussian-deblurring --ycbcr False --noise_level 2.55 --algo GraRED_HP3
# python main_exp_IR_FNEDnCNN_HPPP_table.py --task Gaussian-deblurring --ycbcr False --noise_level 2.55 --algo GraRED_GS_HP3

# python main_deblurring_SR_comparison_PLM_CRED_table.py --task Uniform-deblurring --ycbcr False --noise_level 2.55 --algo CRED --dataset set12
# python main_deblurring_SR_comparison_PLM_CRED_table.py --task Uniform-deblurring --ycbcr False --noise_level 2.55 --algo PLM --dataset set12
# python main_deblurring_SR_comparison_PLM_CRED_table.py --task Uniform-deblurring --ycbcr False --noise_level 2.55 --algo CRED --dataset BSD8
# python main_deblurring_SR_comparison_PLM_CRED_table.py --task Uniform-deblurring --ycbcr False --noise_level 2.55 --algo PLM --dataset BSD8
###超分
# python main_deblurring_SR_comparison_PLM_CRED_table.py --task Super-resolution --ycbcr False --noise_level 2.55 --algo PLM --dataset set12
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='Gaussian-deblurring', help='image deblurring or super-resolution')###Gaussian-deblurring; Uniform-deblurring; Super-resolution
parser.add_argument('--scale', type=int, default=1, help='image scale')
parser.add_argument('--ycbcr', type=str, choices=['True', 'False'], default='False', help='Enable YCbCr mode')
parser.add_argument('--algo', type=str, default='CRED', help='algorithms')
parser.add_argument('--dataset', type=str, default='set12', help='algorithms')
parser.add_argument('--noise_level', type=float, default=2.55, help='noise level of image')
args = parser.parse_args()

def initialize_prox(img, degradation_mode, degradation, sf, device):
    if degradation_mode == 'deblurring':
        k = degradation
        k_tensor = array2tensor(np.expand_dims(k, 2)).double().to(device)
        FB, FBC, F2B, FBFy = utils_sr.pre_calculate_prox2(img, k_tensor, sf)
        return FB, FBC, F2B, FBFy, k_tensor
    elif degradation_mode == 'SR':
        k = degradation
        k_tensor = array2tensor(np.expand_dims(k, 2)).double().to(device)
        FB, FBC, F2B, FBFy = utils_sr.pre_calculate_prox2(img,k_tensor, sf)
        return FB, FBC, F2B, FBFy, k_tensor
    elif degradation_mode == 'inpainting':
        M = array2tensor(degradation).double().to(device)
        My = M*img
        return My
    else:
        print('degradation mode not treated')

def calulate_data_term(k_tensor,degradation_mode, sf,y,img):
        '''
        Calculation of the data term value f(y)
        :param y: Point where to evaluate F
        :param img: Degraded image
        :return: f(y)
        '''
#         k_tensor = array2tensor(np.expand_dims(k, 2)).double().to(device)
        if degradation_mode == 'deblurring':
            deg_y = utils_sr.imfilter(y.double(), k_tensor[0].double().flip(1).flip(2).expand(3, -1, -1, -1))
            f = 0.5 * torch.norm(img - deg_y, p=2) ** 2
        elif degradation_mode == 'SR':
            deg_y = utils_sr.imfilter(y.double(), k_tensor[0].double().flip(1).flip(2).expand(3, -1, -1, -1))
            deg_y = deg_y[..., 0::sf, 0::sf]
            f = 0.5 * torch.norm(img - deg_y, p=2) ** 2
#         elif degradation_mode == 'inpainting':
#             deg_y = M * y.double()
#             f = 0.5 * torch.norm(img - deg_y, p=2) ** 2
        else:
            print('degradation not implemented')
        return f
def calulate_gray_data_term(k_tensor,degradation_mode, sf,y,img):###适合灰度图像
        '''
        Calculation of the data term value f(y)
        :param y: Point where to evaluate F
        :param img: Degraded image
        :return: f(y)
        '''
#         k_tensor = array2tensor(np.expand_dims(k, 2)).double().to(device)
        if degradation_mode == 'deblurring':
            deg_y = utils_sr.imfilter(y.double(), k_tensor)
            f = 0.5 * torch.norm(img - deg_y, p=2) ** 2
        elif degradation_mode == 'SR':
            deg_y = utils_sr.imfilter(y.double(), k_tensor)
            deg_y = deg_y[..., 0::sf, 0::sf]
            f = 0.5 * torch.norm(img - deg_y, p=2) ** 2
#         elif degradation_mode == 'inpainting':
#             deg_y = M * y.double()
#             f = 0.5 * torch.norm(img - deg_y, p=2) ** 2
        else:
            print('degradation not implemented')
        return f
def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def ycbcr2rgb(img):
    '''same as matlab ycbcr2rgb
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                          [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]
    rlt = np.clip(rlt, 0, 255)
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def single2uint(img):
    return np.uint8(img*255.)

def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())

def tensor2uint2(img):
    img = img.data.squeeze().float().cpu().numpy()
    img = norm_proj(img)
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())

def tensor2float(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return img

def tensor2float2(img):
    img = img.data.squeeze().float().cpu().numpy()
    img = norm_proj(img)
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return img
    
def calculate_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 /np.sqrt(mse))

def calculate_grad(img, degradation_mode, FB, FBC,FBFy,sf=1):
    if degradation_mode == 'deblurring':
        grad = utils_sr.grad_solution2(img.double(), FB, FBC, FBFy, 1)
    if degradation_mode == 'SR' :
        grad = utils_sr.grad_solution2(img.double(), FB, FBC, FBFy, sf)
    return grad

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def norm_proj(x):
    x = (x-np.min(x))/(np.max(x)-np.min(x))
    return x

def imsave(img, img_path):
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)
    
def numpy_degradation(x, k, sf=3):
    ''' blur + downsampling
    Args:
        x: HxWxC image, [0, 1]/[0, 255]
        k: hxw, double, positive
        sf: down-scale factor
    Return:
        downsampled LR image
    '''
    x = ndimage.filters.convolve(x, np.expand_dims(k, axis=2), mode='wrap')
    st = 0
    return x[st::sf, st::sf, ...]
def Projection_On_L2_Ball(x, y, radius):
    # x: point to project
    # y: center of the ball
    # radius: radius of the ball
    if torch.norm(x-y, p=2) > radius:
        Px = y + (x-y)/torch.norm(x-y, p=2)*radius
    else:
        Px = x
    return Px

def Landweber_operator(x, y, FB, FBC, delta, lambda_k=1, sf=1):
    ###Lx = x+||T(Ax)-Ax||^2/(||A*(T(Ax)-Ax)||^2) A*(T(Ax)-Ax)
    ###alpha: relaxed parameter (0,1]
    ###Ax####
    FBFx = FB.mul(torch.fft.fftn(x, dim=(-2,-1)))
    AFx = utils_sr.downsample(torch.fft.ifftn(FBFx, dim=(-2,-1)),sf=sf)
    # AFx = torch.real(AFx)
    ####T(Ax)####
    TAx = Projection_On_L2_Ball(AFx, y, delta)
    TAx_norm = torch.norm(TAx-AFx, p=2)**2
    ####A*(T(Ax)-Ax)####
    r = TAx - AFx
    STr = utils_sr.upsample(r, sf=sf)
    FBCSTr = FBC.mul(torch.fft.fftn(STr, dim=(-2,-1)))
    AstarFBCSTr = torch.real(torch.fft.ifftn(FBCSTr, dim=(-2,-1)))
    AstarFBCSTr_norm = torch.norm(AstarFBCSTr, p=2)**2
    if torch.norm(AFx-y, p=2) > delta:
        Lx = x +lambda_k* TAx_norm/AstarFBCSTr_norm*AstarFBCSTr
    else:
        Lx = x
    return Lx

def initialize_cuda_gray_denoiser(device):
    '''
    Initialize the denoiser model with the given pretrained ckpt
    '''
    sys.path.append('./GS_denoising/')
    from lightning_GSDRUNet import GradMatch
    # from lightning_denoiser import GradMatch

    parser2 = ArgumentParser(prog='utils_restoration.py')
    parser2 = GradMatch.add_model_specific_args(parser2)
    parser2 = GradMatch.add_optim_specific_args(parser2)
    hparams = parser2.parse_known_args()[0]
    hparams.act_mode = 'E'
    hparams.grayscale = True
    denoiser_model = GradMatch(hparams)
    # print(self.hparams.pretrained_checkpoint)换成单通道的模型
    hparams.pretrained_checkpoint = './GS_denoising/ckpts/GSDRUNet_grayscale.ckpt'
    checkpoint = torch.load(hparams.pretrained_checkpoint, map_location=device)
    denoiser_model.load_state_dict(checkpoint['state_dict'],strict=False)
    denoiser_model.eval()
    for i, v in denoiser_model.named_parameters():
        v.requires_grad = False
    denoiser_model = denoiser_model.to(device)
    return denoiser_model
    
def load_model(model_type, sigma, device):
    path = "./Provable_PnP/Pretrained_models/" + model_type + "_noise" + str(sigma) + ".pth"
    if model_type == "DnCNN":
        from Provable_PnP.model.models import DnCNN
        net = DnCNN(channels=1, num_of_layers=17)
        model = nn.DataParallel(net).cuda()
    elif model_type == "SimpleCNN":
        from Provable_PnP.model.SimpleCNN_models import DnCNN
        model = DnCNN(1, num_of_layers = 4, lip = 0.0, no_bn = True).cuda()
    elif model_type == "RealSN_DnCNN":
        from Provable_PnP.model.realSN_models import DnCNN
        net = DnCNN(channels=1, num_of_layers=17)
        model = nn.DataParallel(net).cuda()
    elif model_type == "FNE_DnCNN":
        from Provable_PnP.model.full_realsn_models import FNE_DnCNN as DnCNN
        net = DnCNN(channels=1, num_of_layers=17).cuda()
        path = './Pretrained_models2/FNE_DnCNN_5/best_model.pth'
        # path ='./Pretrained_models2/FNE_DnCNN/epoch46_noise5_PSNR34.54_SSIM0.89.pth'
        model = net.cuda(device.index)
        # model = nn.DataParallel(net).cuda()
    elif model_type == "RealSN_SimpleCNN":
        from Provable_PnP.model.SimpleCNN_models import DnCNN
        model = DnCNN(1, num_of_layers = 4, lip = 1.0, no_bn = True).cuda()
    elif model_type == "MMO":
        from training.spc_models import simple_DnCNN
        model = simple_DnCNN(n_ch_in=1, n_ch_out=1, n_ch=64, nl_type='softplus', depth=20, bn=False).cuda()
        path ='./Pretrained_models2/SPCnet_sn_iters_10_0.8/best_model.pth'
        # path = './Pretrained_models2/SPCnet_MMO/epoch6_noise[0, 25]_PSNR34.01_SSIM0.89_128hc.pth'
    else:
        from Provable_PnP.model.realSN_models import DnCNN
        net = DnCNN(channels=1, num_of_layers=17)
        model = nn.DataParallel(net).cuda()

    model.load_state_dict(torch.load(path))
    model.eval()
    return model
##################################算法部分##################################
def SCFP_LFP_PnP(x0,y, k_tensor, FB, FBC, f, opt):
    K = opt['K']
    alpha = opt['alpha']
    beta = opt['beta']
    sigma = opt['sigma']
    sigma_norm = opt['sigma_norm']
    mu_0 = opt['mu_0']
    Sf = opt['sf']
    model_type = opt['model_type']
    model_name = opt['model_name']
    obj_fun = np.zeros(K)
    residual = torch.zeros(K)
    for i in range(K):
        x = x0
        Lx = Landweber_operator(x, y, FB, FBC, sigma_norm, alpha, sf=Sf)
        if model_type != 'FPNet':
            f_est = Lx
            mintmp = torch.min(f_est)
            maxtmp = torch.max(f_est)
            xtilde = (f_est - mintmp) / (maxtmp - mintmp)
            scale_range = 1.0 + sigma/255.0/2.0
            scale_shift = (1 - scale_range) / 2.0
            xtilde = xtilde * scale_range + scale_shift
            r = f(xtilde.float())
            z= xtilde - r
            z= (z - scale_shift) / scale_range
            z = z * (maxtmp - mintmp) + mintmp
            z = (1-beta)*f_est+beta*z
        else:
            if 'dncnn' in model_name:
                z = f(Lx.clamp_(0, 1).float())
                z = (1-beta)*Lx+beta*z 
        x0 = z
    return x0

def SAM_PRO_v2(x0,y, k_tensor, degradation_mode, FB, FBC, FBFy, f, opt):
    K = opt['K']
    alpha = opt['alpha']
    beta = opt['beta']
    sigma = opt['sigma']
    mu_0 = opt['mu_0']
    Sf = opt['sf']
    obj_fun = np.zeros(K)
    residual = torch.zeros(K)
    for i in range(K):
        x = x0
        mu_k = mu_0*(i+1)**(-1.0)
        mu = mu_k if mu_k<1 else 1
        f_est = x
        mintmp = torch.min(f_est)
        maxtmp = torch.max(f_est)
        xtilde = (f_est - mintmp) / (maxtmp - mintmp)
        scale_range = 1.0 + sigma/255.0/2.0
        scale_shift = (1 - scale_range) / 2.0
        xtilde = xtilde * scale_range + scale_shift
        r = f(xtilde.float())
        z= xtilde - r
        z= (z - scale_shift) / scale_range
        z = z * (maxtmp - mintmp) + mintmp
        z = (1-beta)*f_est+beta*z
        grad =  calculate_grad(z, degradation_mode, FB, FBC, FBFy, sf=Sf)
        v_est = z - mu*grad/sigma
        w = v_est
        residual[i] = torch.norm(w-x0, p=2)
        x0 = w
        obj_fun[i] = calulate_data_term(k_tensor,degradation_mode, Sf,x0,y).float().cpu().numpy()
    return x0, obj_fun, residual.float().cpu().numpy()

def RED(x0,y, k_tensor, degradation_mode, FB, FBC, FBFy, f, opt):
    K = opt['K']
    lambdaa = opt['lambda']
    sigma = opt['sigma']
    Sf = opt['sf']
    obj_fun = np.zeros(K)
    residual = torch.zeros(K)
    mu = opt['alpha']####### 0.1 for parrots(sacle =3, super-resolution); 1 for butterfly(sigma=8, uniform PSF)
    PSNR = []
    for i in range(K):
        x = x0
        psnr = calculate_psnr(tensor2uint(x), opt['gt'])
        PSNR.append(psnr)
        obj_fun[i] = 1/sigma**2*calulate_gray_data_term(k_tensor,degradation_mode, Sf,x0,y).float().cpu().numpy()
        grad1 =  calculate_grad(x, degradation_mode, FB, FBC, FBFy, sf=Sf)/sigma
        grad2 =  f(x.float())
        v_est = x - mu*(grad1/sigma**2+lambdaa*grad2)
        w = v_est
        residual[i] = torch.norm(w-x0, p=2)
        x0 = w
    return x0, obj_fun, residual.float().cpu().numpy(), PSNR

def REDPRO(x0,y, k_tensor, degradation_mode, FB, FBC, FBFy, f, opt):
    K = opt['K']
    alpha = opt['alpha']
    beta = opt['beta']
    sigma = opt['sigma']
    Sf = opt['sf']
    obj_fun = np.zeros(K)
    residual = torch.zeros(K)
    PSNR = []
    for i in range(K):
        x = x0
        psnr = calculate_psnr(tensor2uint(x), opt['gt'])
        PSNR.append(psnr)
        obj_fun[i] = 1/sigma**2*calulate_gray_data_term(k_tensor,degradation_mode, Sf,x0,y).float().cpu().numpy()
        mu =alpha*(i+1)**(-0.1)################# 2 for parrots(sacle =3, super-resolution); 4 for butterfly(sigma=8, uniform PSF)
        grad1 =  calculate_grad(x, degradation_mode, FB, FBC, FBFy, sf=Sf)
        f_est = x - mu*grad1/sigma**2
        v_est = beta*(f_est-f(f_est.float()))+(1-beta)*f_est
        w = v_est
        residual[i] = torch.norm(w-x0, p=2)
        x0 = w
    return x0, obj_fun, residual.float().cpu().numpy(), PSNR

def PnP_FBS(x0,y, k_tensor, degradation_mode, FB, FBC, FBFy, f, opt):
    K = opt['K']
    beta = opt['beta']
    sigma = opt['sigma']
    Sf = opt['sf']
    obj_fun = np.zeros(K)
    residual = torch.zeros(K)
    mu0 = 4
    PSNR = []
    for i in range(K):
        x = x0
        psnr = calculate_psnr(tensor2uint(x), opt['gt'])
        PSNR.append(psnr)
        obj_fun[i] = 1/sigma**2*calulate_gray_data_term(k_tensor,degradation_mode, Sf,x0,y).float().cpu().numpy()
        mu = mu0################# 2 for parrots(sacle =3, super-resolution); 4 for butterfly(sigma=8, uniform PSF)
        grad1 =  calculate_grad(x, degradation_mode, FB, FBC, FBFy, sf=Sf)
        f_est = x - mu*grad1/sigma**2
        v_est = beta*(f_est-f(f_est.float()))+(1-beta)*x
        w = v_est
        residual[i] = torch.norm(w-x0, p=2)
        x0 = w
        x_square = torch.norm(x0, p=2)**2
    return x0, obj_fun, residual.float().cpu().numpy(), PSNR

def SAM_PRO_v1(x0,y, k_tensor, degradation_mode, FB, FBC, FBFy, f, opt):
    K = opt['K']
    alpha = opt['alpha']
    beta = opt['beta']
    sigma = opt['sigma']
    mu_0 = opt['mu_0']
    Sf = opt['sf']
    obj_fun = np.zeros(K)
    residual = torch.zeros(K)
    for i in range(K):
        x = x0
        mu_k = mu_0*(i+1)**(-1.0)
        mu = mu_k if mu_k<1 else 1
        f_est = x
        mintmp = torch.min(f_est)
        maxtmp = torch.max(f_est)
        xtilde = (f_est - mintmp) / (maxtmp - mintmp)
        scale_range = 1.0 + sigma/255.0/2.0
        scale_shift = (1 - scale_range) / 2.0
        xtilde = xtilde * scale_range + scale_shift
        r = f(xtilde.float())
        z= xtilde - r
        z= (z - scale_shift) / scale_range
        z = z * (maxtmp - mintmp) + mintmp
        z = (1-beta)*f_est+beta*z
        grad =  calculate_grad(f_est, degradation_mode, FB, FBC, FBFy, sf=Sf)
        v_est = f_est - alpha*grad/sigma
        w = (1-mu)*z+mu*v_est
        residual[i] = torch.norm(w-x0, p=2)
        x0 = w
        obj_fun[i] = calulate_data_term(k_tensor,degradation_mode, Sf,x0,y).float().cpu().numpy()
    return x0, obj_fun, residual.float().cpu().numpy()


def GraRED_GS_HP3(x0, y0, xa, ya, y, k_tensor, degradation_mode, FB, FBC, F2B, FBFy, f, opt,device):
    K = opt['K']
    lambdaa = opt['lambda']
    tau = opt['tau']
    s = opt['s']
    Sf = opt['sf']
    q = opt['q']
    sigma_denoiser = opt['sigma_denoiser']
    anchor_type = opt['anchor_type']
    if anchor_type == 'initialization':
        xa = x0
        # ya = 0* y0
    elif anchor_type == 'denoising':
        xa = x0-f(x0.float())
        ya = xa
        # ya = 0* y0
    x = x0
    y = y0
    obj_fun = np.zeros(K)
    residual = torch.zeros(K)
    alpha = torch.tensor(1/lambdaa).float().repeat(1, 1, 1, 1).to(device)
    for i in range(K):
        mu_k = 1/(i%q+2) if anchor_type == 'restart' else 1/(i+2)
        # mu_k = min(2/(i+1), 1)
        # mu_k = 1 /(i+10)
        x_old = x
        y_old = y
        xa = x if i%q == 0 and anchor_type == 'restart' else xa
        ya = y if i%q == 0 and anchor_type == 'restart' else ya
        d = utils_sr.prox_solution2(x_old-tau*y_old, FB, FBC, F2B, FBFy, alpha, 1)
        x = mu_k*xa+(1-mu_k)*d
        f_est = y_old + s*(2*d-x_old)
        torch.set_grad_enabled(True)
        #Dg, N, g
        Dg, N, g = f.calculate_grad(f_est.float(), sigma_denoiser / 255.)
        torch.set_grad_enabled(False)
        Dg = Dg.detach()
        y = mu_k*ya+(1-mu_k)*Dg#####Dg是残差
        w = x
        residual[i] = torch.norm(w-x0, p=2)
        x0 = w
        x_square = torch.norm(x0, p=2)**2
        obj_fun[i] = calulate_data_term(k_tensor,degradation_mode, Sf,x0,y).float().cpu().numpy()/x_square.float().cpu().numpy()
    return x0, obj_fun, residual.float().cpu().numpy()


def PLM(x0,y, k_tensor, degradation_mode, FB, FBC, f, opt):
    K = opt['K']
    alpha = opt['alpha']
    beta = opt['beta']
    sigma = opt['sigma']
    sigma_norm = opt['sigma_norm']
    Sf = opt['sf']
    obj_fun = np.zeros(K)
    PSNR = []
    for i in range(K):
        x = x0
        psnr = calculate_psnr(tensor2uint(x), opt['gt'])
        PSNR.append(psnr)
        obj_fun[i] = 1/sigma**2*calulate_gray_data_term(k_tensor,degradation_mode, Sf,x0,y).float().cpu().numpy()
        Lx = Landweber_operator(x, y, FB, FBC, sigma_norm, lambda_k=alpha, sf=Sf)
        z = (1-beta)*Lx+beta*(Lx-f(Lx.float()))
        x0 = z
    return x0, obj_fun, PSNR

def CRED_ADMM(x0,y, k_tensor, degradation_mode, FB, F2B, FBC, f, opt):
    K = opt['K']
    beta = opt['beta']
    sigma = opt['sigma']
    sigma_norm = opt['sigma_norm']
    Sf = opt['sf']
    obj_fun = np.zeros(K)
    t0 = x0
    r0 = y - utils_sr.imfilter(x0.double(), k_tensor)
    Lambda_r = torch.zeros_like(x0)
    Lambda_t = torch.zeros_like(x0)
    # x = x0.clone()
    gamma = 1.01
    Beta_r =1
    Beta_t =1
    # alpha = torch.tensor(1/20).float().repeat(1, 1, 1, 1).cuda()
    for i in range(K):
        y_tilde = y+r0-Lambda_r/Beta_r
        x_bar = t0 - Lambda_t/Beta_t
        STy_tilde = utils_sr.upsample(y_tilde, sf=Sf)
        FBFy_tilde = FBC*torch.fft.fftn(STy_tilde, dim=(-2, -1))
        obj_fun = 1/sigma**2*calulate_gray_data_term(k_tensor,degradation_mode, Sf,x0,y).float().cpu().numpy()
        alpha =Beta_t/Beta_r
        x = utils_sr.prox_solution2(x_bar, FB, FBC, F2B, FBFy_tilde, alpha, Sf)
        deg_x = utils_sr.imfilter(x.double(), k_tensor)##Ax
        denoise_t = beta*(t0-f(t0.float()))+(1-beta)*t0
        t0 = 1/(1+Beta_t)*denoise_t+Beta_t/(1+Beta_t)*(x+Lambda_t/Beta_t)
        r0 = Projection_On_L2_Ball(deg_x-y+Lambda_r/Beta_r, 0, sigma_norm)
        Lambda_r +=  Beta_r*(-r0+deg_x-y)###艹艹艹,bug调了很久，要加负号
        Lambda_t +=  Beta_t*(-t0+x)
        Beta_r *= gamma
        Beta_t *= gamma
    return x,obj_fun
def GraRED_HP3(x0, y0, xa, ya, y, k_tensor, degradation_mode, FB, FBC, F2B, FBFy, f, opt):
    K = opt['K']
    lambdaa = opt['lambda']
    tau = opt['tau']
    s = opt['s']
    Sf = opt['sf']
    q = opt['q']
    anchor_type = opt['anchor_type']
    if anchor_type == 'initialization':
        xa = x0
        # ya = 0* y0
    elif anchor_type == 'denoising':
        xa = x0-f(x0.float())
        ya = xa
        # ya = 0* y0
    x = x0
    y = y0
    obj_fun = np.zeros(K)
    residual = torch.zeros(K)
    alpha = torch.tensor(1/lambdaa).float().repeat(1, 1, 1, 1).to(device)
    for i in range(K):
        mu_k = 1/(i%q+10) if anchor_type == 'restart' else 1/(i+10)
        # mu_k = min(2/(i+1), 1)
        # mu_k = 1 /(i+10)
        x_old = x
        y_old = y
        xa = x if i%q == 0 and anchor_type == 'restart' else xa
        ya = y if i%q == 0 and anchor_type == 'restart' else ya
        d =  utils_sr.prox_solution2(x_old-tau*y_old, FB, FBC, F2B, FBFy, alpha, 1)
        x = mu_k*xa+(1-mu_k)*d
        f_est = y_old + s*(2*d-x_old)
        y = mu_k*ya+(1-mu_k)*(f(f_est.float()))#####fest-z是残差
        w = x
        residual[i] = torch.norm(w-x0, p=2)
        x0 = w
        x_square = torch.norm(x0, p=2)**2
        obj_fun[i] = calulate_data_term(k_tensor,degradation_mode, Sf,x0,y).float().cpu().numpy()/x_square.float().cpu().numpy()
    return x0, obj_fun, residual.float().cpu().numpy()
if __name__ == '__main__':
    ycbcr_mode = args.ycbcr == 'True'
    kernel_path = os.path.join('PnP_restoration/kernels', 'Levin09.mat')
    kernels = hdf5storage.loadmat(kernel_path)['kernels']
    # test_imgs = ['bike','butterfly', 'flower', 'girl', 'hat']#'parrots'###bike, butterfly,flower, girl, hat,parrots; 'butterfly', 'flower', 'girl', 'hat'
    if not ycbcr_mode and 'set12' in args.dataset:
        test_path = './datasets/set12/'
        test_imgs = os.listdir(test_path)
        # print(test_imgs)
    elif not ycbcr_mode and 'BSD68' in args.dataset:
        test_path = './datasets/BSD68_gray/'
        test_imgs = os.listdir(test_path)
    elif not ycbcr_mode and 'BSD8' in args.dataset:
        test_path = './datasets/BSD8/'
        test_imgs = os.listdir(test_path)
    else:
        test_imgs = ['bike','butterfly', 'flower', 'girl', 'hat', 'parrots',  'boats','house', 'leaves', 'starfish', 'cameraman', 'peppers']
    img_type = 'gray' if not ycbcr_mode else 'RGB'
    # print('img_type: ', img_type)
    task = args.task
    noise_level =args.noise_level
    algorithm = args.algo#'SAM_PROv2'
    print('ycbcr_mode: ', ycbcr_mode)
    print('algorithm: ', algorithm)
    print('task: ', task)
    print('noise_level: ', noise_level)
    if task == 'Uniform-deblurring': # Uniform blur
        k = (1/81)*np.ones((9,9))
        Sf = 1
    elif task == 'Gaussian-deblurring':  # Gaussian blur
        k = matlab_style_gauss2D(shape=(25,25),sigma=1.6)
        Sf = 1
    elif task == 'Super-resolution':  # Gaussian blur
        k = matlab_style_gauss2D(shape=(7,7),sigma=1.6)
        Sf = args.scale
        noise_level = 5
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_type = 'FNE_DnCNN' #'RealSN_DnCNN'|'RealSN_SimpleCNN'
    f = load_model(model_type, 5, device)
    GS_model = initialize_cuda_gray_denoiser(device)
    total_PSNR = 0
    for name in test_imgs:
        if img_type == 'RGB':
            input_im_uint = imread_uint('test_images/'+name+'.tif',n_channels=3)
        #     input_im_uint = imread_uint('datasets/set12/05.png', n_channels=3)
        else:
            # input_im_uint = imread_uint('test_images/'+name+'.tif',n_channels=1)
            input_im_uint = imread_uint(test_path+name, n_channels=1) if not ycbcr_mode else imread_uint('test_images/'+name+'.tif',n_channels=1)
        #     input_im_uint = imread_uint('datasets/set12/05.png', n_channels=1)
        input_im = np.float32(input_im_uint / 255)
                    # Degrade image
        if Sf>1:#super-resolution
            blur_im = numpy_degradation(input_im, k, sf=Sf)
        else:# debluring
            blur_im = ndimage.filters.convolve(input_im, np.expand_dims(k, axis=2), mode='wrap')
        np.random.seed(seed=0)
        noise = np.random.normal(0, noise_level / 255., blur_im.shape)
        blur_im += noise
        prox_noise_norm = np.sqrt((noise_level / 255.)**2 * blur_im.shape[0] * blur_im.shape[1])
        print('noise norm: ', prox_noise_norm)
        init_im = blur_im
        x_gt = rgb2ycbcr(input_im_uint) if ycbcr_mode else input_im_uint[..., 0]
        if not ycbcr_mode:
            init_im1 = init_im#rgb2ycbcr(norm_proj(init_im))
            img_tensor = array2tensor(init_im1).to(device)
        else:
            init_im1 = rgb2ycbcr(norm_proj(init_im))
            init_im2 = np.expand_dims(init_im1, axis=2)
            img_tensor = array2tensor(init_im2).to(device)
        if img_type == 'RGB' and Sf==1:########ycbcr_mode
            degradation_mode = 'deblurring'
            x0 = img_tensor#初值
        elif Sf>1:########ycbcr_mode super-resolution
            degradation_mode = 'SR'
            x0 = cv2.resize(init_im1, (init_im1.shape[1] * Sf, init_im1.shape[0] * Sf),interpolation=cv2.INTER_CUBIC)
            x0 = np.expand_dims(x0, axis=2)
            x0 = utils_sr.shift_pixel(x0, Sf)
            x0 = array2tensor(x0).to(device)
        else:############gray
            degradation_mode = 'deblurring'
            x0 = img_tensor#初值
            img_tensor = array2tensor(init_im).to(device)
        FB, FBC, F2B, FBFy, k_tensor = initialize_prox(img_tensor, degradation_mode, k, Sf, device)
        ################################################algorithm setting####################################################################
        if noise_level<=5.0:
            sigma_f=5
        elif 15 >= noise_level>5:
            sigma_f=15
        elif 40 >= noise_level> 15:
            sigma_f=25
        else:
            print('error')
        ####################################################################################################################################
        with torch.no_grad():
            if algorithm == 'SAM_PROv1':###S(x) = x-s\nabla f(x)
                opt_r={'alpha':2, 'beta':0.1, 'sigma':noise_level, 'sigma_f':sigma_f, 'K':500, 'mu_0':500, 'sf': Sf}
                x1, objfun,r = SAM_PRO_v1(x0,img_tensor, k_tensor, degradation_mode, FB, FBC, FBFy, f, opt_r)
            elif algorithm == 'PnP_FBS':###S(x) = x-s\nabla f(x)
                opt_PnP_FBS={'alpha':2*noise_level, 'beta':0.1, 'sigma':noise_level, 'sigma_f':sigma_f, 'K':500, 'lambda':0.01,'input_sigma':noise_level**2, 'sf': Sf, 'gt':x_gt}
                x1, objfun, r,_ = PnP_FBS(x0,img_tensor, k_tensor, degradation_mode, FB, FBC, FBFy, f, opt_PnP_FBS)
            elif algorithm == 'SAM_PROv2':###S(x) = Tx-s\nabla f(Tx)
                if name == 'butterfly':
                    opt={'alpha':4, 'beta':0.1, 'sigma':noise_level, 'sigma_f':sigma_f, 'K':500, 'mu_0':1800, 'sf': Sf}
                else:
                    opt={'alpha':4, 'beta':0.1, 'sigma':noise_level, 'sigma_f':sigma_f, 'K':2000, 'mu_0':500, 'sf': Sf}
                x1, objfun,r = SAM_PRO_v2(x0,img_tensor, k_tensor, degradation_mode, FB, FBC, FBFy, f, opt)
            elif algorithm == 'RED':
                # opt_RED={'model_type': model_type,'alpha':2.4, 'beta':0.1, 'sigma':noise_level, 'sigma_f':sigma_f, 'K':2000, 'lambda':0.02,'input_sigma':noise_level**2, 'sf': Sf} #for butterfly 
                opt_RED={'model_type': model_type,'alpha':2*noise_level, 'beta':0.1, 'sigma':noise_level, 'sigma_f':sigma_f, 'K':500, 'lambda':0.01,'input_sigma':noise_level**2, 'sf': Sf, 'gt':x_gt} #for parrots
                x1, objfun, r,_ = RED(x0,img_tensor, k_tensor, degradation_mode, FB, FBC, FBFy, f, opt_RED)
            elif algorithm == 'REDPRO':
                opt_REDPRO={'model_type': model_type,'alpha':2*noise_level, 'beta':0.1, 'sigma':noise_level, 'sigma_f':sigma_f, 'K':500, 'lambda':0.01,'input_sigma':noise_level**2, 'sf': Sf, 'gt':x_gt}
                x1, objfun, r,_ = REDPRO(x0,img_tensor, k_tensor, degradation_mode, FB, FBC, FBFy, f, opt_REDPRO)
            elif algorithm == 'CRED':
                opt_CRED={'model_type': model_type, 'model_name': 'dncnn', 'n_channels':1, 'alpha':1, 'beta':1, 'sigma':noise_level, 'sigma_norm':0.98*prox_noise_norm, 'sigma_f':sigma_f, 'K':200, 'mu_0':500, 'sf': Sf}
                x1, objfun = CRED_ADMM(x0,img_tensor, k_tensor, degradation_mode, FB, F2B, FBC, f, opt_CRED)
            elif algorithm == 'PLM':
                # opt_SCFP={'model_type': model_type, 'model_name': 'dncnn', 'n_channels':1, 'alpha':1, 'beta':0.1, 'sigma':noise_level, 'sigma_norm':prox_noise_norm-0.2, 'sigma_f':sigma_f, 'K':2000, 'mu_0':500, 'sf': Sf}
                opt_SCFP={'model_type': model_type, 'model_name': 'dncnn', 'n_channels':1, 'alpha':1, 'beta':0.1, 'sigma':noise_level, 'sigma_norm':prox_noise_norm-0.2, 'sigma_f':sigma_f, 'K':500, 'mu_0':500, 'sf': Sf, 'gt':x_gt}
                x1, objfun,_ =PLM(x0,img_tensor, k_tensor, degradation_mode, FB, FBC, f, opt_SCFP)
            elif algorithm == 'GraRED_HP3':
                opt_RED_HP3={'model_type': model_type, 'anchor_type': 'initialization', 'q':500, 'lambda_k':0.2, 'beta':1, 'tau':3, 's':1/3,'sigma':noise_level, 'lambda':40,'sigma_f':sigma_f, 'K':500, 'input_sigma':noise_level**2, 'sf': Sf}
                y0 = x0
                # xa = single2tensor4(PT(blur_im)).to(device)
                # ya = 0*xa
                xa = x0
                ya = y0
                x1, objfun, r = GraRED_HP3(x0, y0, xa, ya, img_tensor, k_tensor, degradation_mode, FB, FBC, F2B, FBFy, f, opt_RED_HP3)
            elif algorithm == 'GraRED_GS_HP3':
                #GraRED_GS_HP3参数：tau1.8,s1/.8,sigma_denoiser5,lambd30
                ###Restart_GraRED_GS_HP3参数：除了pepper设置:tau2.4,s1/2.4,sigma_denoiser5,lambd30;其他设置:tau1.8,s1/1.8,sigma_denoiser5,lambd30
                opt_RED_HP3={'model_type': model_type, 'anchor_type': 'restart', 'q':40, 'lambda_k':0.2, 'beta':1, 'tau':2.4, 's':1/2.4,'sigma':noise_level, 'sigma_denoiser':5,'lambda':30,'sigma_f':sigma_f, 'K':500, 'input_sigma':noise_level**2, 'sf': Sf}
                y0 = x0
                # xa = single2tensor4(PT(blur_im)).to(device)
                # ya = 0*xa
                xa = x0
                ya = y0
                x1, objfun, r = GraRED_GS_HP3(x0, y0, xa, ya, img_tensor, k_tensor, degradation_mode, FB, FBC, F2B, FBFy, GS_model, opt_RED_HP3, device)
            else:
                print('algorithm not implemented ^_^')
        if not ycbcr_mode:
            x_est = tensor2uint(x1)
            PSNR = calculate_psnr(x_gt, x_est)
        else:
            ###############################MATLAB PSNR, Follow RED paper################################
            import matlab
            import matlab.engine
            eng = matlab.engine.start_matlab()
            eng.addpath(eng.genpath(eng.fullfile(os.getcwd(),'Metrics')))
            x_gt_luma =x_gt
            im_ycbr = rgb2ycbcr(norm_proj(init_im), only_y=False)
            im_ycbr2 = rgb2ycbcr(norm_proj(init_im), only_y=False)
            x_est_luma = tensor2uint(x1)
            PSNR = eng.ComputePSNR(matlab.uint8(x_gt_luma.tolist()), matlab.uint8(x_est_luma.tolist()))###Followed by RED paper
        total_PSNR += PSNR
        print('{} - PSNR: {:.4f} dB'.format(name, PSNR))
    print('Average PSNR: {:.4f} dB'.format(total_PSNR / len(test_imgs)))