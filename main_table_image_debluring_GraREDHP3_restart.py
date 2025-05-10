from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import torch
import copy
import torch.fft
from math import sqrt
from ChambollePock.image_operators import *
# from ChambollePock.utils import *
# from ChambollePock.tomo_operators import AstraToolbox
from ChambollePock.convo_operators import *
from PnP_restoration.utils import utils_sr
from PnP_restoration.utils.utils_restoration import psnr
from PnP_restoration.utils.utils_restoration import matlab_style_gauss2D, imread_uint
# from PnP_restoration.utils.utils_sr import psf2otf, cconj, cdiv, csum, cmul, p2o, cabs

# from PnP_restoration.utils.utils_restoration import rgb2y, psnr, array2tensor, tensor2array

# ----
VERBOSE = 1


# ----
def p2o2(psf, shape):
    '''
    Convert point-spread function to optical transfer function.
    otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.
    Args:
        psf: NxCxhxw
        shape: [H, W]
    Returns:
        otf: NxCxHxWx2
    '''
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[..., :psf.shape[2], :psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis + 2)
    otf = torch.fft.fftn(otf, dim=(-2, -1))
    # n_ops = torch.sum(torch.tensor(psf.shape).type_as(psf) * torch.log2(torch.tensor(psf.shape).type_as(psf)))
    # otf[..., 1][torch.abs(otf[..., 1]) < n_ops*2.22e-16] = torch.tensor(0).type_as(psf)
    return otf


def single2tensor4(img):  ###1xCxHXW
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().unsqueeze(0)


def splits(a, sf):
    '''split a into sfxsf distinct blocks
    Args:
        a: NxCxWxH
        sf: split factor
    Returns:
        b: NxCx(W/sf)x(H/sf)x(sf^2)
    '''
    b = torch.stack(torch.chunk(a, sf, dim=2), dim=4)
    b = torch.cat(torch.chunk(b, sf, dim=3), dim=4)
    return b


def upsample(x, sf=3):
    '''s-fold upsampler
    Upsampling the spatial size by filling the new entries with zeros
    x: tensor image, NxCxWxH
    '''
    st = 0
    z = torch.zeros((x.shape[0], x.shape[1], x.shape[2] * sf, x.shape[3] * sf)).type_as(x)
    z[..., st::sf, st::sf].copy_(x)
    return z


def downsample(x, sf=3):
    '''s-fold downsampler
    Keeping the upper-left pixel for each distinct sfxsf patch and discarding the others
    x: tensor image, NxCxWxH
    '''
    st = 0
    return x[..., st::sf, st::sf]


def data_solution(x, FB, FBC, F2B, FBFy, alpha, sf):  ###closed solution using FFT
    #     FR = FBFy + torch.rfft(alpha*x,2, onesided=False)
    FR = FBFy + torch.fft.fftn(alpha * x, dim=(-2, -1))
    x1 = FB.mul(FR)  # cmul(FB,FR)#FB.mul(FR)
    FBR = torch.mean(splits(x1, sf), dim=-1, keepdim=False)
    invW = torch.mean(splits(F2B, sf), dim=-1, keepdim=False)
    invWBR = FBR.div(invW + alpha)  # cdiv(FBR, csum(invW, alpha))#FBR.div(invW + alpha)
    FCBinvWBR = FBC * invWBR.repeat(1, 1, sf,
                                    sf)  # cmul(FBC, invWBR.repeat(1, 1, sf, sf,1))#FBC*invWBR.repeat(1, 1, sf, sf)
    FX = (FR - FCBinvWBR) / alpha  # (FR-FCBinvWBR)/alpha.unsqueeze(-1)#(FR-FCBinvWBR)/alpha
    #     Xest = torch.real(torch.ifft(FX, 2, onesided=False))
    Xest = torch.real(torch.fft.ifftn(FX, dim=(-2, -1)))

    return Xest


def pre_calculate(x, k, sf):
    '''
    Args:
        x: NxCxHxW, LR input
        k: NxCxhxw
        sf: integer

    Returns:
        FB, FBC, F2B, FBFy
        will be reused during iterations
    '''
    w, h = x.shape[-2:]
    FB = p2o2(k, (w * sf, h * sf))
    FBC = torch.conj(FB)
    F2B = torch.pow(torch.abs(FB), 2)
    STy = upsample(x, sf=sf)
    #     FBFy = FBC*torch.rfft(STy, 2, onesided=False)
    print(FBC.shape, torch.fft.fftn(STy, dim=(-2, -1)).shape)
    FBFy = FBC * torch.fft.fftn(STy, dim=(-2, -1))
    return FB, FBC, F2B, FBFy


def r2c(x):
    # convert real to complex
    return torch.stack([x, torch.zeros_like(x)], -1)


def cabs2(x):
    return x[..., 0] ** 2 + x[..., 1] ** 2


def my_imshow(img_list, shape=None, cmap=None, nocbar=False):
    if isinstance(img_list, np.ndarray):
        is_array = True
        plt.figure()
        plt.imshow(img_list, interpolation="nearest", cmap=cmap)
        if nocbar is False: plt.colorbar()
        plt.show()

    elif shape:
        num = np.prod(shape)
        # ~ if num > 1 and is_array:
        # ~ print('Warning (my_imshow): requestred to show %d images but only one image was provided' %(num))
        if num != len(img_list):
            raise Exception('ERROR (my_imshow): requestred to show %d images but %d images were actually provided' % (
            num, len(img_list)))

        plt.figure()
        for i in range(0, num):
            curr = str(shape + (i + 1,))
            print(curr)
            curr = curr[1:-1].replace(',', '').replace(' ', '')
            print(curr)
            if i == 0:
                ax0 = plt.subplot(int(curr))
            else:
                plt.subplot(int(curr), sharex=ax0, sharey=ax0)
            plt.imshow(img_list[i], interpolation="nearest", cmap=cmap)
            plt.axis('off')
            if nocbar is False: plt.colorbar()
        plt.show()


def power_method(P, PT, data, n_it=10):
    '''
    Calculates the norm of operator K = [grad, P],
    i.e the sqrt of the largest eigenvalue of K^T*K = -div(grad) + P^T*P :
        ||K|| = sqrt(lambda_max(K^T*K))

    P : forward projection
    PT : back projection
    data : acquired sinogram
    '''
    x = PT(data)
    for k in range(0, n_it):
        x = PT(P(x)) - div(gradient(x))
        s = sqrt(norm2sq(x))
        x /= s
    return sqrt(s)


def power_method2(P, PT, data, n_it=10):
    '''
    Calculates the norm of operator K = [grad, P],
    i.e the sqrt of the largest eigenvalue of K^T*K = -div(grad)  :
        ||K|| = sqrt(lambda_max(K^T*K))
    P : forward projection
    PT : back projection
    data : acquired sinogram
    '''
    x = PT(data)
    for k in range(0, n_it):
        x = - div(gradient(x))
        s = sqrt(norm2sq(x))
        x /= s
    return sqrt(s)


def power_method3(P, PT, data, n_it=10):
    '''
    data: Hxwx1
    Calculates the norm of operator K = grad,
    i.e the sqrt of the largest eigenvalue of K^T*K = -div(grad)  :
        ||K|| = sqrt(lambda_max(K^T*K))

    P : forward projection
    PT : back projection
    data : acquired sinogram
    '''
    x = PT(data)
    x = x[..., 0]
    for k in range(0, n_it):
        x = - div(gradient(x))
        s = sqrt(norm2sq(x))
        x /= s
    return sqrt(s)


def power_method4(P, PT, data, n_it=10):
    '''
    data: Hxwx1
    Calculates the norm of operator K = grad,
    i.e the sqrt of the largest eigenvalue of K^T*K = -div(grad)  :
        ||K|| = sqrt(lambda_max(K^T*K))

    P : forward projection
    PT : back projection
    data : acquired sinogram
    '''
    x = PT(data)
    x = x[..., 0]
    for k in range(0, n_it):
        x = - div(gradient(x))
        s = sqrt(norm2sq(x))
        x /= s
    return sqrt(s)


def calulate_data_term(k_tensor, degradation_mode, sf, y, img):
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


def CP_FFT_deblur(P, PT, ker, x_gt, img_L, x0, y0, Lambda, tau, L, beta, theta, n_it, device):
    '''
    img_L: HxWx1
    P:degradation
    modified from kaizhang
    '''
    theta = theta
    return_energy = True
    sigma = 1/ (L**2*tau)
    # tau = 
    lambd = Lambda
    x0, y0 = x0.to(device), y0.to(device)
    x = x0
    x_bar = x0
    y = y0
    img_L_tensor, k_tensor = single2tensor4(img_L), single2tensor4(np.expand_dims(kern, 2))
    img_L_tensor, k_tensor = img_L_tensor.to(device), k_tensor.to(device)
    FB, FBC, F2B, FBFy = pre_calculate(img_L_tensor, k_tensor, 1)
    alpha = torch.tensor(1 / (lambd * tau))
    alpha = alpha.float().repeat(1, 1, 1, 1).to(device)
    if return_energy:
        en = np.zeros(n_it)
        Psnr = np.zeros(n_it)
    for k in range(n_it):
        # Update dual variables
        #         Id = torch.stack([torch.eye(256), torch.zeros(256, 256)], -1)
        x_old = x
        Psnr[k] = psnr(x_gt[..., 0], x_old.clamp_(0, 1).squeeze(0).squeeze(0).cpu().numpy())
        y_old = y
        y = torch_proj_l2(y_old + sigma * torch_gradient2(x_bar), beta)
        x = data_solution(x_old + tau * torch_div(y, device), FB, FBC, F2B, FBFy, alpha, 1)
        x_bar = x + theta * (x - x_old)
        #         x_old = x
        # Calculate norms
        if return_energy:
            fidelity = calulate_data_term(k_tensor, 'deblurring', 1, x, img_L_tensor).float().cpu().numpy()
            #             fidelity = 0.5*norm2sq(P(np.expand_dims(x,2))-img_L)
            tv = norm1(gradient(x.squeeze(0).squeeze(0).cpu().numpy()))
            energy = 0.5 * lambd * fidelity + beta * tv
            en[k] = energy
            # Psnr[k] = psnr(x_gt[..., 0], x.clamp_(0, 1).squeeze(0).squeeze(0).cpu().numpy())
            # if (VERBOSE and k % 10 == 0):
            #     print("[%d] : energy %e \t fidelity %e \t TV %e" % (k, energy, fidelity, tv))
    if return_energy:
        return en, Psnr, x.squeeze(0).squeeze(0).cpu().numpy()
    else:
        return x.squeeze(0).squeeze(0).cpu().numpy()


def PPP_FFT_deblur(P, PT, ker, x_gt, img_L, x0, y0, Lambda, tau, L, beta, lambda_k, n_it, mode, device):
    '''
    img_L: HxWx1
    P:degradation
    modified from kaizhang
    '''
    # if mode == 'Constant':
    #     lambda_k = lambda_k
    return_energy = True
    sigma = 1/ (L**2*tau)
    # tau = 1.0 / L
    lambd = Lambda
    x0, y0 = x0.to(device), y0.to(device)
    x = x0
    y = y0
    img_L_tensor, k_tensor = single2tensor4(img_L), single2tensor4(np.expand_dims(kern, 2))
    img_L_tensor, k_tensor = img_L_tensor.to(device), k_tensor.to(device)
    FB, FBC, F2B, FBFy = pre_calculate(img_L_tensor, k_tensor, 1)
    alpha = torch.tensor(1 / (lambd * tau))
    alpha = alpha.float().repeat(1, 1, 1, 1).to(device)
    if return_energy:
        en = np.zeros(n_it)
        Psnr = np.zeros(n_it)
    for k in range(n_it):
        # Update dual variables
        #         Id = torch.stack([torch.eye(256), torch.zeros(256, 256)], -1)
        if mode != 'Constant':
            # lambda_k = lambda_k
            lambda_k = 1 + (1 - 1 / (k + 1)) ** 0.5
        # lambda_k = 1 + (1 - 1 / (k + 1)) ** 0.5
        x_old = x
        Psnr[k] = psnr(x_gt[..., 0], x.clamp_(0, 1).squeeze(0).squeeze(0).cpu().numpy())
        y_old = y
        d = data_solution(x_old + tau * torch_div(y_old, device), FB, FBC, F2B, FBFy, alpha, 1)
        x = lambda_k * d + (1 - lambda_k) * x_old
        y = lambda_k * torch_proj_l2(y_old + sigma * torch_gradient2(2 * d - x_old), beta) + (1 - lambda_k) * y_old
        # x = lambda_k * data_solution(x_old + tau * torch_div(y_old, device), FB, FBC, F2B, FBFy, alpha, 1) + (
        #             1 - lambda_k) * x_old
        # y = lambda_k * torch_proj_l2(y + sigma * torch_gradient2(2 * x - x_old), beta) + (1 - lambda_k) * y_old
        #         x_old = x
        # Calculate norms
        if return_energy:
            fidelity = calulate_data_term(k_tensor, 'deblurring', 1, x, img_L_tensor).float().cpu().numpy()
            #             fidelity = 0.5*norm2sq(P(np.expand_dims(x,2))-img_L)
            tv = norm1(gradient(x.squeeze(0).squeeze(0).cpu().numpy()))
            energy = 0.5 * lambd * fidelity + beta * tv
            en[k] = energy
            # Psnr[k] = psnr(x_gt[..., 0], x.clamp_(0, 1).squeeze(0).squeeze(0).cpu().numpy())
            # if (VERBOSE and k % 10 == 0):
            #     print("[%d] : energy %e \t fidelity %e \t TV %e" % (k, energy, fidelity, tv))
    if return_energy:
        return en, Psnr, x.squeeze(0).squeeze(0).cpu().numpy()
    else:
        return x.squeeze(0).squeeze(0).cpu().numpy()


def HPPP_FFT_deblur(P, PT, ker, x_gt, img_L, xa, ya, x0, y0, Lambda, beta, L, n_it, device):
    '''
    img_L: HxWx1
    P:degradation
    x0, y0: chu zhi
    xa, ya: anchor points
    modified from kaizhang
    '''
    return_energy = True
    sigma = 1.0 / L
    tau = 1.0 / L
    lambd = Lambda
    #     x = PT(img_L)
    #     x = x[..., 0]
    x0, y0 = x0.to(device), y0.to(device)
    xa, ya = xa.to(device), ya.to(device)
    x = x0
    y = y0
    #     x = PT(data)
    #     p = gradient(x)
    img_L_tensor, k_tensor = single2tensor4(img_L), single2tensor4(np.expand_dims(kern, 2))
    img_L_tensor, k_tensor = img_L_tensor.to(device), k_tensor.to(device)
    FB, FBC, F2B, FBFy = pre_calculate(img_L_tensor, k_tensor, 1)
    alpha = torch.tensor(1 / (lambd * tau))
    alpha = alpha.float().repeat(1, 1, 1, 1).to(device)
    if return_energy:
        en = np.zeros(n_it)
        Psnr = np.zeros(n_it)
    for k in range(n_it):
        mu = 1 / (k + 2)
        x_old = x
        Psnr[k] = psnr(x_gt[..., 0], x_old.clamp_(0, 1).squeeze(0).squeeze(0).cpu().numpy())
        y_old = y
        #         print((x_old+tau*torch_div(y_old[0]).unsqueeze(0).unsqueeze(0)).shape)
        #         print(torch_div(y_old, device))
        x = mu * xa + (1 - mu) * data_solution(x_old + tau * torch_div(y_old, device), FB, FBC, F2B, FBFy, alpha, 1)
        #         x = x.squeeze(0).squeeze(0).cpu().numpy()
        #         print(torch_gradient(x-mu*xa))
        y = mu * ya + (1 - mu) * torch_proj_l2(
            2 * sigma * torch_gradient2(x - mu * xa) / (1 - mu) - sigma * torch_gradient2(x_old) + y_old, beta)
        #         x_old = x
        # Calculate norms
        if return_energy:
            fidelity = calulate_data_term(k_tensor, 'deblurring', 1, x, img_L_tensor).float().cpu().numpy()
            #             fidelity = 0.5*norm2sq(P(np.expand_dims(x,2))-img_L)
            tv = norm1(gradient(x.squeeze(0).squeeze(0).cpu().numpy()))
            energy = 0.5 * lambd * fidelity + beta * tv
            en[k] = energy
            # Psnr[k] = psnr(x_gt[..., 0], x.clamp_(0, 1).squeeze(0).squeeze(0).cpu().numpy())
            # if (VERBOSE and k % 10 == 0):
            #     print("[%d] : energy %e \t fidelity %e \t TV %e" % (k, energy, fidelity, tv))
    if return_energy:
        return en, Psnr, x.squeeze(0).squeeze(0).cpu().numpy()
    else:
        return x.squeeze(0).squeeze(0).cpu().numpy()


def chambolle_pock_l1_tv(P, PT, x_gt, data, x0, y0, L, Lambda, n_it):
    return_energy = True
    sigma = 1.0 / L
    tau = 1.0 / L
    x = x0
    y = y0
    if return_energy:
        en = np.zeros(n_it)
        Psnr = np.zeros(n_it)
    for k in range(n_it):
        x_old = x
        y_old = y
        #         x = (x_old+tau*div(y_old)+Lambda*tau*data)/(1+Lambda*tau)
        #         y = proj_l2(y_old + sigma*gradient(2*x-x_old), 1)
        x = norm1_threshold(x_old + tau * div(y_old) - data, Lambda * tau) + data
        y = proj_l2(y_old + sigma * gradient(2 * x - x_old), 1)
        if return_energy:
            fidelity = Lambda * norm2sq(P(x) - data)
            tv = norm1(gradient(x))
            energy = fidelity + tv
            en[k] = energy
            Psnr[k] = psnr(x_gt / 255.0, x / 255.0)
            if (VERBOSE and k % 10 == 0):
                print("[%d] : energy %e \t fidelity %e \t TV %e" % (k, energy, fidelity, tv))
    if return_energy:
        return en, Psnr, x
    else:
        return x


def HPPP_l1_tv(P, PT, x_gt, data, xa, ya, x0, y0, L, Lambda, n_it):
    return_energy = True
    sigma = 1.0 / L
    tau = 1.0 / L
    x = x0
    y = y0
    if return_energy:
        en = np.zeros(n_it)
        Psnr = np.zeros(n_it)
    for k in range(n_it):
        #         print(k)
        mu = 1 / (k + 2)
        x_old = x
        y_old = y
        x = mu * xa + (1 - mu) * (norm1_threshold(x_old + tau * div(y_old) - data, Lambda * tau) + data)
        y = mu * ya + (1 - mu) * proj_l2(2 * sigma * gradient(x - mu * xa) / (1 - mu) - sigma * gradient(x_old) + y_old,
                                         1)
        if return_energy:
            fidelity = Lambda * norm2sq(P(x) - data)
            tv = norm1(gradient(x))
            energy = fidelity + tv
            en[k] = energy
            Psnr[k] = psnr(x_gt / 255.0, x / 255.0)
            if (VERBOSE and k % 10 == 0):
                print("[%d] : energy %e \t fidelity %e \t TV %e" % (k, energy, fidelity, tv))
    if return_energy:
        return en, Psnr, x
    else:
        return x


def HPPP_l2_tv(P, PT, x_gt, data, xa, ya, x0, y0, L, Lambda, n_it):
    return_energy = True
    # #############    xa, ya = anchor
    sigma = 1.0 / L
    tau = 1.0 / L
    x = x0
    y = y0
    if return_energy:
        en = np.zeros(n_it)
        Psnr = np.zeros(n_it)
    for k in range(n_it):
        mu = 1 / (k + 2)
        x_old = x
        y_old = y
        x = mu * xa + (1 - mu) * (x_old + tau * div(y_old) + Lambda * tau * data) / (1 + Lambda * tau)
        y = mu * ya + (1 - mu) * proj_l2(2 * sigma * gradient(x - mu * xa) / (1 - mu) - sigma * gradient(x_old) + y_old,
                                         1)
        if return_energy:
            fidelity = 0.5 * Lambda * norm2sq(P(x) - data)
            tv = norm1(gradient(x))
            energy = fidelity + tv
            en[k] = energy
            Psnr[k] = psnr(x_gt, x)
            if (VERBOSE and k % 10 == 0):
                print("[%d] : energy %e \t fidelity %e \t TV %e" % (k, energy, fidelity, tv))
    if return_energy:
        return en, Psnr, x
    else:
        return x


def chambolle_pock_l2_tv(P, PT, x_gt, data, x0, y0, L, Lambda, n_it):
    return_energy = True
    sigma = 1.0 / L
    tau = 1.0 / L
    x = x0
    y = y0
    if return_energy:
        en = np.zeros(n_it)
        Psnr = np.zeros(n_it)
    for k in range(n_it):
        #         print(k)
        x_old = x
        y_old = y
        x = (x_old + tau * div(y_old) + Lambda * tau * data) / (1 + Lambda * tau)
        y = proj_l2(y_old + sigma * gradient(2 * x - x_old), 1)
        if return_energy:
            fidelity = 0.5 * Lambda * norm2sq(P(x) - data)
            tv = norm1(gradient(x))
            energy = fidelity + tv
            en[k] = energy
            Psnr[k] = psnr(x_gt, x)
            if (VERBOSE and k % 10 == 0):
                print("[%d] : energy %e \t fidelity %e \t TV %e" % (k, energy, fidelity, tv))
    if return_energy:
        return en, Psnr, x
    else:
        return x


def chambolle_pock_matrix_complection(M, x_gt, data, x0, y0, L, Lambda, alpha, n_it):
    return_energy = True
    sigma = 1.0
    tau = 1.0 / (L ** 2)
    x = data
    y = 0
    theta = 1.0
    if return_energy:
        en = np.zeros(n_it)
        Psnr = np.zeros(n_it)
    for k in range(n_it):
        #         print(k)
        # Update dual variables
        x_old = x
        x = (2 * tau * M * data + x - tau * y) / (2 * tau * M + 1)
        # Update primal variables
        #         x_old
        y = nuclear_norm_conjugate_prox(y + sigma * (2 * x - x_old), alpha)
        if return_energy:
            fidelity = norm2sq(M * x - data)
            U, s, V = np.linalg.svd(x)
            reg = alpha * np.sum(s)
            energy = fidelity + reg
            en[k] = energy
            Psnr[k] = psnr(x_gt, x)
            if (VERBOSE and k % 10 == 0):
                print("[%d] : energy %e \t fidelity %e \t nuclear norm %e" % (k, energy, fidelity, reg))
    if return_energy:
        return en, Psnr, x
    else:
        return x

def HPPP_TV(P, PT, ker, x_gt, img_L, xa, ya, x0, y0, Lambda, beta, tau, L, n_it, device):
    '''
    img_L: HxWx1
    P:degradation
    x0, y0: chu zhi
    xa, ya: anchor points
    modified from kaizhang
    '''
    return_energy = True
    # tau = 0.2 / L
    sigma = 1/(tau*L**2 )
    lambd = Lambda
    #     x = PT(img_L)
    #     x = x[..., 0]
    x0, y0 = x0.to(device), y0.to(device)
    xa, ya = xa.to(device), ya.to(device)
    x = x0
    y = y0
    #     x = PT(data)
    #     p = gradient(x)
    img_L_tensor, k_tensor = single2tensor4(img_L), single2tensor4(np.expand_dims(ker, 2))
    img_L_tensor, k_tensor = img_L_tensor.to(device), k_tensor.to(device)
    FB, FBC, F2B, FBFy = pre_calculate(img_L_tensor, k_tensor, 1)
    alpha = torch.tensor(1 / (lambd * tau))
    alpha = alpha.float().repeat(1, 1, 1, 1).to(device)
    if return_energy:
        en = np.zeros(n_it)
        Psnr = np.zeros(n_it)
    for k in range(n_it):
        mu = 1 / (k + 2)
        x_old = x
        y_old = y
        Psnr[k] = psnr(x_gt[..., 0], x_old.clamp_(0, 1).squeeze(0).squeeze(0).cpu().numpy())
        d = data_solution(x_old + tau * torch_div2(y_old), FB, FBC, F2B, FBFy, alpha, 1)
        x = mu * xa + (1 - mu) * d
        #         x = x.squeeze(0).squeeze(0).cpu().numpy()
        #         print(torch_gradient(x-mu*xa))
        y = mu * ya + (1 - mu) * torch_proj_l2(y_old+
            sigma * torch_gradient2(2*d-x_old), beta)
        #         x_old = x
        # Calculate norms
        if return_energy:
            fidelity = calulate_data_term(k_tensor, 'deblurring', 1, x, img_L_tensor).float().cpu().numpy()
            #             fidelity = 0.5*norm2sq(P(np.expand_dims(x,2))-img_L)
            tv = norm1(gradient(x.squeeze(0).squeeze(0).cpu().numpy()))
            energy = 0.5 * lambd * fidelity + beta * tv
            en[k] = energy
            # Psnr[k] = psnr(x_gt[..., 0], x.clamp_(0, 1).squeeze(0).squeeze(0).cpu().numpy())
            # if (VERBOSE and k % 10 == 0):
            #     print("[%d] : energy %e \t fidelity %e \t TV %e" % (k, energy, fidelity, tv))
    if return_energy:
        return en, Psnr, x.squeeze(0).squeeze(0).cpu().numpy()
    else:
        return x.squeeze(0).squeeze(0).cpu().numpy()
def Restart_HPPP_TV(P, PT, ker, x_gt, img_L, xa, ya, x0, y0, Lambda, beta, tau, L, n_it, device):
    '''
    img_L: HxWx1
    P:degradation
    x0, y0: chu zhi
    xa, ya: anchor points
    modified from kaizhang
    '''
    return_energy = True
    # tau = 0.2 / L
    sigma = 1/(tau*L**2 )
    q  = 100
    lambd = Lambda
    #     x = PT(img_L)
    #     x = x[..., 0]
    x0, y0 = x0.to(device), y0.to(device)
    xa, ya = xa.to(device), ya.to(device)
    x = x0
    y = y0
    #     x = PT(data)
    #     p = gradient(x)
    img_L_tensor, k_tensor = single2tensor4(img_L), single2tensor4(np.expand_dims(ker, 2))
    img_L_tensor, k_tensor = img_L_tensor.to(device), k_tensor.to(device)
    FB, FBC, F2B, FBFy = pre_calculate(img_L_tensor, k_tensor, 1)
    alpha = torch.tensor(1 / (lambd * tau))
    alpha = alpha.float().repeat(1, 1, 1, 1).to(device)
    if return_energy:
        en = np.zeros(n_it)
        Psnr = np.zeros(n_it)
    for k in range(n_it):
        mu = 1 / (k + 1)
        x_old = x
        y_old = y
        Psnr[k] = psnr(x_gt[..., 0], x_old.clamp_(0, 1).squeeze(0).squeeze(0).cpu().numpy())
        xa = x if k%q == 0 else xa
        ya = y if k%q == 0 else ya
        d = data_solution(x_old + tau * torch_div2(y_old), FB, FBC, F2B, FBFy, alpha, 1)
        x = mu * xa + (1 - mu) * d
        y = mu * ya + (1 - mu) * torch_proj_l2(y_old+
            sigma * torch_gradient2(2*d-x_old), beta)
        #         x_old = x
        # Calculate norms
        if return_energy:
            fidelity = calulate_data_term(k_tensor, 'deblurring', 1, x, img_L_tensor).float().cpu().numpy()
            #             fidelity = 0.5*norm2sq(P(np.expand_dims(x,2))-img_L)
            tv = norm1(gradient(x.squeeze(0).squeeze(0).cpu().numpy()))
            energy = 0.5 * lambd * fidelity + beta * tv
            en[k] = energy
            # Psnr[k] = psnr(x_gt[..., 0], x.clamp_(0, 1).squeeze(0).squeeze(0).cpu().numpy())
            # if (VERBOSE and k % 10 == 0):
            #     print("[%d] : energy %e \t fidelity %e \t TV %e" % (k, energy, fidelity, tv))
    if return_energy:
        return en, Psnr, x.squeeze(0).squeeze(0).cpu().numpy()
    else:
        return x.squeeze(0).squeeze(0).cpu().numpy()

def HPPP_matrix_complection(M, x_gt, data, xa, ya, x0, y0, L, Lambda, alpha, n_it):
    return_energy = True
    # #############    xa, ya = anchor
    sigma = 1.0
    tau = 1.0 / (L ** 2)
    x = xa
    y = ya
    if return_energy:
        en = np.zeros(n_it)
        Psnr = np.zeros(n_it)
    for k in range(n_it):
        #         print(k)
        mu = 1 / (k + 2)
        x_old = x
        y_old = y
        x = mu * xa + (1 - mu) * (2 * tau * M * data + x_old - tau * y_old) / (2 * tau * M + 1)
        # Update primal variables
        #         x_old
        y = mu * ya + (1 - mu) * nuclear_norm_conjugate_prox(
            2 * sigma * (x - mu * xa) / (1 - mu) - sigma * (x_old) + y_old, alpha)
        if return_energy:
            fidelity = norm2sq(M * x - data)
            U, s, V = np.linalg.svd(x)
            reg = alpha * np.sum(s)
            energy = fidelity + reg
            en[k] = energy
            Psnr[k] = psnr(x_gt, x)
            if (VERBOSE and k % 10 == 0):
                print("[%d] : energy %e \t fidelity %e \t nuclear norm %e" % (k, energy, fidelity, reg))
    if return_energy:
        return en, Psnr, x
    else:
        return x

if __name__ == "__main__":
    path = 'random_matrix/'
    header = [' ', 'Cameraman', 'House', 'Pepper', 'Starfish', 'Butterfly', 'Craft', 'Parrots', 'Lena', 'Babana', 'Boat']
    n_it = 2000  # number of iterations
    noise_level_img = 0.02
    index = 1
    if index<9:
        name = '0'+str(index)+'.png'
        img_name = header[index]
    else:
        name = '10.png'
    img_path = './datasets/set12/'+name
    # img_path = './datasets/set12/02.png'

    # I = cv2.imread(img_path, 0)/255
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # I = I.to(device)
    I = imread_uint(img_path, n_channels=1) / 255.0
    x_gt = copy.deepcopy(I)
    ###############Gauss
    kern = matlab_style_gauss2D(shape=(25, 25), sigma=1.6)
    ##############Uniform
    # kern = (1/81)*np.ones((9,9))
    # img_L_tensor, k_tensor = util.single2tensor4(I), util.single2tensor4(np.expand_dims(kern, 2))
    # FB, FBC, F2B, FBFy = sr.pre_calculate(img_L_tensor, k_tensor, 1)
    from scipy.ndimage import filters

    # filters.convolve(np.expand_dims(I, axis=2), np.expand_dims(kern, axis=2), mode = 'wrap')
    K = ConvolutionOperator(kern)
    P = lambda x: K * x
    PT = lambda x: K.T() * x
    # beta = 0.0005 #######balance term of gradient
    beta = 0.0005
    theta = 1
    lambda_k = 1.6
    # Run
    # -----
    sino = P(I)
    print(sino.shape)
    L = power_method3(P, PT, sino, n_it=1000)  # * 1.5
    print("||K|| = %f" % L)
    # en, rec = chambolle_pock(P, PT, sino, beta, L, n_it)
    np.random.seed(seed=0)  # for reproducibility
    sino += np.random.normal(0, noise_level_img, sino.shape)  # add AWGN
    # for i in range(10):
    #     matrix = np.random.rand(256, 256, 1)
    #     print(matrix)
    #     file_name = path+'matrix_' + str(i) + '.npy'
    #     np.save(file_name, matrix)
    P = np.zeros((10, n_it))
    P2 = np.zeros((10, n_it))
    P3 = np.zeros((10, n_it))
    x0 = single2tensor4(sino)
    y0 = 0*torch_gradient2(x0)
    xa = single2tensor4(sino)
    # xa = single2tensor4(sino)
    ya = 0*torch_gradient2(x0)
    en0, Psnr0, rec0 = Restart_HPPP_TV(P, PT, kern, x_gt, sino, xa, ya, x0, y0, 2, beta, 0.1/L, L, n_it, device)
    # en, Psnr, rec = HPPP_FFT_deblur(P, PT, kern, x_gt, sino, xa, ya, x0, y0, 2, beta, L, n_it, device)
    en, Psnr, rec = HPPP_TV(P, PT, kern, x_gt, sino, xa, ya, x0, y0, 2, beta, 0.1/L, L, n_it, device)
    en2, Psnr2, rec2 = CP_FFT_deblur(P, PT, kern, x_gt, sino, x0, y0, 2, 0.1/L, L, beta, theta, n_it, device)
    en3, Psnr3, rec3 = PPP_FFT_deblur(P, PT, kern, x_gt, sino, x0, y0, 2, 0.1/L, L, beta, lambda_k, n_it, 'Constant', device)
    Psnr0 = np.array(Psnr0)
    Psnr = np.array(Psnr)
    Psnr2 = np.array(Psnr2)
    Psnr3 = np.array(Psnr3)
    print(max(Psnr0), max(Psnr), max(Psnr2), max(Psnr3))
# python main_table_image_debluring_GraREDHP3_restart.py