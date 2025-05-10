% Copyright 2017 Google Inc.
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
%     https://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.

% Objective:
%   Minimize E(x) = 1/(2sigma^2)||Hx-y||_2^2 + 0.5*lambda*x'*(x-denoise(x))
%   via the ADMM method.
%   Please refer to Section 4.2 in the paper for more details:
%   "Deploying the Denoising Engine for Solving Inverse Problems -- ADMM".
%
% Inputs:
%   y - the input image
%   ForwardFunc - the degradation operator H
%   BackwardFunc - the transpose of the degradation operator H
%   InitEstFunc - special initialization (e.g. the output of other method)
%   input_sigma - noise level
%   params.lambda - regularization parameter
%   params.beta - ADMM parameter
%   params.outer_iters - number of total iterations
%   params.inner_iters - number of steps to minimize Part1 of ADMM
%   params.use_fft - solve the linear system Az = b using FFT rather than
%                    running gradient descent for params.inner_iters. This
%                    feature is suppoterd only for deblurring
%   params.psf - the Point Spread Function (used only when 
%                use_fft == true).
%   params.inner_denoiser_iters - number of steps to minimize Part2 of ADMM
%   params.effective_sigma - the input noise level to the denoiser
%   orig_im - the original image, used for PSNR evaluation ONLY

% Outputs:
%   im_out - the reconstructed image
%   psnr_out - PSNR measured between x_est and orig_im

function [im_out, psnr_out] = RunGraRED_P3(y, ForwardFunc, BackwardFunc,...
    InitEstFunc, lambda_k, Lambda, input_sigma, params, orig_im)

% print info every PRINT_MOD steps 
QUIET = 0;
PRINT_MOD = floor(params.outer_iters/10);
if ~QUIET
    fprintf('%7s\t%10s\t%12s\n', 'iter', 'PSNR', 'objective');
end

% parameters
%w = 1-w;
lambda = params.lambda;
beta = 0.5/Lambda;
outer_iters = params.outer_iters;
inner_denoiser_iters = params.inner_denoiser_iters;
effective_sigma = params.effective_sigma;

% initialization
x_est = InitEstFunc(y);
xa = x_est;
x_old = x_est;
y_old = x_est;
v_est = x_est;
u_est = x_est*0;
Ht_y = BackwardFunc(y)/(input_sigma^2);

% compute the fft of the psf (useful for deblurring)
if isfield(params,'use_fft') && params.use_fft == true
    [h,w,~] = size(x_est);
    fft_psf = zeros(h, w);
    t = floor(size(params.psf, 1)/2);
    fft_psf(h/2+1-t:h/2+1+t, w/2+1-t:w/2+1+t) = params.psf;
    FB = fft2( fftshift(fft_psf) );
    FBC = conj(FB);
    F2B = abs(FB) .^ 2;
    STy = x_est;
end

for k = 1:1:outer_iters
    % Part1 of the ADMM, approximates the solution of:
    % x = argmin_z 1/(2sigma^2)||Hz-y||_2^2 + 0.5*beta||z - v + u||_2^2
    if isfield(params,'use_fft') && params.use_fft == true
        d = prox_solution2(x_old-y_old, FB, FBC, F2B, STy, input_sigma^2*beta, 3);
        %d = max( min(real(ifft2( b./A )), 255), 0);
    end
    
    x_est =(1-lambda_k)*x_old + lambda_k*d;%%%bug, 73行表明w是宽度，导致计算溢出 [h, w, ~] = size(y);
    %x_est = max( min(x_est, 255), 0);
    f_est = y_old + 2*d-x_old;
    y_est = (1-lambda_k)*y_old+lambda_k*(f_est-Denoiser(f_est, effective_sigma));
    x_old = x_est;
    y_old = y_est;
    if ~QUIET && (mod(k,PRINT_MOD) == 0 || k == outer_iters)
        % evaluate the cost function
        fun_val = CostFunc(y, x_est, ForwardFunc, input_sigma,...
            lambda, effective_sigma);
        im_out = x_est(1:size(orig_im,1), 1:size(orig_im,2));
        psnr_out = ComputePSNR(orig_im, im_out);
        fprintf('%7i %12.5f %12.5f \n', k, psnr_out, fun_val);
    end
end

im_out = x_est(1:size(orig_im,1), 1:size(orig_im,2));
psnr_out = ComputePSNR(orig_im, im_out);
return
function Xest = prox_solution2(x, FB, FBC, F2B, FBFy, alpha, sf)
    % 求解l2邻近算子argmin_u ||y-SHu||^2+alpha||u-x||^2保真项系数为1
    FR = FBFy + alpha * fft2(x);
    x1 = FB .* FR;
    FBR = mean(splits2(x1, sf), 3);
    invW = mean(splits2(F2B, sf), 3); % 求平均就是除以d即sf
    invWBR = FBR ./ (invW + alpha);
    FCBinvWBR = FBC .* repmat(invWBR, [1, 1, sf, sf]);
    FX = (FR - FCBinvWBR) / alpha;
    Xest = real(ifft2(FX));
return

% 添加 splits2 函数
function splits = splits2(input, sf)
    % 将输入分割成 sf x sf 的块
    [h, w, c] = size(input);
    h_sf = floor(h / sf);
    w_sf = floor(w / sf);
    splits = reshape(input, h_sf, sf, w_sf, sf, c);
    splits = permute(splits, [1, 3, 2, 4, 5]);
    splits = reshape(splits, h_sf, w_sf, sf*sf, c);
return