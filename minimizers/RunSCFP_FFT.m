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
%   via steepest descend method.
%   Please refer to Section 4.2 in the paper for more details:
%   "Deploying the Denoising Engine for Solving Inverse Problems -- 
%   Gradient Descent Methods".
%
% Inputs:
%   y - the input image
%   ForwardFunc - the degradation operator H
%   BackwardFunc - the transpose of the degradation operator H
%   InitEstFunc - special initialization (e.g. the output of other method)
%   input_sigma - noise level
%   params.lambda - regularization parameter
%   params.outer_iters - number of total iterations
%   params.effective_sigma - the input noise level to the denoiser
%   orig_im - the original image, used for PSNR evaluation ONLY
%
% Outputs:
%   im_out - the reconstructed image
%   psnr_out - PSNR measured between x_est and orig_im

function [im_out, psnr_out] = RunSCFP_FFT(y, ForwardFunc, BackwardFunc,...
    InitEstFunc, input_sigma, params, orig_im)

% print info every PRINT_MOD steps
QUIET = 0;
PRINT_MOD = floor(params.outer_iters/10);
if ~QUIET
    fprintf('%7s\t%10s\t%12s\t%12s\n', 'iter', 'PSNR', 'objective','relative error');
end

% parameters
lambda = params.lambda;
outer_iters = 1800%params.outer_iters;
effective_sigma = params.effective_sigma;

% compute step size
mu = 2/(1/(effective_sigma^2) + lambda);

% initialization
x_est = InitEstFunc(y);
alpha=0.2;

%[h, w, ~] = size(y);
[h,w,~] = size(x_est);
fft_psf = zeros(h, w);
t = floor(size(params.psf, 1)/2);
fft_psf(h/2+1-t:h/2+1+t, w/2+1-t:w/2+1+t) = params.psf;
FB = fft2( fftshift(fft_psf) );
FBC = conj(FB);


for k = 1:1:outer_iters
    %x_est = x_est(1:size(orig_im,1), 1:size(orig_im,2));
    Lx = Landweber_operator(x_est, y, FB, FBC, input_sigma, 1, 3);
    % denoise
    f_est = Denoiser(Lx, effective_sigma);
    x_est = (1-alpha)*Lx+alpha*f_est;% older v_k
    % update the solution
    
    % project to [0,255]
    x_est = max(min(x_est, 255), 0);

    if ~QUIET && (mod(k,PRINT_MOD) == 0 || k == outer_iters)
        % evaluate the cost function
        fun_val = fidelity_CostFunc(y, x_est, ForwardFunc, input_sigma);
        r_err = relative_error(y, x_est, ForwardFunc, input_sigma);
%         fun_val1 = fidelity_CostFunc(y, w_est, ForwardFunc, input_sigma);
        im_out = x_est(1:size(orig_im,1), 1:size(orig_im,2));
        psnr_out = ComputePSNR(orig_im, im_out);
        fprintf('%7i %12.5f %12.5f %12.5f \n', k, psnr_out, fun_val, r_err);
    end
end
im_out = x_est(1:size(orig_im,1), 1:size(orig_im,2));
psnr_out = ComputePSNR(orig_im, im_out);

return
function fun_val = fidelity_CostFunc(y, x_est, ForwardFunc, input_sigma)
%*sum(sum(x_est.^2))
fun_val = sum(sum((ForwardFunc(x_est)-y).^2))/(2*input_sigma^2);

return
function r_error = relative_error(y, x_est, ForwardFunc, input_sigma)
%*sum(sum(x_est.^2))
fun_val = sum(sum((ForwardFunc(x_est)-y).^2))/(2*input_sigma^2);
r_error = fun_val/sum(sum(x_est));
return
function Px = Projection_On_L2_Ball(x, y, radius)
    % x: point to project
    % y: center of the ball
    % radius: radius of the ball
    if norm(x-y, 2) > radius
        Px = y + (x-y)/norm(x-y, 2)*radius;
    else
        Px = x;
    end
return

function Lx = Landweber_operator(x, y, FB, FBC, delta, alpha, sf)
    % Lx = x+||T(Ax)-Ax||^2/(||A*(T(Ax)-Ax)||^2) A*(T(Ax)-Ax)
    % alpha: relaxed parameter (0,1]
    % Ax
    FBFx = FB .* fftn(x);
    AFx = downsample_matlab(ifftn(FBFx), sf);
    % T(Ax)
    TAx = Projection_On_L2_Ball(AFx, y, delta);
    TAx_norm = norm(TAx-AFx, 'fro')^2;
    % A*(T(Ax)-Ax)
    r = TAx - AFx;
    STr = upsample_matlab(r, sf);
    FBCSTr = FBC .* fftn(STr);
    AstarFBCSTr = real(ifftn(FBCSTr));
    AstarFBCSTr_norm = norm(AstarFBCSTr, 'fro')^2;
    if norm(AFx-y, 'fro') > delta
        Lx = x + alpha* TAx_norm/AstarFBCSTr_norm*AstarFBCSTr;
    else
        Lx = x;
    end
return
function z = upsample_matlab(x, sf)
    z = zeros(size(x, 1)* sf, size(x, 2)* sf, size(x, 3));
    st = 1;
    z(st:sf:end, st:sf:end,:) = x;
return
function y = downsample_matlab(x, sf)
    st =1;
    y = x(st:sf:end, st:sf:end,:);
return