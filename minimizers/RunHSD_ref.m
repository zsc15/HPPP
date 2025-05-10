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

function [im_out, psnr_out] = RunHSD_ref(y, ForwardFunc, BackwardFunc,...
    InitEstFunc, input_sigma, params, orig_im)

% print info every PRINT_MOD steps
QUIET = 0;
PRINT_MOD = floor(params.outer_iters/10);
if ~QUIET
    fprintf('%7s\t%10s\t%12s\t%12s\t%12s \n', 'iter', 'PSNR', 'objective','relative error', 'fixed-point error');
end

% parameters
lambda = params.lambda;
outer_iters = params.outer_iters;
effective_sigma = params.effective_sigma;

% compute step size
mu = 2/(1/(input_sigma^2) + lambda);

% initialization
u_est = InitEstFunc(y);
% v0 = 0.5*Denoiser(x_est, effective_sigma)+0.5*x_est;
v_est0 = 0.5*Denoiser(u_est, effective_sigma)+0.5*u_est;
mu_0 = 2*effective_sigma^2;
for k = 1:1:outer_iters
    % denoise
    mu_k = 2*k^(-0.1);
    v_est = 0.5*Denoiser(u_est, effective_sigma)+0.5*u_est;
    u_est = v_est - mu_k*BackwardFunc(ForwardFunc(2*v_est-v_est0) - y)/(input_sigma^2);
    v_est0 = v_est;% older v_k
    % update the solution
    
    % project to [0,255]
    x_est = max( min(u_est, 255), 0);
    
    if ~QUIET && (mod(k,PRINT_MOD) == 0 || k == outer_iters)
        % evaluate the cost function
        fun_val = fidelity_CostFunc(y, x_est, ForwardFunc, input_sigma);
        r_err = relative_error(y, x_est, ForwardFunc, input_sigma);
        fp_err = fp_error(u_est, effective_sigma);
        im_out = x_est(1:size(orig_im,1), 1:size(orig_im,2));
        psnr_out = ComputePSNR(orig_im, im_out);
        fprintf('%7i %12.5f %12.5f %12.5f %12.5f \n', k, psnr_out, fun_val, r_err, fp_err);
    end
end
x_final = 0.5*Denoiser(u_est, effective_sigma)+0.5*u_est;
im_out = x_final(1:size(orig_im,1), 1:size(orig_im,2));
psnr_out = ComputePSNR(orig_im, im_out);

return
function fun_val = fidelity_CostFunc(y, x_est, ForwardFunc, input_sigma)

fun_val = sum(sum((ForwardFunc(x_est)-y).^2))/(2*input_sigma^2);

return

function r_error = relative_error(y, x_est, ForwardFunc, input_sigma)
%*sum(sum(x_est.^2))
fun_val1 = sum(sum((ForwardFunc(x_est)-y).^2))/(2*input_sigma^2);
r_error = fun_val1/sum(sum(x_est));
return

function fp_err = fp_error(u_est, effective_sigma)
f_u = 0.5*Denoiser(u_est, effective_sigma)+0.5*u_est;
fp_err = sum(sum(u_est-f_u).^2)/sum(sum(u_est));
return 
