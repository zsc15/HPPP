function [est_im, algo_psnr] = main_exp2_GraRED_P3(imgname, noise_level, degradation_model,index)
addpath(genpath('./tnrd_denoising/'));
% SD, FP, and ADMM methods
addpath(genpath('./minimizers/'));
% contains the default params
addpath(genpath('./parameters/'));
% contains basic functions
addpath(genpath('./helper_functions/'));
% test images for the debluring and super resolution problems, 
% taken from NCSR software package
addpath(genpath('./test_images/'));

% set light_mode = true to run the code in a sub optimal but faster mode
% set light_mode = false to obtain the results reported in the RED paper
light_mode = false;

if light_mode
    fprintf('Running in light mode. ');
    fprintf('Turn off to obatain the results reported in RED paper.\n');
else
    fprintf('Light mode option is off. ');
    fprintf('Reproducing the result in RED paper.\n');
end

%% read the original image

file_name = [imgname, '.tif'];%%bike, butterfly, house, parrot, zebra,girl,flower, hat

fprintf('Reading %s image...', file_name);
orig_im = imread(['./test_images/' file_name]);
orig_im = double(orig_im);

fprintf(' Done.\n');


%% define the degradation model

% choose the secenrio: 'UniformBlur', 'GaussianBlur', or 'Downscale'
%degradation_model = 'UniformBlur';

fprintf('Test case: %s degradation model.\n', degradation_model);

switch degradation_model
    case 'UniformBlur'
        % noise level
        input_sigma = noise_level;
        % filter size
        psf_sz = 9;
        % create uniform filter
        psf = fspecial('average', psf_sz);
        % use fft to solve a system of linear equations in closed form
        use_fft = true;
        % create a function-handle to blur the image
        ForwardFunc = ...
            @(in_im) imfilter(in_im,psf,'conv','same','circular');
        % the psf is symmetric, i.e., the ForwardFunc and BackwardFunc
        % are the same
        BackwardFunc = ForwardFunc;
        % special initialization (e.g. the output of other method)
        % set to identity mapping
        InitEstFunc = @(in_im) in_im;
        
    case 'GaussianBlur'
        % noise level
        input_sigma = noise_level;
        % filter size
        psf_sz = 25;
        % std of the Gaussian filter
        gaussian_std = 1.6;
        % create gaussian filter
        psf = fspecial('gaussian', psf_sz, gaussian_std);
        % use fft to solve a system of linear equations in closed form
        use_fft = true;
        % create a function handle to blur the image
        ForwardFunc = ...
            @(in_im) imfilter(in_im,psf,'conv','same','circular');
        % the psf is symmetric, i.e., the ForwardFunc and BackwardFunc
        % are the same
        BackwardFunc = ForwardFunc;
        % special initialization (e.g. the output of other method)
        % set to identity mapping
        InitEstFunc = @(in_im) in_im;
    case 'WidthGaussianBlur'
        % noise level
        input_sigma = noise_level;
        % filter size
        psf_sz = 25;
        % std of the Gaussian filter
        gaussian_std = 2.4;
        % create gaussian filter
        psf = fspecial('gaussian', psf_sz, gaussian_std);
        % use fft to solve a system of linear equations in closed form
        use_fft = true;
        % create a function handle to blur the image
        ForwardFunc = ...
            @(in_im) imfilter(in_im,psf,'conv','same','circular');
        % the psf is symmetric, i.e., the ForwardFunc and BackwardFunc
        % are the same
        BackwardFunc = ForwardFunc;
        % special initialization (e.g. the output of other method)
        % set to identity mapping
        InitEstFunc = @(in_im) in_im;
    case 'MotionBlur'
        % 加载核文件
        kernels_struct = load('PnP_restoration/kernels/Levin09.mat');
        kernels = kernels_struct.kernels;

        % noise level
        input_sigma = noise_level;
        psf = kernels{index};  % 选择第一个核，或者根据需要选择其他核
        % use fft to solve a system of linear equations in closed form
        use_fft = true;
        % create a function handle to blur the image
        ForwardFunc = ...
            @(in_im) imfilter(in_im, psf, 'conv', 'same', 'circular');
        % the psf is symmetric, i.e., the ForwardFunc and BackwardFunc
        % are the same
        BackwardFunc = ForwardFunc;
        % special initialization (e.g. the output of other method)
        % set to identity mapping
        InitEstFunc = @(in_im) in_im;
        
    case 'Downscale'
        % noise level
        input_sigma = noise_level;
        % filter size
        psf_sz = 7;
        % std of the Gaussian filter
        gaussian_std = 1.6;
        % create gaussian filter
        psf = fspecial('gaussian', psf_sz, gaussian_std);
        % scaling factor
        scale = 3;
        % compute the size of the low-res image
        lr_im_sz = [ceil(size(orig_im,1)/scale),...
                    ceil(size(orig_im,2)/scale)];        
        % create the degradation operator
        H = CreateBlurAndDecimationOperator(scale,lr_im_sz,psf);
        % downscale
        ForwardFunc = @(in_im) reshape(H*in_im(:),lr_im_sz);        
        % upscale
        BackwardFunc = @(in_im) reshape(H'*in_im(:),scale*lr_im_sz);
        % special initialization (e.g. the output of other method)
        % use bicubic upscaler
        InitEstFunc = @(in_im) imresize(in_im,scale,'bicubic');
        
    otherwise
        error('Degradation model is not defined');
end


%% degrade the original image

switch degradation_model
    case {'UniformBlur', 'GaussianBlur','WidthGaussianBlur','MotionBlur'}
        fprintf('Blurring...');
        % blur each channel using the ForwardFunc
        input_im = zeros( size(orig_im) );
        for ch_id = 1:size(orig_im,3)
            input_im(:,:,ch_id) = ForwardFunc(orig_im(:,:,ch_id));
        end
        % use 'seed' = 0 to be consistent with the experiments in NCSR
        randn('seed', 0);

    case 'Downscale'
        fprintf('Downscaling...');
        % blur the image, similar to the degradation process of NCSR
        input_im = Blur(orig_im, psf);
        % decimate
        input_im = input_im(1:scale:end,1:scale:end,:);
        % use 'state' = 0 to be consistent with the experiments in NCSR
        randn('state', 0);

    otherwise
        error('Degradation model is not defined');
end

% add noise
fprintf(' Adding noise...');
input_im = input_im + input_sigma*randn(size(input_im));

% convert to YCbCr color space if needed
input_luma_im = PrepareImage(input_im);
orig_luma_im = PrepareImage(orig_im);
name = file_name(1:end-4); % 你想要创建的文件夹的名字
folderName = ['./matlab_results/',name];
if ~exist(folderName, 'dir')
   mkdir(folderName)
end
if strcmp(degradation_model,'Downscale')
    % upscale using bicubic
    input_im = imresize(input_im,scale,'bicubic');
    input_im = input_im(1:size(orig_im,1), 1:size(orig_im,2), :); 
end
fprintf(' Done.\n');
psnr_input = ComputePSNR(orig_im, input_im);


%% minimize the laplacian regularization functional via Steepest Descent

fprintf('Restoring using Gradient RED P3: GraRED-P3\n');

switch degradation_model
    case 'UniformBlur'
        params_admm = GetUniformDeblurADMMParams(light_mode, psf, use_fft);
    case 'WidthGaussianBlur'
        params_admm = GetGaussianDeblurADMMParams(light_mode, psf, use_fft);
    case 'GaussianBlur'
        params_admm = GetGaussianDeblurADMMParams(light_mode, psf, use_fft);
    case 'Downscale'
        %assert(exist('use_fft','var') == 0);
        params_admm = GetSuperResADMMParams(light_mode);
        %params_admm = GetSR_SCFP(light_mode,psf, use_fft);       
    case 'MotionBlur'
        %assert(exist('use_fft','var') == 0);
        %params_admm = GetSuperResADMMParams(light_mode);
        params_admm = GetSR_SCFP(light_mode,psf, use_fft);      
end
params_admm.outer_iters = 500;
[est_im, algo_psnr] = RunGraRED_P3(input_luma_im,...
                             ForwardFunc,...
                             BackwardFunc,...
                             InitEstFunc,...
                             0.2,...
                             40,...
                             input_sigma,...
                             params_admm,...
                             orig_luma_im);
% convert back to rgb if needed
out_im = MergeChannels(input_im,est_im);
imwrite(uint8(out_im),['./results/GraRED_HP3_' file_name]);
fprintf('Done.\n');
myCluster = parcluster('local');
jobs = myCluster.Jobs;
for i = 1:length(jobs)
    if strcmp(jobs(i).State, 'running')
        delete(jobs(i));
    end
end
end
