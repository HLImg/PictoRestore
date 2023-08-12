%% 计算psnr和ssim

clear ; clc ;
% 加载mat文件
hq = load("hq.mat") ;
predict = load("predict.mat") ;

n_images = 1280 ;
psnr_values = zeros(n_images, 1);
ssim_values = zeros(n_images, 1);



for i = 0 : 1279
    hq_key = "gt_" + i ;
    pre_key = "pre_" + i ;

    hq_img = hq.(hq_key) ;
    pre_img = predict.(pre_key) ;

    psnr_values(i + 1) = psnr(pre_img, hq_img) ;
    ssim_values(i + 1) = ssim(pre_img, hq_img) ;
end

mean_psnr = mean(psnr_values) ;
mean_ssim = mean(ssim_values) ;

fprintf("Mean PSNR : %f\n", mean_psnr) ;
fprintf("Mean SSIM : %f \n", mean_ssim) ;