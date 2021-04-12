%% Demostration of Colour Transfer methods :
close all
clear all
clc
%% Read Source & target Images :
%You may use the database : files/.... or Examples files 
Im_target = imread('./samples/targets/train_1_00014_040_037.png');
Im_source = imread('./samples/sources/train_1_00012_054_056.png');

m_target = imread('./samples/targets/label/train_1_00014_040_037.png')/255;
m_source = imread('./samples/sources/label/train_1_00012_054_056.png')/255;
mask2 = imresize(m_target,[size(Im_target,1),size(Im_target,2)]);
mask1 = imresize(m_source,[size(Im_source,1),size(Im_source,2)]);
%%
Im_trg_d=  im2double(Im_target);
Im_src_d = im2double(Im_source );
mask1 = ones(size(Im_src_d,1),size(Im_src_d,2));
mask2 = ones(size(Im_trg_d,1),size(Im_trg_d,2));

%% Correlated Color Space : RGB 
tic,
IR1= Color_Transfer_CCS_with_mask(Im_src_d,Im_trg_d,mask1,mask2);  %Call the C-Color Transfer function ( source,target )
time=toc;
%% Results : 
figure; 
subplot(1,3,1); imshow(Im_source); title('Original Image'); axis off
subplot(1,3,2); imshow(Im_target); title('Target Image'); axis off
subplot(1,3,3); imshow(IR1); title('Result Image '); axis off

imwrite(IR1,'result_Image.jpg');
fprintf('the color transfer algorithm took : %6.2f seconds \n',time);

