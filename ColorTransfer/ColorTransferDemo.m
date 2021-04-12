%% Demostration of Colour Transfer methods :
close all
clear all
clc
%% Read Source & target Images :
%You may use the database : files/.... or Examples files 
Im_target = imread('./samples/targets/train_1_00014_040_037.png');
Im_source = imread('./samples/sources/train_1_00012_054_056.png');

%%
Im_trg_d=  im2double(Im_target);
Im_src_d = im2double(Im_source );
%% Correlated Color Space : RGB 
tic,
IR1= Color_Transfer_CCS(Im_src_d,Im_trg_d);  %Call the C-Color Transfer function ( source,target )
time=toc;
%% Results : 
figure; 
subplot(1,3,1); imshow(Im_source); title('Original Image'); axis off
subplot(1,3,2); imshow(Im_target); title('Target Image'); axis off
subplot(1,3,3); imshow(IR1); title('Result Image '); axis off

imwrite(IR1,'result_Image.jpg');
fprintf('the color transfer algorithm took : %6.2f seconds \n',time);

