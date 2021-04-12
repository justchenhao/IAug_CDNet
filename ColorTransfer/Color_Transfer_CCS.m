function est_im = Color_Transfer_CCS(source,target)
%% We inspired this algorithm from different sources 
%
%   Color_Transfor_CCS(SOURCE,TARGET) returns the colour transfered source
%   image SOURCE according to the target image TARGET.
%
rgb_s = reshape(im2double(source),[],3)';
rgb_t = reshape(im2double(target),[],3)';

%% compute mean of the three aaxis : 
mean_s = mean(rgb_s,2);
mean_t = mean(rgb_t,2);

%% compute covariance for th two images: 
cov_s = cov(rgb_s');
cov_t = cov(rgb_t');
%% decompose covariances matrix using svd :
[U_s,A_s,~] = svd(cov_s);
[U_t,A_t,~] = svd(cov_t);

rgbh_s = [rgb_s;ones(1,size(rgb_s,2))];

%% compute transformation ( Rotation,Translation and Scaling )
% translations
T_t = eye(4); T_t(1:3,4) =  mean_t; %Target Image
T_s = eye(4); T_s(1:3,4) = -mean_s; %Source Image
% rotations
% Blkdiag : placing the matrix in diagonal
R_t = blkdiag(U_t,1);  R_s = blkdiag(inv(U_s),1);
% scalings :
% Note : for S_t ,after searching about previous work , we found that the
% S_t values must be taken as sqrt of the eigenvalues.(see original paper)
S_t = diag([diag(A_t).^(0.5);1]);
S_s = diag([diag(A_s).^(-0.5);1]);
%% estimation of rgbh_e : 
rgbh_e = T_t*R_t*S_t*S_s*R_s*T_s*rgbh_s;   % estimated RGBs
rgb_e = rgbh_e(1:3,:);
%% final result
est_im = reshape(rgb_e',size(source));
end