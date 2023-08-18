function [seg, assessed_rectangles, optimal_rectangle, D0] = dict_segmentation(img_rgb, w, K)

rng(0);

% INPUTS:
% img_rgb  - input image in the RGB colorspace.
% K        - number of atoms to the initial NMF-learned dicitionary.

% OUTPUT:
% seg      - segmentation result.

%[input_img_tmp, assessed_rectangles, optimal_rectangle] = aux_EuclidianDistanceIntensity(img_rgb, w);
input_img_tmp = double(mean(img_rgb, 3));

input_image = (input_img_tmp - min(input_img_tmp(:))) / (max(input_img_tmp(:)) - min(input_img_tmp(:)));

nr = size(input_image, 1);
nc = size(input_image, 2);
nb = size(input_image, 3);

[Y, diff_nr, diff_nc] = im2patches(input_image, w);

opt = statset('maxiter', 100);
[D0, X0] = nnmf(Y, K, 'opt', opt);

[D, X] = uitdl(Y, D0, X0);

cidxs = ngc(X);

Y(:, cidxs == 1) = 0;
Y(:, cidxs == 2) = 1;

seg_tmp1 = patches2im(Y, w, nr, nc, nb, diff_nr, diff_nc);
seg_tmp2 = mean(seg_tmp1, 3);

nr2 = nr - diff_nr;
nc2 = nc - diff_nc;

if sum(seg_tmp2(end,:)) + sum(seg_tmp2(:,end)) > (nr + nc)/2
    seg = ones(nr, nc); 
else
    seg = zeros(nr, nc);
end

seg(1:nr2,1:nc2) = seg_tmp2;


