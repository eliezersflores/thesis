function [intensity,assessedRectangles,optimalRectangle,maskSkinBorders,vectColorSkinLAB,lab] = aux_EuclidianDistanceIntensity(RGB,wse)
% This function computes an intensity image where each pixel value is
% proportional to the Euclidian distance to the (estimated) colour of the background skin.
% Computations are done in the approx. perceptually uniform Lab colour space.
%
%  Input
%   - RGB: is a colour image of the skin lesion, values 0-255
%   - wse: width of the border region, heuristic. If not set a default will
%          be used
%
%  Outputs
%   - intensity:        is a grey scale image, double, values scaled into 0-255
%   - maskSkinBorders:   binary mask showing the selected region near the image border
%                       containing (hopefully) mainly skin pixels
%   - vectColorSkinLAB: vector of the estimated colour of the skin
%
%   $by Maciel Zortea - mzortea@gmail.com
%   $Revision: 1.0 $  $Date:  9/VI/2015$

% P.s.: this function was slightly modified by Eliezer Flores to return all
% the assessed rectangles and to return the optimal rectangle separately
% instead of 'maskSkinBorders'.

if max(RGB(:))>255 || min(RGB(:))<0
    error(' RGB: is a color image of the skin lesion, values 0-255')
end

[nr,nc,nb] = size(RGB);

if nargin == 1
    % define input parameters: heuristic
    wse = (0.01*max([nr,nc]));
    wse = 2.*round((wse+1)/2)-1; % rounds toward the closest odd number.
    %Otherwise, medfilt2 has problems if input is even
end

%fprintf('. Compute the Euclidian distance intensity image')

C = makecform('srgb2lab');
%lab = applycform(RGB,C);
lab = applycform(double(RGB)/255,C);

% Attempt to select skin pixels, near the border of the images
[maskSkinBordersOptimized, assessedRectangles] = aux_find_reference_skin(RGB, lab,4*wse);
%overlayed = imoverlay(RGB, borders, [0 0 1]);
%imwrite(overlayed, ['rectangular_shaped_region_NM3_' num2str(k) '.png'], 'png');

maskSkinBorders = maskSkinBordersOptimized;
maskSkinBorders_vect = reshape(maskSkinBorders,nr*nc,1);

[~,L] = bwboundaries(maskSkinBorders); % Added by Eliezer Flores
optimalRectangle = boundarymask(L); % Added by Eliezer Flores

x_vect = double(reshape(lab,nr*nc,3));

X = (x_vect(maskSkinBorders_vect==1,:));

% Estimate a basic statistic from the Skin: median values in Lab
% Note: no filtering yet
vectColorSkinLAB = median(X);
%fprintf('. LAB median of %d (skin) pixels near the borders of the image [%3.2f %3.2f %3.2f]\n',size(X,1),vectColorSkinLAB)

% Compute the Euclidian distance of all image pixels to the LABColorSkin of
% the skin.
% Note: this changes the 3D Lab to a 1D distance, easier to segment
% using stantard methods
y = sqrt(sum([x_vect - repmat(vectColorSkinLAB,nr*nc,1)].^2,2));
intensity = reshape(y,nr,nc);

% Filter the resulting image to eliminate noise
%  channel = medfilt2(y_img, [wse wse]);

end