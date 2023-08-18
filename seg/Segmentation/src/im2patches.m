function [patches, diff_nr, diff_nc] = im2patches(img, w)

nr = size(img, 1);
nc = size(img, 2);
nb = size(img, 3);

nr2 = floor(nr/w)*w;
nc2 = floor(nc/w)*w;
img2 = img(1:nr2,1:nc2,:);

patches = [];
for b = 1:nb
    patches = [patches; im2col(img2(:,:,b), [w w],'distinct')];
end

diff_nr = nr - nr2;
diff_nc = nc - nc2;