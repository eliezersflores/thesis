function img = patches2im(patches, w, nr, nc, nb, diff_nr, diff_nc)

nr2 = nr - diff_nr;
nc2 = nc - diff_nc;

img = [];
for b = 1:nb
    img = cat(3, img, col2im(patches((b-1)*w*w+1:b*w*w,:), [w w], [nr2 nc2], 'distinct'));
end




    

