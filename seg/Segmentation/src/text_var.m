function s = text_var(image)

im2 = double(image);

L = sum(im2,3)/3;
        
[lin col p] = size(im2);
tams = 7:4:43;
vls = zeros(lin,col,length(tams));
for i = 1:length(tams);
    t = tams(i);
    s = t/7;
    h = fspecial('gaussian',[t t],s);

    m = imfilter(L,h,'symmetric','same') + eps;
    vls(:,:,i) = ((L./m)-L);
end
s = max(vls,[],3);

s = (s-min(min(s)))./(max(max(s))-min(min(s)));

end

