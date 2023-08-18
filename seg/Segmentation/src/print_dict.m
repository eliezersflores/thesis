function print_dict(D, file_name)

    m = size(D, 1); 

    ms = sqrt(m); 

    if ms*ms == m % If working with an intensity image
        nb = 1;
        w = ms; 
    else % If working with a 3-channels color image
        nb = 3;
        w = sqrt(m/3); 
    end

    K = size(D, 2); 
    s = ceil(sqrt(K)); % suplot size (width and height). 

    dx = 1/s; 
    dy = 1/s; 
    hor_space = 0.05; 
    ver_space = 0.05;

    for k = 1:K

        row = floor((k-1)/s) + 1; 
        col = mod(k-1, s) + 1; 

        xi = (col-1)*dx; 
        yi = (s - row)*dy;

        pos = [xi yi dx-hor_space dy-ver_space];

        subplot(s, s, k); imagesc(patches2im(D(:,k), w, w, w, nb, 0, 0)); colormap('gray'); axis off;
        ax = gca;
        ax.Position = pos;

    end 

    saveas(gcf, file_name);
    
end

