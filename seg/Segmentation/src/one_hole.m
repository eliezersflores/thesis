function [im_out] = one_hole(im_in)
    
[M,N] = size(im_in);
gaussian = gaussian2D(M,N);

cell_holes = holes(im_in);
n_holes = numel(cell_holes);

measures = zeros(1,n_holes);

for i = 1:n_holes   
    measures(i) = sum(sum(cell_holes{i}.*gaussian));
end;

[~,max_pos] = max(measures);
im_out = im2bw(cell_holes{max_pos});

end

function [ cellHoles ] = holes( bwImage )
    
    [M,N] = size(bwImage);

    CC = bwconncomp(bwImage);
    S = regionprops(CC,'PixelList');
    numberOfHoles = numel(S);
    
    cellHoles = cell(1,numberOfHoles);
    
    for i = 1:numberOfHoles
        
        hole = zeros(M,N);
        points = S(i).PixelList;
        
        for j = 1:size(points,1)
            
            cords = points(j,:);
            hole(cords(2),cords(1)) = 1;

        end;
        
        cellHoles{i} = hole;
       
    end;
    
end

function gauss_mask = gaussian2D(M,N)

gauss_mask = zeros(M,N);

for x = 1:M
    for y = 1:N
        x0 = round(M/2);
        y0 = round(N/2);
        gauss_mask(x,y) = exp(-(((x-x0)^2/(5*M)) + ((y-y0)^2)/(5*N)));
    end;
end;

end
