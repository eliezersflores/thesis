function [posSkinBordersOptimized,assessedRectangles] = aux_find_reference_skin(RGB, lab,wse)
% select the region with the lower total coefficient of variation for the three channels.
% enter up to 1/4 of the lower side of the image

% P.s.: this function was slightly modified by Eliezer Flores to return all
% the assessed rectangles. 

%disp('. search for the region with lower total variance channel-wise.')

[nr,nc,nb] = size(lab);

aux = zeros(nr,nc);
aux(2:end-1,2:end-1)=1;

se = strel('square',wse);

x_vect = double(reshape(lab,nr*nc,3));

Tab_var = [];
curvar = inf;
posSkinBordersOptimized = aux;

n = round(1/4*min([nr,nc])/wse); % each step it moves only wse/2

assessedRectangles = cell(1, n);
for k=1:n
    
    aux2 = imerode(aux,se);
    posSkinBorders = aux-aux2;
    posSkinBorders_vect = reshape(posSkinBorders,nr*nc,1);
            
    [~, L] = bwboundaries(posSkinBorders); % added by Eliezer Flores
    assessedRectangles{k} = boundarymask(L); % added by Eliezer Flores
    
    X = (x_vect(posSkinBorders_vect==1,:));
    % totvar = sum(var(X,1));
    totvar = sum(std(X,1)./mean(X,1));
    
    if totvar < curvar
        %fprintf('%d\n',k);
        posSkinBordersOptimized = posSkinBorders;
        curvar = totvar;
        % fprintf('. update background location at step %3.0d. with total CV = %5.3f\n',k,curvar)
    end
    %     fprintf('. update at step %3.0d. with tot variance %5.2f\n',k,totvar)
    aux = aux2;
end
% imshow(posSkinBorders)
% fprintf('. final background location tested step %3.0d. with total CV = %5.3f\n',k,totvar)

