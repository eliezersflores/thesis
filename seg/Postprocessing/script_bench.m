clear all;
close all;
clc;

% EDIT HERE (1 of 4):
verbose = 0;

% EDIT HERE (2 of 4):
imgs_dir_root = '/home/eliezer/Área de Trabalho/Drive/PESQUISAS E PROJETOS/thesis/MClass-D';
imgs_dir_melanoma = [imgs_dir_root '/melanoma'];
imgs_dir_notmelanoma = [imgs_dir_root '/notmelanoma'];

% EDIT HERE (3 of 4):
segs_dir_root = '/home/eliezer/Área de Trabalho/Drive/PESQUISAS E PROJETOS/thesis/seg/Segmentation/MClass-D_K3_A1';
segs_dir_melanoma = [segs_dir_root '/melanoma'];
segs_dir_notmelanoma = [segs_dir_root '/notmelanoma'];

% EDIT HERE (4 of 4):
dst_name = 'dictK3_mclassd';

fid = fopen([imgs_dir_root '_preprocessed/elapsed_time.csv']);
preprocessing_time = cell2mat(textscan(fid, '%f', 'Delimiter', ','));
fclose(fid);

fid = fopen([segs_dir_root '/elapsed_time.csv']);
segmentation_time = cell2mat(textscan(fid, '%f', 'Delimiter', ','));
fclose(fid);

curr_dir = pwd;
dst_dir_root = [curr_dir '/' dst_name];

if strcmp(dst_name, 'dictK3A5_dermis')
 
    copyfile(segs_dir_root, dst_name);
    
else

    dst_dir_melanoma = [dst_dir_root '/melanoma'];
    dst_dir_notmelanoma = [dst_dir_root '/notmelanoma'];
    if ~exist(dst_dir_root, 'dir')
        mkdir(dst_dir_root);
        mkdir(dst_dir_melanoma);
        mkdir(dst_dir_notmelanoma);
    end

    postprocessing_time = [];

    if verbose == 1

        dst_dir_details_root = [curr_dir '/' dst_name '_details'];
        dst_dir_details_melanoma = [dst_dir_details_root '/melanoma'];
        dst_dir_details_notmelanoma = [dst_dir_details_root '/notmelanoma'];

        if ~exist(dst_dir_details_root, 'dir')
            mkdir(dst_dir_details_root);
            mkdir(dst_dir_details_melanoma);
            mkdir(dst_dir_details_notmelanoma);
        end    

    end

    dirs_cell = cell(4,2);
    dirs_cell{1,1} = imgs_dir_melanoma;
    dirs_cell{1,2} = imgs_dir_notmelanoma;
    dirs_cell{2,1} = segs_dir_melanoma;
    dirs_cell{2,2} = segs_dir_notmelanoma;
    dirs_cell{3,1} = dst_dir_melanoma;
    dirs_cell{3,2} = dst_dir_notmelanoma;

    if verbose == 1

        dirs_cell{4,1} = dst_dir_details_melanoma;
        dirs_cell{4,2} = dst_dir_details_notmelanoma;

    end

    for dirs = dirs_cell

        imgs_dir = dirs{1};
        segs_dir = dirs{2};
        dst_dir = dirs{3};

        if verbose == 1

            dst_dir_details = dirs{4};

        end

        cd(segs_dir);
        dir_struct = dir('*.png');
        cd(curr_dir);
        nimgs = numel(dir_struct); % Getting the number of images and file names.

        for i = 1:nimgs

            segs_dir_str_parts = split(segs_dir, '/');
            fprintf('Post-processing %s image %d of %d...\n', segs_dir_str_parts{end}, i, nimgs);

            name = dir_struct(i).name(1:end-4);
            orig_img = imread([imgs_dir '/' name '.jpg']);
            orig_seg = imread([segs_dir '/' name '.png']);

            tic; 

            if verbose == 1

                imwrite(orig_seg, [dst_dir_details '/' dst_name '_modif0.png']);

            end

            nr = size(orig_seg, 1);
            nc = size(orig_seg, 2);

            kappa = floor(0.2*min(nr, nc));
            mask = zeros(nr, nc);
            mask(1:kappa, 1:kappa) = 1;
            mask(1:kappa, end-kappa+1:end) = 1;
            mask(end-kappa+1:end, 1:kappa) = 1;
            mask(end-kappa+1:end, end-kappa+1:end) = 1;

            if sum(orig_seg(mask==1)~=0) > sum(orig_seg(mask==1)==0)
                orig_seg = max(max(orig_seg)) - orig_seg;
            end

            w = 2*round((0.01*max(nr, nc)+1)/2) - 1;

            gauss_mask = fspecial('gaussian', [nr nc] , 0.1*max(nr, nc));

            regs_mask = labelmatrix(bwconncomp(orig_seg));
            n_regs = max(regs_mask(:));
            % figure; imshow(regs_mask, []);

            if n_regs > 1

                obj_fun = zeros(1, n_regs);

                for reg_idx = 1:n_regs

                    obj_fun(reg_idx) = sum(sum((regs_mask == reg_idx).*gauss_mask));

                end

                [~, opt_idx] = max(obj_fun);

                modif_seg1 = regs_mask == opt_idx;

            else

                modif_seg1 = orig_seg;

            end

            if verbose == 1

                imwrite(modif_seg1, [dst_dir_details '/' dst_name '_modif1.png']);

            end

            gauss_mask2 = fspecial('gaussian', [3*w, 3*w], w);
            modif_seg2 = imfilter(im2double(modif_seg1), gauss_mask2, 'replicate');
            modif_seg3 = modif_seg2 > 0.1;

            if verbose == 1

                imwrite(modif_seg3, [dst_dir_details '/' name '_modif2.png']);

            end

            modif_seg4 = imfill(modif_seg3, 'holes');

            seg = modif_seg4;

            if verbose == 1

                imwrite(modif_seg4, [dst_dir_details '/' name '_modif3.png']);

            end        

            postprocessing_time = [postprocessing_time; toc];

            imwrite(modif_seg4, [dst_dir '/' name '.png']);

            width = 5; 
            contour = abs(imdilate(seg, ones(2*width + 1, 2*width + 1)) - seg) > 0;
            overlayed = imoverlay(orig_img, contour, [1 0 0]); 

            imwrite(overlayed, [dst_dir '/' name '.jpg']);

        end    

    end

end

if startsWith(dst_name, 'dict') && contains(dst_name, 'A5') % dictK3A5
    elapsed_time = preprocessing_time + segmentation_time;
elseif startsWith(dst_name, 'dict') && ~contains(dst_name, 'A1') % dictK3A2, dictK3A3 and dictK3A4
    elapsed_time =  preprocessing_time + segmentation_time + postprocessing_time;
else % dictK3A1 and other methods. 
    elapsed_time = segmentation_time + postprocessing_time;
end

csvwrite([dst_dir_root '/elapsed_time.csv'], elapsed_time)