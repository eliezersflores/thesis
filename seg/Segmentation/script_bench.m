clear all;
close all;
clc;

addpath('src');
addpath('src/Ncut_9');

% EDIT HERE (1 of 4)
verbose = 0;

% EDIT HERE (2 of 4)
dataset_name = 'MClass-D';

% EDIT HERE (3 of 4)
%for K = 3:9
for K = 3:3

    % EDIT HERE (4 of 4)
    %for alternative_method = 1:4
    for alternative_method = 1:1

        fprintf('* With K = %d and alternative_method = %d\n', K, alternative_method);

        curr_dir = pwd;

        %% Setting the source and destination dirs:

        if alternative_method == 1
            src_dir_root = ['/home/eliezer/Área de Trabalho/thesis/' dataset_name];
        else
            src_dir_root = ['/home/eliezer/Área de Trabalho/thesis/' dataset_name '_preprocessed'];
        end
        src_dir_melanoma = [src_dir_root '/melanoma'];
        src_dir_notmelanoma = [src_dir_root '/notmelanoma'];

        if alternative_method == 0
            dst_dir_root = [curr_dir '/' dataset_name '_K' num2str(K)];
        else
            dst_dir_root = [curr_dir '/' dataset_name '_K' num2str(K) '_A' num2str(alternative_method)];
        end
        dst_dir_melanoma = [dst_dir_root '/melanoma'];
        dst_dir_notmelanoma = [dst_dir_root '/notmelanoma'];

        if ~exist(dst_dir_root, 'dir')
            mkdir(dst_dir_root);
            mkdir(dst_dir_melanoma);
            mkdir(dst_dir_notmelanoma);
        end

        if verbose == 1

            if alternative_method == 0
                dst_dir_details_root = [curr_dir '/' dataset_name '_K' num2str(K) '_details'];
            else
                dst_dir_details_root = [curr_dir '/' dataset_name '_K' num2str(K) '_A' num2str(alternative_method) '_details'];
            end
            dst_dir_details_melanoma = [dst_dir_details_root '/melanoma'];
            dst_dir_details_notmelanoma = [dst_dir_details_root '/notmelanoma'];

            if ~exist(dst_dir_details_root, 'dir')
                mkdir(dst_dir_details_root);
                mkdir(dst_dir_details_melanoma);
                mkdir(dst_dir_details_notmelanoma);
            end

        end

        %% Segmenting all the images of the selected dataset.

        dirs_cell = cell(3,2);
        dirs_cell{1,1} = src_dir_melanoma;
        dirs_cell{1,2} = src_dir_notmelanoma;
        dirs_cell{2,1} = dst_dir_melanoma;
        dirs_cell{2,2} = dst_dir_notmelanoma;

        if verbose == 1
            dirs_cell{3,1} = dst_dir_details_melanoma;
            dirs_cell{3,2} = dst_dir_details_notmelanoma;
        end

        elapsed_times = [];
        for dirs = dirs_cell

            src_dir = dirs{1};
            dst_dir = dirs{2};

            if verbose == 1
                dst_dir_details = dirs{3};
            end

            cd(src_dir);
            dir_struct = dir('*.jpg');
            cd(curr_dir);
            nimgs = numel(dir_struct); 

            for i = 1:nimgs

                src_dir_str_parts = split(src_dir, '/');
                fprintf('Segmenting %s image %d of %d...\n', src_dir_str_parts{end}, i, nimgs);

                name = dir_struct(i).name(1:end-4);
                img = imread([src_dir '/' name '.jpg']);

                tic;

                nr = size(img, 1);
                nc = size(img, 2);

                w = 2*round((0.01*max(nr, nc)+1)/2) - 1;

                if alternative_method == 2

                    seg = dict_segmentation_A2(img, w, K); 

                elseif alternative_method == 3

                    addpath('src/ompbox10');
                    addpath('src/ksvdbox13');                    
                   
                    seg = dict_segmentation_A3(img, w, K); 

                elseif alternative_method == 4
                    
                    addpath(genpath('src/vlfeat-0.9.20'));
                    run('vl_setup');
                    
                    seg = dict_segmentation_A4(img, w, K); 
                    
                else

                    if verbose == 0

                        seg = dict_segmentation(img, w, K);

                    else

                        [seg, assessed_rectangles, optimal_rectangle, D0] = dict_segmentation(img, w, K);

                        for r = 1:length(assessed_rectangles)    
                            if sum(sum(assessed_rectangles{r})) == sum(sum(optimal_rectangle))
                                imwrite(imoverlay(img, assessed_rectangles{r}, [0 0 1]), [dst_dir_details '/' name '_rect' num2str(r) '.png']);
                            else
                                imwrite(imoverlay(img, assessed_rectangles{r}, [0 0 0]), [dst_dir_details '/' name '_rect' num2str(r) '.png']);
                            end
                        end

                        print_dict(D0, [dst_dir_details '/' name '_D0.png']);

                    end

                end

                elapsed_time = toc;
                elapsed_times = [elapsed_times elapsed_time];

                imwrite(seg, [dst_dir '/' name '.png']);

            end

        end

        csvwrite([dst_dir_root '/elapsed_time.csv'], elapsed_times);

    end % End for 'alternative_method'.
    
end % End for 'K'.
