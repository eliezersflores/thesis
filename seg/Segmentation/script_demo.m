clear all;
close all;
clc;

addpath('src');
addpath('Ncut_9');

dataset_name = 'dermis'; % EDIT HERE
K = 9; % EDIT HERE
alternative_method = 0; % EDIT HERE

curr_dir = pwd;

% Setting the source dataset dir:
src_dir = ['/home/eliezer/Dropbox/Datasets/msls_atten_' dataset_name];

     
name = 'SSM23';
img_rgb = imread([src_dir '/' name '.jpg']);

nr = size(img_rgb, 1);
nc = size(img_rgb, 2);

w = 2*round((0.01*max(nr, nc)+1)/2) - 1;

if alternative_method == 0

    seg = dict_segmentation(img_rgb, w, K);

elseif alternative_method == 1

    seg = dict_segmentation_a1(img_rgb, w, K);

elseif alternative_method == 2

    addpath('ompbox10');
    addpath('ksvdbox13');

    seg = dict_segmentation_a2(img_rgb, w, K); 

elseif alternative_method == 3

    addpath(genpath('vlfeat-0.9.20'));
    run('vl_setup');

    seg = dict_segmentation_a3(img_rgb, w, K); 

end
        
