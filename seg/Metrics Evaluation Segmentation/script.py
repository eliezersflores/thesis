import cv2
import math
import numpy as np
import os
import sys

from segmentation_metrics import accuracy, sensitivity, specificity, xor_error, intersection_over_union, bf1score
 
curr_dir = os.getcwd()

# EDIT HERE (1 of 1):
#dst_dir = 'Results_Hyperparameter_Tuning'
#dst_dir = 'Results_Model_Selection'
dst_dir = 'Results_SOTA'

if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

# P.s.: the method names below must be folder names in 'Postprocessing' or 'sota'.
methods = []

if dst_dir == 'Results_Hyperparameter_Tuning':
    methods.append('dictK3_dermis')
    methods.append('dictK4_dermis')
    methods.append('dictK5_dermis')
    methods.append('dictK6_dermis')
    methods.append('dictK7_dermis')
    methods.append('dictK8_dermis')
    methods.append('dictK9_dermis')
elif dst_dir == 'Results_Model_Selection':
    methods.append('dictK3_dermis')
    methods.append('dictK3A1_dermis')
    methods.append('dictK3A2_dermis')
    methods.append('dictK3A3_dermis')
    methods.append('dictK3A4_dermis')
    methods.append('dictK3A5_dermis')
elif dst_dir == 'Results_SOTA':
    methods.append('dsnet_dermnet')
    methods.append('dsnet_dermquest')
    methods.append('fcn_dermnet')
    methods.append('fcn_dermquest')
    methods.append('ftu_dermnet')
    methods.append('ftu_dermquest')
    methods.append('jf_dermnet')
    methods.append('jf_dermquest')
    methods.append('pspnet_dermnet')
    methods.append('pspnet_dermquest')
    methods.append('segnet_dermnet')
    methods.append('segnet_dermquest')
    methods.append('unet_dermnet')
    methods.append('unet_dermquest')
    methods.append('dictK3_dermnet')
    methods.append('dictK3_dermquest')

for method in methods:

    fbase = os.path.join(curr_dir, dst_dir, method)
    if os.path.exists(fbase + '_bf1.npy'):
        print(f'{method} files were already there! skipping these ones...')
        continue

    print(f'Working on {method}...')

    if dst_dir == 'Results_SOTA':
        segs_dir_root = os.path.join('/home/eliezer/Área de Trabalho/thesis/seg/sota', method)
    else:
        segs_dir_root = os.path.join('/home/eliezer/Área de Trabalho/thesis/seg/Postprocessing', method)

    segs_dir_melanoma = os.path.join(segs_dir_root, 'melanoma')
    segs_dir_notmelanoma = os.path.join(segs_dir_root, 'notmelanoma')
    
    if method.split('_')[-1] == 'DermIS'.lower():
        dataset_name = 'DermIS'
    elif method.split('_')[-1] == 'DermNet'.lower():
        dataset_name = 'DermNet'
    elif method.split('_')[-1] == 'DermQuest'.lower():
        dataset_name = 'DermQuest'
    else:
        print('There is a problem with the dataset name indicated by the method name!')
    
    gts_dir_root = os.path.join('/home/eliezer/Área de Trabalho/thesis', dataset_name)
    gts_dir_melanoma = os.path.join(gts_dir_root, 'melanoma')
    gts_dir_notmelanoma = os.path.join(gts_dir_root, 'notmelanoma')

    vec_acc = []
    vec_sens = []
    vec_spec = []
    vec_xor = []
    vec_iou = []
    vec_bf1 = []
    
    for segs_dir, gts_dir in zip([segs_dir_melanoma, segs_dir_notmelanoma], [gts_dir_melanoma, gts_dir_notmelanoma]):
    
        fnames_tmp = os.listdir(gts_dir)
        fnames = [fname for fname in fnames_tmp if fname.endswith('.png')]
        nimgs = len(fnames)
                  
        kernel = np.ones((3,3))
        for i in range(nimgs):
            
            print('Evaluating metrics for the {:s} image {:3d} of {:3d}...'.format(gts_dir.split('/')[-1], i + 1, nimgs))
            
            fname = fnames[i]       
            
            seg_tmp1 = cv2.imread(os.path.join(segs_dir, fname))
            seg_tmp2 = cv2.cvtColor(seg_tmp1, cv2.COLOR_BGR2GRAY)
            seg = cv2.normalize(seg_tmp2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            nr = seg.shape[0]
            nc = seg.shape[1]
            
            gt_tmp1 = cv2.imread(os.path.join(gts_dir, fname))
            gt_tmp2 = cv2.cvtColor(gt_tmp1, cv2.COLOR_BGR2GRAY)
            gt = cv2.normalize(gt_tmp2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                   
            acc = accuracy(seg, gt)
            sens = sensitivity(seg, gt)
            spec = specificity(seg, gt)
            xor = xor_error(seg, gt)
            iou = intersection_over_union(seg, gt)
            
            vec_acc.append(acc)
            vec_sens.append(sens)
            vec_spec.append(spec)
            vec_xor.append(xor)
            vec_iou.append(iou)
              
            seg_contour = abs(cv2.dilate(seg, kernel, iterations = 1) - seg)
            seg_contour[seg_contour > 0] = 1
            gt_contour = abs(cv2.dilate(gt, kernel, iterations = 1) - gt)
            
            theta = math.ceil(0.75*np.sqrt(nr**2 + nc**2)/100)
            bf1 = bf1score(seg_contour, gt_contour, theta)
            
            vec_bf1.append(bf1)
        
    vec_acc = np.array(vec_acc)
    vec_sens = np.array(vec_sens)
    vec_spec = np.array(vec_spec)
    vec_xor = np.array(vec_xor)
    vec_iou = np.array(vec_iou)
    vec_bf1 = np.array(vec_bf1)
    
    if dst_dir != 'Results_SOTA':
        vec_time = np.genfromtxt(os.path.join(segs_dir_root, 'elapsed_time.csv'), delimiter=',')
    
    f = open(fbase + '_results.txt', 'w+')
    
    print('acc mean = {:.9f}'.format(vec_acc.mean()), file=f)
    print('acc median = {:.9f}'.format(np.median(vec_acc)), file=f)
    print('acc std = {:.9f}\n'.format(vec_acc.std()), file=f)
      
    print('sens mean = {:.9f}'.format(vec_sens.mean()), file=f)
    print('sens median = {:.9f}'.format(np.median(vec_sens)), file=f)
    print('sens std = {:.9f}\n'.format(vec_sens.std()), file=f)
    
    print('spec mean = {:.9f}'.format(vec_spec.mean()), file=f)
    print('spec median = {:.9f}'.format(np.median(vec_spec)), file=f)
    print('spec std = {:.9f}\n'.format(vec_spec.std()), file=f)
       
    print('xor mean = {:.9f}'.format(vec_xor.mean()), file=f)
    print('xor median = {:.9f}'.format(np.median(vec_xor)), file=f)
    print('xor std = {:.9f}\n'.format(vec_xor.std()), file=f)
    
    print('iou mean = {:.9f}'.format(vec_iou.mean()), file=f)
    print('iou median = {:.9f}'.format(np.median(vec_iou)), file=f)
    print('iou std = {:.9f}\n'.format(vec_iou.std()), file=f)
        
    print('bf1 mean = {:.9f}'.format(vec_bf1.mean()), file=f)
    print('bf1 median = {:.9f}'.format(np.median(vec_bf1)), file=f)
    print('bf1 std = {:.9f}\n'.format(vec_bf1.std()), file=f)

    if dst_dir != 'Results_SOTA':

        print('time mean = {:.9f}'.format(vec_time.mean()), file=f)
        print('time median = {:.9f}'.format(np.median(vec_time)), file=f)
        print('time std = {:.9f}'.format(vec_time.std()), file=f)
            
    f.close()
    
    np.save(fbase + '_acc.npy', vec_acc)
    np.save(fbase + '_sens.npy', vec_sens)
    np.save(fbase + '_spec.npy', vec_spec)
    np.save(fbase + '_xor.npy', vec_xor) 
    np.save(fbase + '_iou.npy', vec_iou)   
    np.save(fbase + '_bf1.npy', vec_bf1)

    if dst_dir != 'Results_SOTA':
        np.save(os.path.join(curr_dir, dst_dir, method + '_time.npy'), vec_time)
