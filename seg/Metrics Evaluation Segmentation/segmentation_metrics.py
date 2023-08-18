#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 10:38:47 2021

@author: eliezer
"""

import cv2
import numpy as np

def accuracy(seg_binary, gt_binary):
    
    tp = np.sum(seg_binary * gt_binary)
    fp = np.sum(seg_binary * (1-gt_binary))
    tn = np.sum((1-seg_binary) * (1-gt_binary))
    fn = np.sum((1-seg_binary) * gt_binary)
        
    return (tp + tn) / (tp + tn + fp + fn)
    
def sensitivity(seg_binary, gt_binary):
    
    tp = np.sum(seg_binary * gt_binary)
    #fp = np.sum(seg_binary * (1-gt_binary))
    #tn = np.sum((1-seg_binary) * (1-gt_binary))
    fn = np.sum((1-seg_binary) * gt_binary)
    
    return tp / (tp + fn)

def specificity(seg_binary, gt_binary):
    
    #tp = np.sum(seg_binary * gt_binary)
    fp = np.sum(seg_binary * (1-gt_binary))
    tn = np.sum((1-seg_binary) * (1-gt_binary))
    #fn = np.sum((1-seg_binary) * gt_binary)
        
    return tn / (tn + fp)

def xor_error(seg_binary, gt_binary):
    
    tp = np.sum(seg_binary * gt_binary)
    fp = np.sum(seg_binary * (1-gt_binary))
    #tn = np.sum((1-seg_binary) * (1-gt_binary))
    fn = np.sum((1-seg_binary) * gt_binary)
        
    return min((fp + fn) / (tp + fn), 1)

def intersection_over_union(seg_binary, gt_binary):
    
    tp = np.sum(seg_binary * gt_binary)
    fp = np.sum(seg_binary * (1-gt_binary))
    #tn = np.sum((1-seg_binary) * (1-gt_binary))
    fn = np.sum((1-seg_binary) * gt_binary)
    
    return tp / (tp + fp + fn)

def bf1score(seg_contour, gt_contour, theta):
        
    nr = seg_contour.shape[0]
    nc = seg_contour.shape[1]
    
    hits = 0
    total = 0;
    for r in range(nr):
        for c in range(nc):
            if seg_contour[r,c] == 1:
                mask = np.zeros(shape=(nr, nc))
                cv2.circle(mask, (c, r), theta, 1, -1)
                hits += min(sum(sum(mask*gt_contour)), 1)
                total += 1
    
    if total == 0:
        return 0
    else:
        return hits/total
    
