#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import scipy.io
from sklearn.metrics import accuracy_score as accuracy
from vlfeat._libvl import ffi, lib
from vlfeat.wrapper import gmm

import sys

# DDF = Dictionary of Deep Features
class DDF:

    def __init__(self, lamb, vphi, verbose = False):

        self.lamb = lamb
        self.vphi = vphi
        self.verbose = verbose

    def fit(self, Ytrain, Ttrain):

        vphi = self.vphi
        verbose = self.verbose

        if verbose == True:
            print('=> Fitting:')

        c = len(np.unique(Ttrain))
        self.c = c

        for i in range(c): 
           
            if verbose == True:
                print('. building the dictionary of deep features for class {:2d} of {:2d}...'.format(i + 1, c), end='')

            Fi = Ytrain[:,Ttrain==i]
            dict_size = round(vphi*Fi.shape[1])
            
            labels_i = i*np.ones((dict_size))

            means, _, _, _, _ = gmm(Fi, dict_size, gmm_initialization = 'random')    
            Di = means.T

            if i == 0:
                labels = labels_i
                D = Di
            else:
                labels = np.concatenate((labels, labels_i))
                D = np.hstack((D, Di))

            if verbose == True:
                print('Done!')

        self.D = D
        self.labels = labels

    def predict(self, Y):

        lamb = self.lamb
        verbose = self.verbose
        D = self.D
        labels = self.labels

        if verbose == True:
            print('=> Predicting:')

        K = D.shape[1]    
        n = Y.shape[1]
        X0 = np.zeros((K,n)) 
       
        if verbose == True:
            print('. sparsely representing data in the built dictionary of deep features...', end='')

        eig_vals, eig_vecs = np.linalg.eig(np.dot(np.transpose(D),D))
        L = np.max(eig_vals)
        
        X_old = X0
        mu_old = 1
        XX_old = X_old
        
        grad_step = 1/L
        lamb_x_grad_step = lamb*grad_step
        
        for t in range(100):
            Dt = np.transpose(D)
            Dt_x_D = np.dot(Dt,D)    
            grad = np.dot(Dt_x_D,XX_old)-np.dot(Dt,Y)
            Z = XX_old - grad_step*grad
            X_new =  np.maximum(Z-lamb_x_grad_step,X0) + np.minimum(Z+lamb_x_grad_step,X0) 
            mu_new = 0.5*(1+np.sqrt(1+4*mu_old**2))
            XX_new = X_new + ((mu_old-1)/(mu_new))*(X_new-X_old)
            error = np.linalg.norm(X_new-X_old,1)/K
            if error < 1e-8:
                break;
            X_old = X_new
            mu_old = mu_new
            XX_old = XX_new
      
        if verbose == True:
            print('Done!')

        X = X_new
        self.X = X
    
        c = self.c

        if verbose == True:
            print('. verifying the class whose dictionary provides the lowest reconstruction error...', end='')

        E = np.zeros((c,Y.shape[1]))
        for i in range(c):
            Di = D[:, labels == i] 
            Xi = X[labels == i, :]
            recov = np.dot(Di, Xi)
            E[i, :] = np.sum((Y-recov)**2, axis=0)

        pred = np.argmin(E, axis=0)

        if verbose == True:
            print('Done!')

        return pred

    def predict_proba(self, Y):

        lamb = self.lamb
        verbose = self.verbose
        D = self.D
        labels = self.labels

        if verbose == True:
            print('=> Predicting:')

        K = D.shape[1]    
        n = Y.shape[1]
        X0 = np.zeros((K,n)) 
       
        if verbose == True:
            print('. sparsely representing data in the built dictionary of deep features...', end='')

        eig_vals, eig_vecs = np.linalg.eig(np.dot(np.transpose(D),D))
        L = np.max(eig_vals)
        
        X_old = X0
        mu_old = 1
        XX_old = X_old
        
        grad_step = 1/L
        lamb_x_grad_step = lamb*grad_step
        
        for t in range(100):
            Dt = np.transpose(D)
            Dt_x_D = np.dot(Dt,D)    
            grad = np.dot(Dt_x_D,XX_old)-np.dot(Dt,Y)
            Z = XX_old - grad_step*grad
            X_new =  np.maximum(Z-lamb_x_grad_step,X0) + np.minimum(Z+lamb_x_grad_step,X0) 
            mu_new = 0.5*(1+np.sqrt(1+4*mu_old**2))
            XX_new = X_new + ((mu_old-1)/(mu_new))*(X_new-X_old)
            error = np.linalg.norm(X_new-X_old,1)/K
            if error < 1e-8:
                break;
            X_old = X_new
            mu_old = mu_new
            XX_old = XX_new
      
        if verbose == True:
            print('Done!')

        X = X_new
        self.X = X
    
        c = self.c

        if verbose == True:
            print('. verifying the class whose dictionary provides the lowest reconstruction error...', end='')

        E = np.zeros((c,Y.shape[1]))
        for i in range(c):
            Di = D[:, labels == i] 
            Xi = X[labels == i, :]
            recov = np.dot(Di, Xi)
            E[i, :] = np.sum((Y-recov)**2, axis=0)

        pred_proba = 1 - E/np.sum(E, axis=0)

        if verbose == True:
            print('Done!')

        return pred_proba 
