import pandas as pd
import numpy as np
from numpy.linalg import *
import scipy as sp
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.decomposition import *
from sklearn.metrics import *
from sklearn.neighbors import *
from collections import *
from pyclust import *
from sklearn.manifold import *
from sklearn.cluster import KMeans

class EKmeans():
    def __init__(self,X,D,method='jaccard',n_centers=2,max_iters=1000,tol=1e-20,verbose=0):
        self.n_centers = n_centers
        self.max_iters = max_iters
        self.D = D
        self.method = method
        self.X = X
        self.clusters = np.array([-1 for i in range(X.shape[0])])
        self.centers = self.X[np.random.permutation(X.shape[0])[:self.n_centers]]
        self.pre_centers = []
        self.debug = []
        self.lr = 0.34
        self.tol = tol
        self.verbose = verbose
        return
    
    def metrics_compactness(self):
        s = 0.0
        for c in range(self.n_centers):
            idx_cluster = np.where(self.clusters==c)[0]
            for a in self.X[ idx_cluster ]:
                dst = compute_distance(a=a,b=self.centers[c],D=self.D,metric=self.method)
                s += dst*1.0/len(idx_cluster)
        return s
                
    def __update_centers(self,smooth=True):
        self.pre_centers = np.array(self.centers)
        for c in range(self.n_centers):
            temp = self.X[ np.where(self.clusters==c)[0] ]
            if len(temp):
                self.centers[c] = np.array([ np.mean( temp, 0 ) ])
#             print(self.centers,self.X[ np.where(self.clusters==c)[0] ]) ### debug
        if smooth:
            self.centers = (self.centers-self.pre_centers)*self.lr + self.pre_centers
        return
    
    def __log(self,x):
        if self.verbose:
            print(x)
        return
    
    def fit(self):
        for kiter in range(self.max_iters):
            self.__log('iteration %d'%kiter)
            for i in range(self.X.shape[0]):
                d_min_dist = 1e10
                record_node = -1
                for idx,c in zip(range(len(self.centers)),self.centers):
                    dst = compute_distance(a=self.X[i],b=c,D=self.D,metric=self.method)
                    if dst<d_min_dist:
                        d_min_dist = dst
                        record_node = idx
                self.clusters[i] = record_node
            self.__update_centers()
            move_dist = np.mean(norm(self.pre_centers-self.centers,axis=1))
            self.debug.append(move_dist)
            if (kiter - kiter//20 *20 ==1) and move_dist<self.tol:
                break
            if kiter==10:
#                 raise Exception('stop')
                pass
        return