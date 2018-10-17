#!/usr/bin/python
# coding: UTF-8

import pandas as pd
import numpy as np
import pickle
import datetime
import collections
import matplotlib.pyplot as plt
import tqdm
import json
from sklearn.cluster import DBSCAN 


class Clustering:
    """
    clustering KL-divergence and detect anomalies
    Ensembling results

    Attributes
    ----------
    epses : [float]
        epsilon candidates for searching proper DBSCAN epsilon paramters
    min_samples : int
        DBSCAN parameter (minimum clustering sample number)
        default ~= one week (8)
    max_classes : int
        maximum number of clusters
    max_anomalies : int
        max number of anomalies
    fallback_max_anomalies : int
        max number of anomalies
        (applied when parameter searching is failed with max_anomalies)
    """

    def __init__(self, epses=None, min_samples=None, max_classes=None, max_anomalies=None):
        if epses is None:
            self.epses = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
        else:
            self.epses = epses

        if min_samples is None:
            self.min_samples = 8
        else:
            self.min_samples = min_samples

        if max_classes is None:
            self.max_classes = 10
        else:
            self.max_classes = max_classes

        if max_anomalies is None:
            self.max_anomalies = 30
        else:
            self.max_anomalies = max_anomalies

        self.fallback_max_anomalies = 100

        self.anomalies = []


    def clustering(self, kld):
        """
        search the best epsilon value and return clustering results with best parameters
        
        Parameters
        ----------
        kld : np.array(float32)
            1-D KL-divergence array

        Returns
        -------
        labels : np.array(int)
            clustering label array

        Notes
        -----
        search the best eps value from self.epses.
        there are two constraints:
            1. the number of clusters should be under the self.max_classes
            2. the numebr of anomalies should be under the self.max_anomalies
        if there are no epses fulfill above conditions, try with self.fallback_max_anomalies
    
        """

        # make sure the input shape is 1-D array
        # if len(kld.shape) != 1:
        #     assert

        # scaling KLD
        if kld.max() == kld.min():
            scaled_data = np.array([[idx/len(kld), val/kld.max()] for idx,val in enumerate(kld)])
        else:
            scaled_data = np.array([[idx/len(kld), (val-kld.min())/(kld.max()-kld.min())] for idx,val in enumerate(kld)])

        # store clustering resutls for each eps
        class_cnt = []
        for eps in self.epses:
            dbs = DBSCAN(eps=eps, min_samples=self.min_samples)
            labels = dbs.fit_predict(scaled_data)
            anomaly_cnt = np.sum(labels == -1)
            class_cnt.append((labels.max(), anomaly_cnt))

        # search eps satisfying constraints
        best_eps = [[idx, (c, a)] for idx, (c, a) in enumerate(class_cnt) 
                                    if (self.max_classes > c) and (self.max_anomalies > a)]

        # if no results, try with self.fallback_max_anomalies
        if best_eps == []:
            best_eps = [[idx, (c, a)] for idx, (c, a) in enumerate(class_cnt) 
                                        if (self.max_classes > c) and (self.fallback_max_anomalies > a)]

        # clustering with the best eps
        eps = self.epses[best_eps[0][0]]
        dbs = DBSCAN(eps=eps, min_samples=self.min_samples)
        labels = dbs.fit_predict(scaled_data)

        return labels


    def ensemble(self, all_anomaly):
        """
        search the best epsilon value and return clustering results with best parameters
        
        Parameters
        ----------
        all_anomalies : np.array
            2-D array [[label_index, data]] 

        Returns
        -------
        all_maj_anomaly : list

        Notes
        -----
    
        """
        
        all_maj_anomaly = []
        for ind in range(120):
            anomaly_set = []
            for k,v in collections.Counter([j[1] for j in all_anomaly if j[0]==ind]).items():
                if v>4:
                    anomaly_set.append(k)
            all_maj_anomaly.append((ind,anomaly_set))
        return all_maj_anomaly


    def detect_anomaly(self, klds, num_labels=120, num_trials=10, start="20120101", end="20130331", verbose=False):
        """
        
        Parameters
        ----------
        klds : 
            2-D array [[label_index, data]] 

        Returns
        -------
        all_maj_anomaly : list
            converted to date index (20120101 - 201310331)

        Notes
        -----
    
        """

        if verbose:
            print("start clustering...")

        all_anomaly = []
        for i in tqdm.tqdm(range(num_labels), total=num_labels):
            for cnt in range(num_trials):
                data_length = int(klds[cnt][1].shape[0]/num_labels)
                data = klds[cnt][1].squeeze()[i*data_length:(i+1)*data_length]
                labels = self.clustering(data)

                all_anomaly.extend([(i, d_.date()) for d_ in pd.date_range(start, end)[np.where(labels==-1)[0]]])

        if verbose:
            print("ensembling...")

        results = self.ensemble(all_anomaly)

        return results

