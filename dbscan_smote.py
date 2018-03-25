from imblearn.over_sampling.base import BaseOverSampler
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class DBSCANSMOTE(BaseOverSampler):
    ''' Clusters the input data using DBScan and then oversamples using smote the defined clusters'''

    def __init__(self,
                 ratio="auto",
                 random_state=None,
                 normalize=True,
                 eps=0.5,
                 min_samples=5,
                 metric='euclidean',
                 metric_params=None,
                 algorithm='auto',
                 leaf_size=30,
                 p=None,
                 n_jobs=1,
                 dbscan_object=None):

        super(DBSCANSMOTE, self).__init__(ratio=ratio, random_state=random_state)
        self._normalize = normalize

        if dbscan_object is None:
            self._cluster_class = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric=metric,
                metric_params=metric_params,
                algorithm=algorithm,
                leaf_size=leaf_size,
                p=p,
                n_jobs=n_jobs
            )
        else:
            self._cluster_class = dbscan_object

    def _fit_cluster(self, X, y=None):
        ''' Normalizes the data into a [0,1] range,
        then applies DBSCAN on the input data'''

        if self._normalize:
            min_max = MinMaxScaler()
            X_ = min_max.fit_transform(X)
        else:
            X_ = X

        self._cluster_class.fit(X_, y)

    def _calculate_imb_ratio(self, X, y, cluster_labels=None):
        '''
        Calculate the imbalance ratio for each cluster.
        Right now it only allows for the binary case
        :param X:
        :param y: the vector with target observation
        :param minority_label:
        :param cluster_labels:
        :return:
        '''

        if cluster_labels is None:
            cluster_labels = self._cluster_class.labels_

        minority_label = self.minority_class

        unique_labels = np.unique(cluster_labels)

        # Remove label of obs identified as noise:
        unique_labels = unique_labels[unique_labels != -1]

        imbalance_ratio = {}

        for label in unique_labels:
            cluster_obs = y[cluster_labels == label]

            minority_obs = cluster_obs[cluster_obs == minority_label].size
            majority_obs = cluster_obs[cluster_obs != minority_label].size

            #To prevent division by zero errors
            if majority_obs == 0:
                majority_obs = 1

            imb_ratio = minority_obs / majority_obs

            imbalance_ratio[label] = imb_ratio

        return imbalance_ratio

    def _sample(self, X, y):
        self._fit_cluster(X, y)
        self.minority_class = self._find_minority_label(y)

        return self._calculate_imb_ratio(X, y)

    def _find_minority_label(self, y):
        (values, counts) = np.unique(y, return_counts=True)
        ind = np.argmin(counts)

        return values[ind]

    def get_labels(self):
        '''Returns the cluster labels of the fitted data'''

        return self._cluster_class.labels_
