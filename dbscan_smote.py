from imblearn.over_sampling.base import BaseOverSampler
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.spatial.distance import pdist


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
                n_jobs=n_jobs)
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

    def _filter_clusters(self, X, y, cluster_labels=None):
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
            cluster_labels = self.labels

        minority_label = self.minority_class

        unique_labels = np.unique(cluster_labels)

        # Remove label of observations identified as noise by DBSCAN:
        unique_labels = unique_labels[unique_labels != -1]

        filtered_clusters = []

        for label in unique_labels:
            cluster_obs = y[cluster_labels == label]

            minority_obs = cluster_obs[cluster_obs == minority_label].size
            majority_obs = cluster_obs[cluster_obs != minority_label].size

            imb_ratio =  (majority_obs + 1) / (minority_obs + 1)

            if imb_ratio < 1:
                filtered_clusters.append(label)

        return filtered_clusters

    def _calculate_sampling_weights(self, X, y, filtered_clusters, cluster_labels = None):

        if cluster_labels is None:
            cluster_labels = self.labels

        sparsity_factors = {}

        for cluster in filtered_clusters:

            # Observations belonging to current cluster and from the minority class
            obs = np.all([cluster_labels == cluster, y == self.minority_class], axis=0)
            n_obs = obs.sum()

            cluster_X = X[obs]

            # pdist calculates the condensed distance matrix, which is the upper triangle of the regular distance matrix
            # We can just calculate the mean over that vector, considering that that d(a,b) only exists once ( d(b,a) and
            # d(a,a) is not present).
            distance = pdist(cluster_X, 'euclidean')

            average_minority_distance = np.mean(distance)

            density_factor = average_minority_distance / (n_obs**2)

            sparsity_factor = 1 / density_factor

            sparsity_factors[cluster] = sparsity_factor

        sparsity_sum = sum(sparsity_factors.values())


        sampling_weights = {}

        for cluster in sparsity_factors:
            sampling_weights[cluster] = sparsity_factors[cluster] / sparsity_sum

        return sampling_weights




    def _sample(self, X, y):

        # Create the clusters and set the labels
        self._fit_cluster(X, y)

        self.labels = self._cluster_class.labels_

        # Finds the minority class
        # To be re written for the multiclass case
        self.minority_class = self._find_minority_label(y)

        # Filters the clusters using the method in K Means SMOTE
        clusters_to_use = self._filter_clusters(X, y, self._cluster_class.labels_)

        # Calculates the sampling weights
        sampling_weights = self._calculate_sampling_weights(X, y, clusters_to_use)

        n_to_generate = self.ratio_[self.minority_class]

        for cluster in sampling_weights:
            mask = self.labels == cluster
            X_c = X[mask]
            y_c = y[mask]

            n_obs = mask.sum()

            n_new = n_to_generate * sampling_weights[cluster]

            print("Cluster: {} has {} obs and a sampling weight of {}, so {} samples should be added". format(cluster, n_obs, sampling_weights[cluster], n_new))

            temp_dic = self.ratio_.copy()

            temp_dic[self.minority_class] = round(n_new)

            print(temp_dic)

        return sampling_weights


    def _find_minority_label(self, y):
        (values, counts) = np.unique(y, return_counts=True)
        ind = np.argmin(counts)

        return values[ind]

    def get_labels(self):
        '''Returns the cluster labels of the fitted data'''

        return self._cluster_class.labels_
