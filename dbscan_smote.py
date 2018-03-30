from imblearn.over_sampling.base import BaseOverSampler
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.spatial.distance import pdist
from imblearn.over_sampling import SMOTE
from warnings import filterwarnings, catch_warnings
from sklearn.exceptions import  DataConversionWarning

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
                 k_neighbors = 5,
                 n_jobs=1):

        super(DBSCANSMOTE, self).__init__(ratio=ratio, random_state=random_state)
        self._normalize = normalize
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.metric_params = metric_params
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.k_neighbors = k_neighbors
        self.n_jobs = n_jobs


    def _fit_cluster(self, X, y=None):
        ''' Normalizes the data into a [0,1] range,
        then applies DBSCAN on the input data'''

        if self._normalize:
            min_max = MinMaxScaler()
            # When the input data is int it will give a warning when converting to double
            with catch_warnings():
                filterwarnings("ignore", category=DataConversionWarning)
                X_ = min_max.fit_transform(X)
        else:
            X_ = X

        self._cluster_class.fit(X_, y)

    def _filter_clusters(self, y, cluster_labels=None, minority_label=None):
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

        unique_labels = np.unique(cluster_labels)

        # Remove label of observations identified as noise by DBSCAN:
        unique_labels = unique_labels[unique_labels != -1]

        filtered_clusters = []

        for label in unique_labels:
            cluster_obs = y[cluster_labels == label]

            minority_obs = cluster_obs[cluster_obs == minority_label].size
            majority_obs = cluster_obs[cluster_obs != minority_label].size

            imb_ratio = (majority_obs + 1) / (minority_obs + 1)

            if imb_ratio < 1:
                filtered_clusters.append(label)

        return filtered_clusters

    def _calculate_sampling_weights(self, X, y, filtered_clusters, cluster_labels=None, minority_class=None):

        if cluster_labels is None:
            cluster_labels = self.labels

        sparsity_factors = {}

        for cluster in filtered_clusters:

            # Observations belonging to current cluster and from the minority class
            obs = np.all([cluster_labels == cluster, y == minority_class], axis=0)
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
        self._set_DBSCAN()
        self._fit_cluster(X, y)

        self.labels = self._cluster_class.labels_

        X_resampled = X.copy()
        y_resampled = y.copy()

        with catch_warnings():
            filterwarnings("ignore", category=UserWarning, module="imblearn")

            for target_class in self.ratio_:
                print(self.ratio_)

                n_to_generate = self.ratio_[target_class]

                clusters_to_use = self._filter_clusters(y, self._cluster_class.labels_, target_class)

                if not clusters_to_use and n_to_generate > 0:
                    temp_dic = {target_class: n_to_generate}
                    X_cluster = X.copy()
                    y_cluster = y.copy()

                    n_obs = X_cluster.shape[0]

                    minority_obs = y_cluster[y_cluster == target_class]

                    if self.k_neighbors > minority_obs.size - 1:
                        k_neighbors = minority_obs.size - 1
                    else:
                        k_neighbors = self.k_neighbors

                    over_sampler = SMOTE(ratio=temp_dic, k_neighbors=k_neighbors)
                    over_sampler.fit(X_cluster, y_cluster)

                    X_cluster_resampled, y_cluster_resampled = over_sampler.sample(X_cluster, y_cluster)

                    # Save the newly generated samples only
                    X_cluster_resampled = X_cluster_resampled[n_obs:, :]
                    y_cluster_resampled = y_cluster_resampled[n_obs:, ]

                    # Add the newly generated samples to the data to be returned
                    X_resampled = np.concatenate((X_resampled, X_cluster_resampled))
                    y_resampled = np.concatenate((y_resampled, y_cluster_resampled))

                else:
                    sampling_weights = self._calculate_sampling_weights(X, y, clusters_to_use, self.labels, target_class)

                    for cluster in sampling_weights:
                        mask = self.labels == cluster
                        X_cluster = X[mask]
                        y_cluster = y[mask]

                        n_obs = mask.sum()

                        artificial_index = -1

                        # There needs to be at least two unique values of the target variable
                        if np.unique(y_cluster).size < 2:
                            art_x = np.zeros((1, X.shape[1]))
                            artificial_index = n_obs

                            artificial_y = np.unique(y)[np.unique(y) != target_class][0]

                            X_cluster = np.concatenate((X_cluster, art_x), axis=0)
                            y_cluster = np.concatenate((y_cluster, np.asarray(artificial_y).reshape((1,))), axis=0)

                        minority_obs = y_cluster[y_cluster == target_class]

                        n_new = n_to_generate * sampling_weights[cluster]

                        temp_dic = {target_class: int(round(n_new) + minority_obs.size)}

                        # We need to make sure that k_neighors is less than the number of observations in the cluster
                        if self.k_neighbors > minority_obs.size -1 :
                            k_neighbors = minority_obs.size - 1
                        else:
                            k_neighbors = self.k_neighbors

                        over_sampler = SMOTE(ratio=temp_dic, k_neighbors=k_neighbors)
                        over_sampler.fit(X_cluster, y_cluster)

                        X_cluster_resampled, y_cluster_resampled = over_sampler.sample(X_cluster, y_cluster)

                        # If there was a observation added, then it is necessary to remove it now
                        if artificial_index > 0:
                            X_cluster_resampled = np.delete(X_cluster_resampled, artificial_index, axis=0)
                            y_cluster_resampled = np.delete(y_cluster_resampled, artificial_index)

                        # Save the newly generated samples only
                        X_cluster_resampled = X_cluster_resampled[n_obs:, :]
                        y_cluster_resampled = y_cluster_resampled[n_obs:, ]

                        # Add the newly generated samples to the data to be returned
                        X_resampled = np.concatenate((X_resampled, X_cluster_resampled))
                        y_resampled = np.concatenate((y_resampled, y_cluster_resampled))

        return X_resampled, y_resampled

    def _find_minority_label(self, y):
        (values, counts) = np.unique(y, return_counts=True)
        ind = np.argmin(counts)

        return values[ind]

    def get_labels(self):
        '''Returns the cluster labels of the fitted data'''

        return self._cluster_class.labels_

    def _set_DBSCAN(self):
        self._cluster_class = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric,
            metric_params=self.metric_params,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            p=self.p,
            n_jobs=self.n_jobs)
