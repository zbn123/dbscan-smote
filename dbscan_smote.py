from imblearn.over_sampling.base import  BaseOverSampler
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler


class DBSCANSMOTE(BaseOverSampler):
    ''' Clusters the input data using DBScan and then oversamples using smote the defined clusters
    PARAMETERS:
    - normalize = whether to normalize the data before clustering
    - *kwargs = arguments to the passed to the DBSCAN object'''

    def __init__(self,
                 ratio="auto",
                 random_state=None,
                 normalize = True,
                 eps=0.5,
                 min_samples=5,
                 metric= 'euclidean',
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

    def _fit_clusters(self, X, y=None):
        ''' Applies DBSCAN on the input data, return the cluster labels'''

        if self._normalize:
            min_max = MinMaxScaler()
            X_ = min_max.fit_transform(X)
        else:
            X_ = X

        self._cluster_class.fit(X_, y)

        return self._cluster_class.labels_

    def _sample(self, X, y):

        return(self._fit_clusters(X,y))
