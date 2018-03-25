from imblearn.over_sampling.base import  BaseOverSampler
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler


class DBSCANSMOTE(BaseOverSampler):
    ''' Clusters the input data using DBScan and then oversamples using smote the defined clusters
    PARAMETERS:
    - normalize = whether to normalize the data before clustering
    - *kwargs = arguments to the passed to the DBSCAN object'''

    _cluster_class = DBSCAN()
    _normalize = True

    def __init__(self, ratio="auto", random_state=None, normalize = True,  *kwargs):

        super(DBSCANSMOTE, self).__init__(ratio=ratio, random_state=random_state)
        self._normalize = normalize
        self._cluster_class = DBSCAN(*kwargs)

    def _fit_clusters(self, X, y=None):

        if self._normalize:
            min_max = MinMaxScaler()
            X_ = min_max.fit_transform(X)
        else:
            X_ = X

        self._cluster_class.fit(X_, y)

        return self._cluster_class.labels_

