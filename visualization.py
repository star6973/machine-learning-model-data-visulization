import logging
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS, Isomap

class Visualization(object):

    def __init__(self):
        logging.debug("loading {}".format(self.__class__.__name__))

    def _build_visualization(self):
        raise Exception("_build_visualization() method not implemented")

    def fit(self, X, y=None):
        raise Exception("fit() method not implemented")

    def transform(self, X):
        raise Exception("transform() method not implemented")

    def fit_transform(self, X, y=None):
        raise Exception("fit_transform() method not implemented")


class myPCA(Visualization):

    def __init__(self):
        self.visualization = self._build_visualization()

    def _build_visualization(self):
        return PCA(n_components=2, random_state=42)

    def fit(self, X, y=None):
        return self.visualization.fit(X, y)

    def transform(self, X):
        return self.visualization.transform(X)

    def fit_transform(self, X, y=None):
        return self.visualization.fit_transform(X, y)


class myMDS(Visualization):

    def __init__(self):
        self.visualization = self._build_visualization()

    def _build_visualization(self):
        return MDS(n_components=2, dissimilarity="precomputed", random_state=42)

    def fit(self, X, y=None):
        return self.visualization.fit(X, y)

    def fit_transform(self, X, y=None):
        return self.visualization.fit_transform(X, y)


class myTSNE(Visualization):

    def __init__(self):
        self.visualization = self._build_visualization()

    def _build_visualization(self):
        return TSNE(n_components=2, random_state=42)

    def fit(self, X, y=None):
        return self.visualization.fit(X, y)

    def fit_transform(self, X, y=None):
        return self.visualization.fit_transform(X, y)


class user_defined_dimension_reduction(Visualization):
    pass