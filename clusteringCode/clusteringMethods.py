# Implementation using an abstract class.
# An abstract method has a declaration but does not have an implementation.
# When we want to provide a common interface for different implementations of a component, we use an abstract class.

from abc import ABC, abstractmethod  # This allows to create an abstract class.
from sklearn import decomposition as dec
import pandas as pd


class ClusteringWithDimensionalityReduction(ABC):

    def __init__(self, clustering_data, principal_components: int, entities_label: str, features_labels: list
                 , clustering_method: str = "k-means"):  # The variable target is a number in the index.

        self.data = clustering_data[features_labels]
        self.components = principal_components
        self.method = clustering_method  # Currently there are two methods: k-means and hierarchical clustering.
        self.target = entities_label
        self.labels = features_labels

    def principal_components_estimation(self):
        pca = dec.PCA(n_components=self.components, svd_solver="full")  # Method full as standard.
        principal_components = pd.DataFrame(data=pca.fit_transform(self.data), columns=["component_1", "component_2"])
        principal_components = principal_components.merge(data[self.target].to_frame()
                                                          , left_index=True, right_index=True)
        principal_components = principal_components.iloc[:, [2, 1, 0]]  # Reorders the dataframe columns.
        # It will work regardless of the number of features as the components are still 2.
        return principal_components

    @abstractmethod
    def clustering_components(self):  # To override with multiple clustering methods.
        pass


class KmeansClusteringPCA(ClusteringWithDimensionalityReduction):  # To complete.

    # Overriding the abstract method for the Kmeans clustering implementation.
    def clustering_components(self):
        pass


class HierarchicalClusteringPCA(ClusteringWithDimensionalityReduction):  # To complete.

    # Overriding the abstract method for hierarchical clustering implementation.
    def clustering_components(self):
        pass
