# Class implementation of the both clustering methods.

from sklearn import decomposition as dec
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import pandas as pd


class ClusteringWithDimensionalityReduction:

    def __init__(self, clustering_data, principal_components: int, entities_label: str, features_labels: list,
                 number_of_clusters: int, clustering_method: str = "k-means"):

        self.data = clustering_data[features_labels]  # Leave required features
        self.components = principal_components
        self.method = clustering_method  # Currently there are two methods: k-means and hierarchical clustering.
        self.target = entities_label
        self.labels = features_labels
        self.clusters = number_of_clusters
        self.data_target = clustering_data[self.target]  # Pops the labels for the different entities.

    def principal_components_estimation(self):
        pca = dec.PCA(n_components=self.components, svd_solver="full")  # Method full as standard.
        principal_components = pd.DataFrame(data=pca.fit_transform(self.data), columns=["component_1", "component_2"])
        principal_components = principal_components.merge(self.data_target.to_frame()
                                                          , left_index=True, right_index=True)
        principal_components = principal_components.iloc[:, [2, 1, 0]]  # Reorders the dataframe columns.
        # It will work regardless of the number of features as the components are still 2.
        return principal_components

    def kmeans_clustering(self):
        pca_labels = ["component_1", "component_2"]
        cluster = KMeans(n_clusters=self.clusters, max_iter=100)  # 100 iterations by default.
        cluster.fit_predict(ClusteringWithDimensionalityReduction.principal_components_estimation(self)[pca_labels])
        return cluster

    def hierarchical_clustering(self):
        pca_labels = ["component_1", "component_2"]
        cluster = AgglomerativeClustering(n_clusters=self.clusters, affinity='euclidean', linkage='ward')
        cluster.fit_predict(ClusteringWithDimensionalityReduction.principal_components_estimation(self)[pca_labels])
        return cluster

    def export_data_clusters(self, clustering_method):
        new_data = pd.DataFrame(clustering_method.labels_, columns=['categories'])
        final_data = new_data.join(self.data)
        return final_data
