import pandas as pd
import matplotlib.pyplot as plt
from sklearn import decomposition as dec
from sklearn.cluster import KMeans
import os

# Declaring my working directory:
valid_working_directory: bool = False

while not valid_working_directory:

    try:
        working_directory: str = input('Enter your working directory:')
        os.chdir(working_directory)
        break  # If there are no issues with the directory, then proceed.

    except Exception as error:
        print(error)
        print('Enter a valid working directory')

        ask_question: str = input('Would you like to enter a valid working directory? Type Y/N:')

        if ask_question == 'Y':
            continue
        elif ask_question == 'N':
            exit()  # Stops running.
        else:
            print('Please enter a valid answer.')

data_filename: str = input('Enter your file name:')

# Importing data:
try:
    my_data: pd.DataFrame = pd.read_csv(data_filename, delimiter=';')
    print(my_data.head(10))
    number_of_features: int = len(my_data.columns)
    features_labels = my_data[1:number_of_features].columns.values.tolist()
    target_label = features_labels[0]  # Specifies the target label
    features_labels = features_labels[1:4]

    # Implementing a simple principal component analysis to reduce the features.

    pca = dec.PCA(n_components=2, svd_solver='full')
    principal_components = pca.fit_transform(my_data[features_labels])
    principal_dataframe: pd.DataFrame = pd.DataFrame(data=principal_components, columns=['component_1', 'component_2'])
    new_features = [target_label, 'component_1', 'component_2']

    # Defining the dataframe with the components before clustering

    labeled_principal_components: pd.DataFrame = pd.DataFrame(my_data.join(principal_dataframe))
    labeled_principal_components: pd.DataFrame = labeled_principal_components[new_features]

    # Estimating the clusters

    cluster = KMeans(n_clusters=3, max_iter=100)
    cluster.fit_predict(labeled_principal_components[new_features[1:2]])

    plt.figure(figsize=(10, 7))
    plt.scatter(labeled_principal_components['component_1'], labeled_principal_components['component_2'],
                c=cluster.labels_)
    plt.show()

    # Getting the final dataframe along with the clusters
    liquidity_groups: pd.DataFrame = pd.DataFrame(cluster.labels_, columns=['groups'])
    final_dataframe: pd.DataFrame = labeled_principal_components.join(liquidity_groups)
    print(final_dataframe.head(10))

except Exception as error:

    print('Value type is:', error)
    print('Please load a .CSV file only')
