import os
import pandas as pd
from clusteringCode.clusteringMethods import ClusteringWithDimensionalityReduction

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

try:
    data = pd.read_csv(data_filename, delimiter=";")
    features = data.columns.to_list()
    target = features.pop(0)

    components = ClusteringWithDimensionalityReduction(data, 2, target, features, 3)
    my_clusters = components.hierarchical_clustering()
    print(components.export_data_clusters(my_clusters))

except Exception as error:

    print("Unexpected error:", error)
