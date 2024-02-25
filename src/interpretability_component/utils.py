import pandas as pd
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds


def visualize_mi_matrix(matrix, columns, save_pdf=False, filename="plot"):
    """
    Visualize a matrix with columns and save it as pdf as filename

    Args:
        matrix (np.ndarray): scoring matrix
        columns ([str]): array with column names of the scoring matrix
        save_pdf (bool): save the visualization as pdf?
        filename (str): file path and name

    """
    mi_df = pd.DataFrame(list(matrix), columns=columns)

    columns = mi_df.columns.values.tolist()

    plt.imshow(mi_df, cmap='Purples')

    # Adding details to the plot
    plt.title("Scores between Factors and Latent Dimensions")
    plt.xticks(range(0, len(columns)), columns, rotation='vertical')

    # Adding a color bar to the plot
    plt.colorbar()

    # Save image as pdf
    if save_pdf:
        plt.savefig(filename + ".pdf", bbox_inches='tight')
        print("Figure saved as " + filename + ".pdf")

    # Displaying the plot
    plt.show()


def visualize_corr(corr, save_pdf=False, filename="plot"):
    """
    Visualize a matrix and save it as pdf as filename

    Args:
        corr (np.ndarray): correlation matrix
        save_pdf (bool): save the visualization as pdf?
        filename (str): file path and name

    """
    columns = corr.columns.values.tolist()

    plt.imshow(corr.map(abs), cmap='Purples')

    # Adding details to the plot
    plt.title("Absolute Correlation between Factors")
    plt.yticks(range(0, len(columns)), columns)
    plt.xticks(range(0, len(columns)), columns, rotation='vertical')
    plt.tick_params('y', labelright=True, labelleft=False, right=True, left=False)

    # Adding a color bar to the plot
    plt.colorbar(location='left')

    # Save image as pdf
    if save_pdf:
        plt.savefig(filename + ".pdf", bbox_inches='tight')
        print("Figure saved as " + filename + ".pdf")

    # Displaying the plot
    plt.show()


def save_dataset_as_csv(data_str, size=256, directory="data/"):
    """
    save the ECG dataset as csv for export in Visualization Tool

    Args:
        data_str (str): name of tfds dataset with ECG data
        size (int): number of desired samples
        directory (str): directory for the saved csv

    """
    data = list(tfds.load(data_str)["train"].take(size))
    df = pd.DataFrame(data)

    df_res = df.drop(['ecg', 'quality'], axis=1).map(lambda d: d.numpy())
    df_res['ecg'] = df['ecg'].map(lambda ecg: ecg['I'].numpy()).tolist()

    df_res.to_csv(directory + data_str + "_data.csv")
    print("saved " + data_str + "_data.csv")


def save_discrete_features_as_csv(discrete_features, features_names, data_str, directory="data/"):
    """
    save the discrete feature array of a ECG dataset as csv for export in Visualization Tool

    Args:
        discrete_features ([bool]): list of True/False indicating if each feature is discrete
        features_names ([str]): list of features' names
        data_str (str): name of tfds dataset with ECG data
        directory (str): directory for the saved csv

    """
    df = pd.DataFrame(features_names, columns=[["features_names"]])
    df["is_discrete"] = discrete_features

    df.to_csv(directory + data_str + "_discrete_features.csv")
    print("saved " + data_str + "_discrete_features.csv")


def save_dis_scores_as_csv(scores, model_num, directory="tests/dsprites_tests/results/"):
    """
    save the achieved disentanglement score as csv

    Args:
        scores (dict): dictionary of metrics with achieved scores
        model_num (int): number of the model as part of the file path
        directory (str): directory for the saved csv

    """
    df = pd.DataFrame()
    df["metrics"] = scores.keys()
    df["values"] = scores.values()
    df.to_csv(directory + str(model_num) + "_disentanglement_scores.csv")
    print("saved " + str(model_num) + "_disentanglement_scores.csv")
