from src.interpretability_component.data_preprocessor import DataPreprocessor
import pandas as pd
import numpy as np


class DSprites(DataPreprocessor):
    """
    A class that prepares the dSprites dataset for evaluation of the Variational Autoencoder
    """

    data_str = "dsprites"

    def __init__(self, size=256, model_num=0):
        self.vae_str = "BetaVAE"
        self.size = size
        self.meta_data, self.z = self.get_meta_and_z_data(model_num)
        self.discrete_features = self.get_discrete_features()

    def get_meta_and_z_data(self, model_num=0):
        # get file from trained model of Locatello et al.
        df = pd.read_csv("./data/dsprites" + str(model_num) + ".csv").head(self.size)

        z = np.squeeze((df['mean'].map(lambda x: np.matrix(x)).tolist()))
        meta_data = df.drop(['Unnamed: 0', 'image', 'mean'], axis=1)
        return meta_data, z

    def get_discrete_features(self):
        return [True, True, True, True, True, False, False, False, False, False]
