import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

from abc import ABC, abstractmethod


class DataPreprocessor(ABC):
    """
    A class that prepares data for evaluation of the Variational Autoencoder
    (using the Strategy Pattern)
    """

    @property
    @abstractmethod
    def data_str(self): pass

    def __init__(self, vae_str, size=256):
        self.vae_str = vae_str
        self.size = size
        self.model = self.get_model()
        self.meta_data, self.z = self.get_meta_and_z_data()
        self.discrete_features = self.get_discrete_features()

    def get_model(self):
        return tf.keras.models.load_model(self.vae_str)

    @abstractmethod
    def get_meta_and_z_data(self): pass

    @abstractmethod
    def get_discrete_features(self): pass  # for unknown: [False] * len(self.meta_data.columns)


class Zheng(DataPreprocessor):
    """
    A class that prepares the Zheng dataset for evaluation of the Variational Autoencoder
    """

    data_str = "zheng"

    def get_meta_and_z_data(self):
        data = list(tfds.load(self.data_str)["train"].take(self.size))
        df = pd.DataFrame(data)

        meta_data = df.drop(['ecg', 'quality'], axis=1).map(lambda d: d.numpy())

        ecg_data = df['ecg'].map(lambda ecg: ecg['I']).tolist()
        z_mean, z_log_var, z = self.model.encode(ecg_data)

        return meta_data, z

    def get_discrete_features(self):
        dic = {'age': False, 'atrial_rate': False, 'beat': True, 'gender': True, 'q_offset': False,
               'q_onset': False, 'qrs_count': False, 'qrs_duration': False, 'qt_corrected': False,
               'qt_interval': False, 'r_axis': False, 'rhythm': True, 't_axis': False, 't_offset': False,
               'ventricular_rate': False}
        return self.meta_data.columns.map(lambda factor: dic.get(factor)).tolist()


class PTB(DataPreprocessor):
    """
    A class that prepares the PTB dataset for evaluation of the Variational Autoencoder
    """

    data_str = "ptb"

    def get_meta_and_z_data(self):
        data = list(tfds.load(self.data_str)["train"].take(self.size))
        df = pd.DataFrame(data)

        meta_data = df.drop(['ecg', 'quality', 'diagnostic'], axis=1).map(lambda d: d.numpy())

        ecg_data = df['ecg'].map(lambda ecg: ecg['I']).tolist()
        z_mean, z_log_var, z = self.model.encode(ecg_data)

        return meta_data, z

    def get_discrete_features(self):
        dic = {'age': False, 'gender': True}
        return self.meta_data.columns.map(lambda factor: dic.get(factor)).tolist()


class Medalcare(DataPreprocessor):
    """
    A class that prepares the Medalcare dataset for evaluation of the Variational Autoencoder
    """

    data_str = "medalcare"

    def get_meta_and_z_data(self):
        data = list(tfds.load(self.data_str)["train"].take(self.size))
        df = pd.DataFrame(data)

        meta_data = df.drop(['ecg'], axis=1).map(lambda d: d.numpy())

        ecg_data = df['ecg'].map(lambda ecg: ecg['I']).tolist()
        z_mean, z_log_var, z = self.model.encode(ecg_data)

        return meta_data, z

    def get_discrete_features(self):
        return [False] * len(self.meta_data.columns)


class Icentia(DataPreprocessor):
    """
    A class that prepares the Icentia11k dataset for evaluation of the Variational Autoencoder
    """

    data_str = "icentia11k"

    def get_meta_and_z_data(self):
        print(tfds.load(self.data_str))
        data = list(tfds.load(self.data_str)["train"].take(self.size))
        df = pd.DataFrame(data)

        meta_data = df.drop(['ecg'], axis=1).map(lambda d: d.numpy())

        ecg_data = df['ecg'].map(lambda ecg: ecg['I']).tolist()
        z_mean, z_log_var, z = self.model.encode(ecg_data)

        return meta_data, z

    def get_discrete_features(self):
        return [False] * len(self.meta_data.columns)


class Synth(DataPreprocessor):
    """
    A class that prepares the Synth dataset for evaluation of the Variational Autoencoder
    """

    data_str = "synth"

    def get_meta_and_z_data(self):
        print(tfds.load(self.data_str))
        data = list(tfds.load(self.data_str)["train"].take(self.size))
        df = pd.DataFrame(data)

        meta_data = df.drop(['ecg'], axis=1).map(lambda d: d.numpy())

        ecg_data = df['ecg'].map(lambda ecg: ecg['I']).tolist()
        z_mean, z_log_var, z = self.model.encode(ecg_data)

        return meta_data, z

    def get_discrete_features(self):
        return [False] * len(self.meta_data.columns)
