import torch.nn as nn

from pythae.models.base.base_utils import ModelOutput

from pythae.models.nn import BaseEncoder, BaseDecoder

class Encoder(BaseEncoder):
    def __init__(self, args):
        BaseEncoder.__init__(self)

        self.input_dim = (1, 500)
        self.latent_dim = args.latent_dim
        self.n_channels = 1

        self.encode_1 = nn.Sequential(
            nn.Conv1d(in_channels=self.n_channels, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU()
        )

        self.encode_2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU()
        )

        self.encode_3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=4, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU()
        )

        self.encode_4 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU()
        )

        self.embedding = nn.Linear(128 * 16, args.latent_dim)
        self.log_var = nn.Linear(128 * 16, args.latent_dim)

    def forward(self, x):

        x = self.encode_1(x)
        x = self.encode_2(x)
        x = self.encode_3(x)
        x = self.encode_4(x)

        x = x.reshape(x.shape[0], -1)

        output = ModelOutput(
            embedding=self.embedding(x),
            log_covariance=self.log_var(x)
        )
        return output


class Decoder(BaseDecoder):
    def __init__(self, args):
        BaseDecoder.__init__(self)

        self.input_dim = (1, 500)
        self.latent_dim = args.latent_dim
        self.n_channels = 1

        self.linear = nn.Linear(args.latent_dim, 128*16)

        self.decode_1 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=4, stride=4, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU()
        )

        self.decode_2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=5, stride=4),
            nn.BatchNorm1d(32),
            nn.LeakyReLU()
        )

        self.decode_3 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=32, out_channels=self.n_channels, kernel_size=4, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):

        x = self.linear(x)
        x = x.view(-1, 128, 16)
        x = self.decode_1(x)
        x = self.decode_2(x)
        x = self.decode_3(x)

        output = ModelOutput(reconstruction=x)

        return output
