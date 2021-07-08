import numpy as np
import paddle.vision.transforms as transforms
from paddle.io import DataLoader
from paddle.vision import datasets
import paddle.nn as nn
import paddle

img_shape = (1, 28, 28)

class Generator(nn.Layer):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1D(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        self.model = nn.Sequential(
            *block(100, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = paddle.reshape(img, [img.shape[0], *img_shape])
        return img

G = Generator()
print(G.parameters())
print(G.children())