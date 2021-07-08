import argparse
import os
import numpy as np
import math
import cv2

import paddle.vision.transforms as transforms
from paddle.io import DataLoader
from paddle.vision import datasets
import paddle.nn as nn
import paddle

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=200, help="interval betwen image samples")
opt = parser.parse_args()
# print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

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
            *block(opt.latent_dim, 128, normalize=False),
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


class Discriminator(nn.Layer):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = paddle.reshape(img, [img.shape[0], -1])
        validity = self.model(img_flat)
        return validity

G = paddle.Model(Generator())
D = paddle.Model(Discriminator())

G.summary((-1, opt.latent_dim))
D.summary((-1, *img_shape))


# Loss function
adversarial_loss = paddle.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()
# print("-1")
# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)

dataset = datasets.MNIST(
        mode='train',
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    )
# print(len(dataset))
dataloader = DataLoader(
    dataset=dataset,
    batch_size=opt.batch_size,
    shuffle=True,
)
# print("0")
# Optimizers
optimizer_G = paddle.optimizer.Adam(parameters=generator.parameters(), learning_rate=opt.lr, beta1=opt.b1, beta2=opt.b2)
optimizer_D = paddle.optimizer.Adam(parameters=discriminator.parameters(), learning_rate=opt.lr, beta1=opt.b1, beta2=opt.b2)

# ----------
#  Training
# ----------
# print("0.1")

# for i, (img, _ ) in enumerate(dataloader()):
    # print(img.shape)


for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader()):
        # print("1")
        # Adversarial ground truths
        valid = paddle.ones(shape=[imgs.shape[0],1])
        fake = paddle.zeros(shape=[imgs.shape[0],1])
        # print("2")
        # Configure input
        real_imgs = imgs
        # print("3")
        # -----------------
        #  Train Generator
        # -----------------
        # print("4")
        # Sample noise as generator input
        z = paddle.to_tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)), dtype='float32')

        # Generate a batch of images
        gen_imgs = generator(z)
        # print("5")
        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        # print("6")
        g_loss.backward()
        optimizer_G.step()
        optimizer_G.clear_grad()
        # print("7")
        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()
        optimizer_D.clear_grad()
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.numpy(), g_loss.numpy())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            #save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
            # print("tensor", gen_imgs.shape)
            imga = []
            # print("numpy", gen_imgs.numpy().shape)
            # print("numpy 25", gen_imgs.numpy()[:25].shape)
            for i, f in enumerate(gen_imgs.numpy()[:25]):
                # print("f", f.shape)
                f = f.transpose((1,2,0))
                if i % 5 == 0:
                    if i != 0:
                        timg = np.concatenate(imgl, axis=1)
                        imga.append(timg)
                    imgl = []
                imgl.append(f)
            timg = np.concatenate(imgl, axis=1)
            imga.append(timg)
            timg = np.concatenate(imga, axis=0)
            timg *= 255
            cv2.imwrite("images/%d.png" % batches_done, timg.astype(np.uint8))