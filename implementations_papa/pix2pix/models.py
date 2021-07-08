import paddle.nn as nn
import paddle.nn.functional as F
import paddle


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           U-NET
##############################

class UNetDown(nn.Layer):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.8):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2D(in_size, out_size, 4, 2, 1, bias_attr=False)]
        if normalize:
            layers.append(nn.InstanceNorm2D(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Layer):
    def __init__(self, in_size, out_size, dropout=0.6):
        super(UNetUp, self).__init__()
        layers = [
            nn.Conv2DTranspose(in_size, out_size, 4, 2, 1, bias_attr=False),
            nn.InstanceNorm2D(out_size),
            nn.ReLU(),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = paddle.concat((x, skip_input), axis=1)
        return x


class GeneratorUNet(nn.Layer):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDOwn(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Pad2D([1,0,1,0]),
            nn.Conv2D(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up1(u1, d6)
        u3 = self.up1(u2, d5)
        u4 = self.up1(u3, d4)
        u5 = self.up1(u4, d3)
        u6 = self.up1(u5, d2)
        u7 = self.up1(u6, d1)

        return self.final(u7)

##############################
#        Discriminator
##############################

class Discriminator(nn.Layer):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2D(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2D(out_filters))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64 , normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Pad2D([1,0,1,0]),
            nn.Conv2D(512, 1, 4, padding=1, bias_attr=False)
        )
    
    def forward(self, img_A, img_B):
        img_input = paddle.concat((img_A, img_B), 1)
        return self.model(img_input)