import paddle
from paddle.io import DataLoader
from paddle.vision import datasets
from paddle.vision import transforms

dataset = datasets.MNIST(
        mode='train',
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(28), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    )

dataloader = DataLoader(
    dataset=dataset,
    batch_size=8,
    shuffle=True,
)
print(type(dataset))
print(type(dataloader))
for i, (img, _) in enumerate(dataloader()):
    print(img.shape)