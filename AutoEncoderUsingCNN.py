# Implementation of Auto Encoder using CNN and Transpose CNN

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# Data Set, Transformation and Data Loader
transform = transforms.ToTensor()
mnist_data = datasets.MNIST(root='/home/idrbt-06/Desktop/PY_TORCH/Auto_Encoder/Data', train=True, download=True, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset=mnist_data, batch_size=10, shuffle=True)


# Preparation of the Model of Auto Encoder.
# Here I have used CNN to prepare encoder.
# For preparation of decoder I have used Transpose CNN.
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=2,padding=1,bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=2,padding=1,bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=7,bias=False),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=7,bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32,out_channels=16,kernel_size=3,stride=2,padding=1,output_padding=1,bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16,out_channels=1,kernel_size=3,stride=2,padding=1,output_padding=1,bias=False),
            nn.Sigmoid()
        )
    
    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

model = Autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# Training of Auto Encoder
num_epochs = 200
original_and_reconstructed_images = []

for epoch in range(0, num_epochs, 1):
    for (images,labels) in data_loader:

        outputs = model(images)
        loss = criterion(outputs, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')
    original_and_reconstructed_images.append((epoch+1, images, outputs),)


# Display of Original and Reconstructed Images
# Here we are considering every 25 epoch
for k in range(0, num_epochs, 25):
    plt.figure(k)
    (a, b, c) = original_and_reconstructed_images[k]
    print('This is a=', a)
    print('This is the size of b=', b.shape)
    print('This is the size of c=', c.shape)
    original = b
    reconstructed = c
    original = original.detach().numpy()
    reconstructed = reconstructed.detach().numpy()
    for i in range(0, 10, 1):
        plt.subplot(2, 10, i+1)
        plt.imshow(original[i][0])
        plt.subplot(2, 10, 10+i+1)
        plt.imshow(reconstructed[i][0])
plt.show()

