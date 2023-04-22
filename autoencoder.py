import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

from stacked_mnist import StackedMNISTData, DataMode


class Autoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,12),
            nn.ReLU(),
            nn.Linear(12,3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,784),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x.permute(0,3,1,2))
        decoded = self.decoder(encoded)
        return decoded.permute(0,2,3,1)



class Autoencoder_cnn(nn.Module):

    def __init__(self, n_channels, criterion):
        super().__init__()
        self.criterion = criterion #nn.MSELoss()
        self.n_channels = n_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels,16,3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16,32,3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32,64,7)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64,32,7),
            nn.ReLU(),
            nn.ConvTranspose2d(32,16,3, stride=2,padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16,n_channels,3, stride=2,padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self,x):
        x = torch.tensor(x, dtype=torch.float32)
       # x = x[:,:,:,0].unsqueeze(1)
        channel_tensors = [x[:,:,:,i].unsqueeze(1) for i in range(self.n_channels)] # Extract each channel and add a new dimension for the channel
        x = torch.cat(channel_tensors, dim=1) 
        encoded = self.encoder(x)
        return encoded
    
    def decode(self, encoded):
        decoded = self.decoder(encoded)
        decoded= decoded.detach().numpy()
        return decoded

    def generate(self, n_samples):
        with torch.no_grad():
            z = torch.randn(n_samples, 64, 1, 1)
            samples = self.decode(z)
        return samples

    def fit(self, gen:StackedMNISTData, epochs, batch_size=64, visualize=False):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001) #TODO: generalize
        if visualize: #show original images
            outputs = []
        for epoch in range(epochs):
            batch_generator = gen.batch_generator(training=True, batch_size=batch_size)
            for img, _ in batch_generator:
                img = torch.tensor(img, dtype=torch.float32)
                #img = img[:,:,:,0].unsqueeze(1) # TODO: this works!
                channel_tensors = [img[:,:,:,i].unsqueeze(1) for i in range(self.n_channels)] # Extract each channel and add a new dimension for the channel
                img = torch.cat(channel_tensors, dim=1) # Concatenate the channels along the channel dimension
                recon = self(img)
                loss = self.criterion(recon,img)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            print(f"Epoch:{epoch+1}, Loss:{loss.item():.4f}")
            if visualize:
                outputs.append((epoch,img,recon))
        
        # show images and reconstructions after training is finished
        if visualize:
            for k in range(0, epochs, 2):
                plt.figure(figsize=(9,2))
                plt.suptitle(f"Epoch {k+1}")
               # plt.gray()
                imgs = outputs[k][1].detach().numpy()
                recon = outputs[k][2].detach().numpy()
                for i, item in enumerate(imgs):
                    if i >= 9: break
                    plt.subplot(2,9,i+1)
                    plt.imshow(np.transpose(item, (1, 2, 0)).astype(np.float)) 
                        
                for i, item in enumerate(recon):
                    if i >= 9: break
                    plt.subplot(2,9,9+i+1)
                    plt.imshow(np.transpose(item, (1, 2, 0)).astype(np.float))
        

def main():
    model = Autoencoder_cnn(n_channels=3,criterion=nn.BCELoss())
    gen = StackedMNISTData(mode=DataMode.COLOR_BINARY_COMPLETE, default_batch_size=64)
    model.fit(gen, epochs=2,visualize=True)
                

if __name__ == "__main__":
    main()