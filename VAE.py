import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

from stacked_mnist import StackedMNISTData, DataMode
from autoencoder import Autoencoder_cnn


class VAE(nn.Module):

    def __init__(self, n_channels, criterion, latent_dim):
        super().__init__()
        self.criterion = criterion #nn.MSELoss()
        self.latent_dim = latent_dim
        self.n_channels = n_channels
        
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels,16,3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(), #(batch_size, 16, 14,14)
            nn.Conv2d(16,32,3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),  #(batch_size, 32, 7,7)
            nn.Conv2d(32,64,3,stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), # (batch_size, 64, 4, 4)
            nn.Conv2d(64,128,2,stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),  #(batch_size, 128, 3, 3). 
            nn.Flatten(),
            nn.Linear(128*3*3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.z_mean = nn.Linear(64,latent_dim)
        self.z_log_var = nn.Linear(64,latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128*3*3),
            nn.BatchNorm1d(128*3*3),
            nn.ReLU(),
            nn.Unflatten(1, (128, 3, 3)),
            nn.ConvTranspose2d(128, 64, 2, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, n_channels, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def sample(self, z_mean: torch.Tensor, z_log_var: torch.Tensor):
        eps = torch.randn(z_mean.size(0), z_mean.size(1)) # noise N(0,1) - from standard normal distribution
        z = z_mean + eps*torch.exp(z_log_var/2.) 
        return z

    def encode(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.sample(z_mean, z_log_var)
        return encoded
    
    def forward(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.sample(z_mean, z_log_var)
        decoded = self.decoder(encoded)
        return encoded, z_mean, z_log_var, decoded
    
    def fit(self, gen:StackedMNISTData, epochs, batch_size=64, visualize=False):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0005) 
        if visualize: #show original images
            outputs = []
        for epoch in range(epochs):
            batch_generator = gen.batch_generator(training=True, batch_size=batch_size)
            for img, _ in batch_generator:
                img = torch.tensor(img, dtype=torch.float32)
                channel_tensors = [img[:,:,:,i].unsqueeze(1) for i in range(self.n_channels)] # Extract each channel and add a new dimension for the channel
                img = torch.cat(channel_tensors, dim=1) # Concatenate the channels along the channel dimension
                encoded, z_mean, z_log_var, decoded = self(img)
                reconstruction_losses = [self.criterion(decoded[i], img[i]) for i in range(len(img))]
                #reconstruction_loss = self.criterion(decoded,img)
                #KL(Q(z|X) || P(z)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
                kl_div = -0.5*torch.sum(1+z_log_var-z_mean**2-torch.exp(z_log_var), axis=1)
                loss = torch.stack(reconstruction_losses) + 0.0015*kl_div #0.01
                mean_loss = loss.mean()
                
                #loss = reconstruction_loss + 0.005* kl_div_mean
            
                optimizer.zero_grad()
                mean_loss.backward()
                optimizer.step()
            
            print(f"Epoch:{epoch+1}, Reconstruction Loss:{torch.stack(reconstruction_losses).mean().item():.4f}.\
                   KL-div loss: {kl_div.mean().item():.4f}\
                   Loss: {mean_loss.item():.4f}")
            if visualize:
                outputs.append((epoch,img,decoded))

        # show images and reconstructions after training is finished
        if visualize:
            for k in range(0, epochs, 2):
                plt.figure(figsize=(9,2))
                plt.suptitle(f"Epoch {k+1}")
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

    
    def get_reconstruction_loss(self, img, N=10000):
        """
        Calculates the reconstruction losses for a given batch of  images.
        Args:
            img (numpy.ndarray): The input image to calculate the reconstruction loss for.
        Returns:
            float: An array of reconstruction loss values.
        """
        with torch.no_grad():
            img = torch.tensor(img, dtype=torch.float32) # add extra dimension in the beginning
            channel_tensors = [img[:,:,:,i].unsqueeze(1) for i in range(self.n_channels)] # Extract each channel and add a new dimension for the channel
            img = torch.cat(channel_tensors, dim=1) # Concatenate the channels along the channel dimension
            loss = torch.zeros(len(img))
            for _ in range(N):
                _, _, _, decoded = self(img)
                log_loss =  [self.criterion(decoded[i], img[i]).item() for i in range(len(img))]
                loss += np.exp(log_loss)/N
        return np.array(loss)
    

    def get_top_k_anomalious(self, imgs, k, show=False, N=10000):


        reconstruction_loss = self.get_reconstruction_loss(imgs, N)


        # Get the indices of the k images with the highest reconstruction loss
        top_k_indices = np.argsort(reconstruction_loss)[-k:]
        anomalies = imgs[top_k_indices]
        if show:
            n_rows = math.ceil(k / 10)  # calculate the number of rows
            plt.figure(figsize=(15, n_rows * 3))  # adjust the figure height 
            i = 0
            for item in anomalies:
                plt.subplot(n_rows, 10, i + 1)
                i = i+1
                plt.imshow(item.astype(np.float), vmin=0, vmax=1)  # specify cmap and scaling
                plt.axis('off')  # turn off axis
        return anomalies


    def decode(self, z):
        decoded = self.decoder(z)
        decoded = decoded.detach().numpy()
        return decoded

    def generate(self, n_samples):
        with torch.no_grad():
            z = torch.randn(n_samples,  self.latent_dim)
            samples = self.decode(z)
        return samples

    
    def save(self, path):
        torch.save(self.state_dict(), path)

    @staticmethod
    def load(path, n_channels, criterion, latent_dim):
        model = VAE(n_channels, criterion, latent_dim)
        model.load_state_dict(torch.load(path))
        return model
            

def main():
    #gen = StackedMNISTData(mode=DataMode.MONO_FLOAT_COMPLETE, default_batch_size=10)
        #model = VAE(n_channels=1, criterion=nn.BCELoss(), latent_dim=16)
    #model = VAE.load("autoencoders/vae16", n_channels=1, criterion=nn.BCELoss(),latent_dim=16)
    #model.fit(gen, epochs=200)
    #model.save("autoencoders/vae16")
    gen = StackedMNISTData(mode=DataMode.COLOR_BINARY_MISSING, default_batch_size=9)
    vae_color16_miss = VAE.load("autoencoders/vae_color16_miss", n_channels=3, criterion=nn.BCELoss(),latent_dim=16)
    #vae_color16_miss= VAE(n_channels=3, criterion=nn.BCELoss(), latent_dim=16)
    vae_color16_miss.fit(gen, epochs=200)
    vae_color16_miss.save("autoencoders/vae_color16_miss")

if __name__ == "__main__":
    main()