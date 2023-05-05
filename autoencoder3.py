import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

from stacked_mnist import StackedMNISTData, DataMode


class Autoencoder_cnn3(nn.Module):
    def __init__(self, n_channels, criterion, latent_dim):
        super().__init__()
        self.criterion = criterion 
        self.latent_dim = latent_dim
        self.n_channels = n_channels
        
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1, padding=0),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(128*3*3, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, latent_dim),
                nn.BatchNorm1d(latent_dim)
            )
            for _ in range(n_channels)
        ])

        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 128*3*3),
                nn.BatchNorm1d(128*3*3),
                nn.ReLU(),
                nn.Unflatten(1, (128, 3, 3)),
                nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=1, padding=0),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=0),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.Sigmoid()
            )
            for _ in range(n_channels)
        ])

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        x = [x[:,:,:,i].unsqueeze(1) for i in range(self.n_channels)] # Extract each channel and add a new dimension for the channel
        #x = torch.cat(channel_tensors, dim=1) 
        encoded = []
        for i in range(self.n_channels):
            encoded.append(self.encoder[i](x[i]))
        encoded = torch.stack(encoded, dim=1)
        decoded = []
        for i in range(self.n_channels):
            decoded.append(self.decoder[i](encoded[:,i,:]))
        decoded = torch.cat(decoded, dim=1)
        return decoded
    
    def fit(self, gen:StackedMNISTData, epochs, batch_size=64, visualize=False):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0005) 
        if visualize: #show original images
            outputs = []
        for epoch in range(epochs):
            batch_generator = gen.batch_generator(training=True, batch_size=batch_size)
            for img, _ in batch_generator:

                # Split input into separate channels
                img_t = torch.tensor(img, dtype=torch.float32)
                img_t = [img_t[:,:,:,i].unsqueeze(1) for i in range(self.n_channels)] 
                
                optimizer.zero_grad()
            
                # Encode each channel separately
                encoded = []
                for i in range(self.n_channels):
                    encoded.append(self.encoder[i](img_t[i]))
                encoded = torch.stack(encoded, dim=1)
                
                # Decode each channel separately
                decoded = []
                for i in range(self.n_channels):
                    decoded.append(self.decoder[i](encoded[:,i,:]))
                decoded = torch.cat(decoded, dim=1)  # Concatenate decoded channels
                
                # Calculate loss and optimize
                #loss = self.criterion(decoded, img)
                loss = torch.mean(torch.stack([self.criterion(decoded[:,i:i+1,:,:], img_t[i]) for i in range(self.n_channels)]))
                loss.backward()
                optimizer.step()
            
            print(f"Epoch:{epoch+1}, Loss:{loss.item():.4f}")
            if visualize:
                outputs.append((epoch,img,decoded))

         # show images and reconstructions after training is finished
        if visualize:
            for k in range(0, epochs, 2):
                plt.figure(figsize=(9,2))
                plt.suptitle(f"Epoch {k+1}")
               # plt.gray()
                imgs = outputs[k][1]
                recon = outputs[k][2].detach().numpy()
                for i, item in enumerate(imgs):
                    if i >= 9: break
                    plt.subplot(2,9,i+1)
                    plt.imshow(item.astype(np.float))                       
                        
                for i, item in enumerate(recon):
                    if i >= 9: break
                    plt.subplot(2,9,9+i+1)
                    plt.imshow(np.transpose(item, (1, 2, 0)).astype(np.float))


    def decode(self, z):
        decoded = []
        for i in range(self.n_channels):
            decoded.append(self.decoder[i](z[:,i,:]))
        decoded = torch.cat(decoded, dim=1)  # Concatenate decoded channels
        return decoded

    def generate(self, n_samples):
        with torch.no_grad():
            z = torch.randn(n_samples, self.n_channels, self.latent_dim)
            samples = self.decode(z)
        return samples.detach().numpy()
    
    def get_reconstruction_loss(self, img):
        """
        Calculates the reconstruction losses for a given batch of  images.
        Args:
            img (numpy.ndarray): The input image to calculate the reconstruction loss for.
        Returns:
            float: An array of reconstruction loss values.
        """
        with torch.no_grad():
            decoded = self(img)
            img_t = torch.tensor(img, dtype=torch.float32)
            img_t = [img_t[:,:,:,i].unsqueeze(1) for i in range(self.n_channels)] 
            loss =  [torch.mean(torch.stack([self.criterion(decoded[j,i:i+1,:,:], img_t[i][j]) for i in range(self.n_channels)])) for j in range(len(decoded))]
        return np.array(loss)


    def get_top_k_anomalious(self, imgs, k, show=False):
        reconstruction_loss = self.get_reconstruction_loss(imgs)
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
    

    def save(self, path):
        torch.save(self.state_dict(), path)

    @staticmethod
    def load(path, n_channels, criterion, latent_dim):
        model = Autoencoder_cnn3(n_channels, criterion, latent_dim)
        model.load_state_dict(torch.load(path))
        return model
            


#### VAE #####
class VAE3(nn.Module):

    def __init__(self, n_channels, criterion, latent_dim):
        super().__init__()
        self.criterion = criterion #nn.MSELoss()
        self.latent_dim = latent_dim
        self.n_channels = n_channels
        
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1, padding=0),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(128*3*3, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
              #  nn.Linear(64, latent_dim), # TODO:
               # nn.BatchNorm1d(latent_dim)
            )
            for _ in range(n_channels)
        ])

        
        self.z_mean = nn.ModuleList([
            nn.Linear(64, latent_dim)
            for _ in range(n_channels)
        ])

        self.z_log_var = nn.ModuleList([
            nn.Linear(64, latent_dim)
            for _ in range(n_channels)
        ])


        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 128*3*3),
                nn.BatchNorm1d(128*3*3),
                nn.ReLU(),
                nn.Unflatten(1, (128, 3, 3)),
                nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=1, padding=0),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=0),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.Sigmoid()
            )
            for _ in range(n_channels)
        ])

    
    def sample(self, z_mean: torch.Tensor, z_log_var: torch.Tensor):
        eps = torch.randn_like(z_mean) # generate random noise with the same shape as z_mean
        z = z_mean + eps*torch.exp(z_log_var/2.) 
        return z
    


    def encode(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        x = [x[:,:,:,i].unsqueeze(1) for i in range(self.n_channels)] # Extract each channel and add a new dimension for the channel
    
        encoded_list = []
        z_mean_list = []
        z_log_var_list = []
        
        for i in range(self.n_channels):
            encoded_list.append(self.encoder[i](x[i]))
            z_mean_list.append(self.z_mean[i](encoded_list[i]))
            z_log_var_list.append(self.z_log_var[i](encoded_list[i]))
        
        encoded = torch.stack(encoded_list, dim=1)
        z_mean = torch.stack(z_mean_list, dim=1)
        z_log = torch.stack(z_log_var_list, dim=1)
        return encoded, z_mean, z_log
    
    def decode(self, z):
        decoded = []
        for i in range(self.n_channels):
            decoded.append(self.decoder[i](z[:,i,:]))
        decoded = torch.cat(decoded, dim=1)  # Concatenate decoded channels
        return decoded

    def forward(self, x):
        x,  z_mean, z_log = self.encode(x)
        encoded = self.sample(z_mean, z_log)
        decoded = self.decode(encoded)
        return encoded, z_mean, z_log, decoded
    
    def generate(self, n_samples):
        with torch.no_grad():
            z = torch.randn(n_samples, self.n_channels, self.latent_dim)
        decoded = self.decode(z)
        return decoded.detach().numpy()
    

    def fit(self, gen:StackedMNISTData, epochs, batch_size=64, visualize=False, kl_div_weight=0.01):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0005) 
        if visualize: #show original images
            outputs = []
        for epoch in range(epochs):
            batch_generator = gen.batch_generator(training=True, batch_size=batch_size)
            for img, _ in batch_generator:
                encoded, z_mean, z_log, decoded = self(img)
                x = torch.tensor(img, dtype=torch.float32)
                x = [x[:,:,:,i].unsqueeze(1) for i in range(self.n_channels)]
                reconstruction_losses = [torch.stack([self.criterion(decoded[k,i:i+1,:,:], x[i][k]) for i in range(self.n_channels)]).mean() for k in range(len(x[0]))]
                #reconstruction_loss = self.criterion(decoded,img)
                #KL(Q(z|X) || P(z)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
                kl_div = -0.5*torch.sum(1+z_log-z_mean**2-torch.exp(z_log), axis=2)
                loss = torch.stack(reconstruction_losses) + kl_div_weight*kl_div #0.01
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

        if visualize:
            for k in range(0, epochs, 2):
                plt.figure(figsize=(9,2))
                plt.suptitle(f"Epoch {k+1}")
               # plt.gray()
                imgs = outputs[k][1]
                recon = outputs[k][2].detach().numpy()
                for i, item in enumerate(imgs):
                    if i >= 9: break
                    plt.subplot(2,9,i+1)
                    plt.imshow(item.astype(np.float))                       
                        
                for i, item in enumerate(recon):
                    if i >= 9: break
                    plt.subplot(2,9,9+i+1)
                    plt.imshow(np.transpose(item, (1, 2, 0)).astype(np.float))

    def save(self, path):
        torch.save(self.state_dict(), path)

    @staticmethod
    def load(path, n_channels, criterion, latent_dim):
        model = VAE3(n_channels, criterion, latent_dim)
        model.load_state_dict(torch.load(path))
        return model


def main():
    gen = StackedMNISTData(mode=DataMode.COLOR_BINARY_COMPLETE, default_batch_size=10)            
    test_img, test_cls = gen.get_full_data_set(training=False)
    vae = VAE3(n_channels=1,criterion=nn.BCELoss(),latent_dim=8)
    #vae(test_img)
    vae.fit(gen,epochs=4,batch_size=64, visualize=True, kl_div_weight = 0.01) #.01 for rgb
  #  vae.generate(10)


if __name__ == "__main__":
    main()