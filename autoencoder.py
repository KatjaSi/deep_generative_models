import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

from stacked_mnist import StackedMNISTData, DataMode


class Autoencoder_cnn(nn.Module):

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
            nn.Linear(64,latent_dim),
            nn.BatchNorm1d(latent_dim)
        )

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

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self,x):
        x = torch.tensor(x, dtype=torch.float32)
        channel_tensors = [x[:,:,:,i].unsqueeze(1) for i in range(self.n_channels)] # Extract each channel and add a new dimension for the channel
        x = torch.cat(channel_tensors, dim=1) 
        encoded = self.encoder(x)
        return encoded
    
    def decode(self, z):
        decoded = self.decoder(z)
        decoded = decoded.detach().numpy()
        return decoded

    def generate(self, n_samples):
        with torch.no_grad():
            z = torch.randn(n_samples,  self.latent_dim)
            samples = self.decode(z)
        return samples

    def fit(self, gen:StackedMNISTData, epochs, batch_size=64, visualize=False):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0005) #TODO: generalize
        if visualize: #show original images
            outputs = []
        for epoch in range(epochs):
            batch_generator = gen.batch_generator(training=True, batch_size=batch_size)
            for img, _ in batch_generator:
                img = torch.tensor(img, dtype=torch.float32)
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
        

    def get_reconstruction_loss(self, img):
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
            recon = self(img)
            loss = [self.criterion(recon[i], img[i]).item() for i in range(len(img))]
        return np.array(loss)

    def get_anomalious_imgs(self, img, img_cls = None, rule ="3-sigma", show_random=False, n_random=10):
        """
        Calculates the anomaly images for a given input image by comparing the reconstruction loss with the threshold value 
        based on the 3-sigma rule. If the img_cls parameter is provided, it also returns the corresponding class labels of the anomaly images.
        If show_random is set to True, the method selects n_random random anomaly images and displays them in a figure.
        
        The method returns the anomaly images, and optionally, the corresponding class labels.

        Args:

        img (numpy.ndarray): The input image to calculate the anomaly images for.
        img_cls (numpy.ndarray, optional): The class labels for the input image. Defaults to None.
        show_random (bool, optional): If True, displays n_random random anomaly images. Defaults to False.
        n_random (int, optional): The number of random anomaly images to display. Defaults to 10.

        Returns:

        If img_cls is provided, returns a numpy.ndarray of the anomaly images and the corresponding class labels.
        If img_cls is not provided, returns a numpy.ndarray of the anomaly images.
        """
        reconstruction_loss = self.get_reconstruction_loss(img)
        if rule == "3-sigma":
            anomaly_loss = reconstruction_loss.mean() +3*reconstruction_loss.std() # 3-sigma rule 
        elif rule == "2-sigma":
            anomaly_loss = reconstruction_loss.mean() +2*reconstruction_loss.std() # 2-sigma rule 
        else:
            raise Exception(f"No such rule as {rule} can be applied.")
        anomaly_imgs = img[reconstruction_loss > anomaly_loss]
        if show_random:
            n_images = min(n_random, len(anomaly_imgs))
            n_rows = math.ceil(n_images / 10)  # calculate the number of rows
            plt.figure(figsize=(15, n_rows * 3))  # adjust the figure height 
            # select  n_random random row indices
            row_indices = np.random.choice(anomaly_imgs.shape[0], size = n_images, replace=False)
            # select the corresponding sub-arrays
            n_random = anomaly_imgs[row_indices]
            for i, item in enumerate(anomaly_imgs[:n_images]):
                plt.subplot(n_rows, 10, i + 1)
                plt.imshow(item.astype(np.float), cmap='gray', vmin=0, vmax=1)  # specify cmap and scaling
                plt.axis('off')  # turn off axis
        if img_cls is not None:
            anomaly_imgs_cls = img_cls[reconstruction_loss > anomaly_loss]
            return anomaly_imgs, anomaly_imgs_cls
        return anomaly_imgs
        

    def save(self, filepath):
        # Create the directory if it does not exist
        os.makedirs(filepath, exist_ok=True)

        encoder_state_dict = self.encoder.state_dict()
        decoder_state_dict = self.decoder.state_dict()
        torch.save(encoder_state_dict, filepath+"/encoder")
        torch.save(decoder_state_dict, filepath+"/decoder")

    @staticmethod
    def load(filepath, n_channels, latent_dim):
        encoder_state_dict = torch.load(filepath+"/encoder")
        decoder_state_dict = torch.load(filepath+"/decoder")
        model =  Autoencoder_cnn(n_channels=n_channels, criterion=nn.BCELoss(),latent_dim=latent_dim)
        model.encoder.load_state_dict(encoder_state_dict)
        model.decoder.load_state_dict(decoder_state_dict)
        return model
        
        



def main():
    #gen_miss = StackedMNISTData(mode=DataMode.MONO_BINARY_MISSING, default_batch_size=10)
    #model_miss = Autoencoder_cnn(n_channels=1, criterion=nn.BCELoss(),latent_dim=16)
    #model_miss.fit(gen_miss, epochs=16, visualize=True)
    #test_set = gen_miss.get_full_data_set(training = False)
    #test_img, test_cls = test_set
    #anomalies = model_miss.get_anomalious_imgs(test_img,test_cls,show_random=True, n_random=10)
    #gen = StackedMNISTData(mode=DataMode.MONO_BINARY_COMPLETE, default_batch_size=10)
    #model = Autoencoder_cnn(n_channels=1, criterion=nn.BCELoss(),latent_dim=2) #16
    #model.fit(gen, epochs=40, visualize=True)
    model = Autoencoder_cnn.load("autoencoders/model_1", n_channels=1, latent_dim=4)
                

if __name__ == "__main__":
    main()