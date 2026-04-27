import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import random
from torchvision.transforms import Compose, ToTensor, Lambda
from datetime import datetime
import torch, torchvision
import subprocess
from torchvision import datasets, transforms
from PIL import Image
import zipfile
import io as imgio
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage import io, color


'''
combine all training loss plots into the training funcitons (GAN**)
save plot at end of training function per model
UPDATE all models to save every x epochs

'''

class CelebAZipDataset(Dataset):
    def __init__(self, zip_path, transform=None):
        self.zip_path = zip_path
        self.transform = transform

        # Open zip once to collect all image filenames
        with zipfile.ZipFile(zip_path, 'r') as zf:
            self.image_names = sorted([
                name for name in zf.namelist()
                if name.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # Re-open zip per worker (required for DataLoader multiprocessing)
        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            with zf.open(self.image_names[idx]) as f:
                img = Image.open(imgio.BytesIO(f.read())).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img,0



#--------------------------------------------------------------------------------------------------------------------
#
#           VAE model
#
#--------------------------------------------------------------------------------------------------------------------

class VAELoss(nn.Module):
    def forward(self,recon_x, x, mu, logvar):
        """
        VAE loss = reconstruction loss + KL divergence
        - Reconstruction loss: binary cross-entropy (since pixel values are in [0,1])
        - KL divergence: between N(mu, var) and N(0,1)
        """
        # Reconstruction loss (per pixel, summed over all pixels, averaged over batch)
        BCE = nn.functional.binary_cross_entropy(recon_x, x.view(x.size(0),-1), reduction='sum')
        
        # KL divergence: see Appendix B of VAE paper
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return BCE + KL


class VAEModel(nn.Module):
    def __init__(self, input_dim=12288,
                 hidden_dim=1024, 
                 latent_dim=128,
                 batch_size=128,
                 learning_rate=1e-3,
                 device=None):
        super(VAEModel, self).__init__()


        self.hidden_dim = hidden_dim
        self.input_dim=input_dim
        self.latent_dim=latent_dim
        # self.epochs=epochs
        self.batch_size=batch_size
        self.lr = learning_rate
        self.device = device
        self.train_loss_vector = []
        
        dt = datetime.now()
        date = dt.strftime('%Y%m%d%H%M%S')
        self.date = date

        self.optimizer=None
        self.lossfn=None
        self.fitted = False
        # self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()               # Output in [0,1] for pixel values
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick: sample from N(mu, var) using N(0,1)."""
        std = torch.exp(0.5 * logvar)   # Standard deviation
        eps = torch.randn_like(std)      # eps ~ N(0,1)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        # Flatten the image
        x_flat = x.view(x.size(0), -1)
        mu, logvar = self.encode(x_flat)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def train(self, train_loader, saveX, epochs, model):
        # -------------------- Training Loop --------------------
        train_losses = []
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (data, _) in enumerate(train_loader):
                data = data.to(self.device)
                self.optimizer.zero_grad()
                
                recon_batch, mu, logvar = self.forward(data)#self.model(data)
                #loss = loss_function(recon_batch, data, mu, logvar)
                loss = self.lossfn.forward(recon_batch, data, mu, logvar)

                loss.backward()
                total_loss += loss.item()
                self.optimizer.step()
                
                if batch_idx % 10 == 0:
                    print(f'Epoch {epoch+1} [{batch_idx*len(data)}/{len(train_loader.dataset)}] Loss: {loss.item()/len(data):.4f}')
            

            avg_loss = total_loss / len(train_loader.dataset)
            train_losses.append(avg_loss)
            print(f'====> Epoch {epoch+1} Average loss: {avg_loss:.4f}')

            # save model every saveX epochs
            if epoch % saveX == 0:
                print(f'Epoch {epoch+1}- saving model')
                self.save(epoch+1, model)

        # -------------------- Plot Training Loss --------------------
        self.save(epochs+1, model)
        self.saveDec(model)
        self.fitted =True

        figName= self.date+"VAEModel_train.pdf"
        plt.figure()
        plt.plot(range(1, epochs+1), train_losses, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig(figName)

        self.train_loss_vector.append(train_losses)
        return

    def save(self,epochNum, model):
        # if not self.fitted:
        #     raise ValueError("Model must be trained before saving")
        
        # filename = self.date+'VAEModel_save_'+str(epochNum)+'.onnx'
        filename = "VAEModel_save.onnx"
        # self.eval()
        model.to('cpu')
        dummy_input = torch.zeros(1, 3, 64, 64)

        # Export the model to ONNX format
        torch.onnx.export(
            model,              # Model to export
            dummy_input,            # Example model input
            filename,           # Output file name
            export_params=True,     # Store trained parameters
            opset_version=18,       # ONNX opset version
            do_constant_folding=True,  # Optimize constants
            input_names=['input'],  # Input tensor name
            output_names=['output'], # Output tensor name
            dynamic_axes={          # Allow dynamic batch size
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        model.to(self.device)
        # self.train()

    def saveDec(self, model): 
        decoder = self.decoder
        dummy_input = torch.randn(1, self.latent_dim)

        filename = "VAEDecoder_save.onnx"
        # self.eval()
        model.to('cpu')

        # Export the model to ONNX format
        torch.onnx.export(
            decoder,              # Model to export
            dummy_input,            # Example model input
            filename,           # Output file name
            export_params=True,     # Store trained parameters
            opset_version=18,       # ONNX opset version
            do_constant_folding=True,  # Optimize constants
            input_names=['latent'],  # Input tensor name
            output_names=['image'], # Output tensor name
            dynamic_axes={          # Allow dynamic batch size
                'latent': {0: 'batch_size', 2:'height', 3:'width' },
                'image': {0: 'batch_size', 2:'height', 3:'width'}
            }
        )
        model.to(self.device)
        # model.train()

    def test(self, test_loader):
        # -------------------- Test: Reconstructions --------------------
        self.model.eval()
        with torch.no_grad():
            # Get a batch of test images
            data, _ = next(iter(test_loader))
            data = data.to(self.device)
            recon, _, _ = self.model(data)
            
            # Move to CPU for plotting
            data = data.cpu()
            recon = recon.cpu().view(-1, 1, 28, 28)

        # Plot original and reconstructed images
        figName = self.date + "VAEModel_test.pdf"
        n = 10  # number to display
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # Original
            ax = plt.subplot(2, n, i+1)
            plt.imshow(data[i].squeeze(), cmap='gray')
            plt.title("Original")
            plt.axis('off')
            
            # Reconstruction
            ax = plt.subplot(2, n, i+1+n)
            plt.imshow(recon[i].squeeze(), cmap='gray')
            plt.title("Reconstructed")
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(figName)

    def generate_samples(self, num=25):
        # -------------------- Generate New Digits --------------------
        with torch.no_grad():
            # Sample random latent vectors from the prior (standard normal)
            z = torch.randn(64, latent_dim).to(device)
            samples = model.decode(z).cpu().view(-1, 1, 64, 64)

        # Plot generated digits
        plt.figure(figsize=(10, 10))
        for i in range(num):
            plt.subplot(5, 5, i+1)
            plt.imshow(samples[i].squeeze(), cmap='gray')
            plt.axis('off')
        plt.suptitle("Generated Images")
        plt.tight_layout()
        figName = self.date + "VAE_gen_img_"+imgName+'.pdf'
        plt.savefig(figName)

        return samples


#--------------------------------------------------------------------------------------------------------------------
#
#           Model Trainer and evl - can be used for any model GAN, VAE, Diffusion
#
#--------------------------------------------------------------------------------------------------------------------

class GenModelTrainer():
    def __init__(self, dataLoader, device, model = None, epochs=10,
                lossfn=None, opt=None, sch=None, saveX=10):

        self.device = device
        self.epochs = epochs
        self.saveX = saveX
        self.data = dataLoader
        
        if lossfn is None:
            lossfn=VAELoss()
        self.loss = lossfn
        
        if model is None:
            model=VAEModel().to(self.device)
        self.model = model.to(self.device)
        
        if opt is None:
            opt=optim.Adam(self.model.parameters(), lr=0.01)
        self.optimizer = opt

        if sch is None:
            sch=optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)    
        self.scheduler = sch    

    def train(self):
        self.model.train(self.data, self.saveX, self.epochs, self.model)

    # def save(self): #called every x epochs - model will save as part of training and doesnt need to be called from trainer seperatly
    #     self.model.save()

    def evaluation(self):
        self.model.evaluation()


class GenModelEval():
    def __init__(self):
        # transform = transforms.Grayscale(num_output_channels=1)
        # gray_img = transform(img)
        self.vl_results=[]
        self.tenengrad_results=[]
        self.FDHF_results=[]
        self.mlstd_results=[]
        self.GLCM_results=[]

    def perform_eval_metrics(self, img):
        vl = self.var_laplacian(img)
        tn = self.tenengrad(img)
        FDHF = self.FDHFEnergyRatio(img)
        mlstd = self.mlstd(img)
        glcm = self.GLCMContrast(img)
        # print(f"vl: {vl}\n tn: {tn}\n fdhf: {FDHF}\n mlstd: {mlstd}\n glcm: {glcm}\n")
        # self.visualize()

    def var_laplacian(self, image):
        laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
        self.vl_results.append(laplacian_var)
        # return
        return laplacian_var

    def tenengrad(self, img):
        Sx=np.array([[-1,0,1],
            [-2,0,2],
            [-1,0,1]])

        Sy=np.array([[-1,-2,-1],
            [0,0,0],
            [1,2,1]])

        Gx = cv2.filter2D(img,-1, Sx)
        Gy = cv2.filter2D(img,-1, Sy)
        G = np.sqrt(Gx**2 + Gy**2)
        G = G.mean()
        self.tenengrad_results.append(G)
        return G
        # return

    def FDHFEnergyRatio(self, img, low_freq_radius=30): #taken from the google ai result
        # 1. Compute 2D Fourier Transform
        dft = np.fft.fft2(img)
        
        # 2. Shift zero frequency to center
        dft_shifted = np.fft.fftshift(dft)
        
        # 3. Calculate Power Spectrum (magnitude squared)
        power_spectrum = np.abs(dft_shifted)**2
        total_energy = np.sum(power_spectrum)
        
        # 4. Define High Frequency region (exclude a central circle/square)
        rows, cols = img.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create a circular mask for low frequencies
        y, x = np.ogrid[:rows, :cols]
        mask = (x - ccol)**2 + (y - crow)**2 <= low_freq_radius**2
        
        # High frequency energy is everything NOT in the mask
        high_freq_energy = np.sum(power_spectrum[~mask])
        
        # 5. Calculate Ratio
        hfer = high_freq_energy / total_energy
        self.FDHF_results.append(hfer)
        return hfer
        # return
   
    def mlstd(self, img): #from google ai result
        
        # 1. Compute local mean (E[X])
        ksize = (5, 5) # Neighborhood size
        local_mean = cv2.blur(img, ksize)

        # 2. Compute local mean of squares (E[X^2])
        local_mean_sq = cv2.blur(img**2, ksize)

        # 3. Calculate LSD: sqrt(E[X^2] - (E[X])^2)
        local_std = np.sqrt(np.maximum(local_mean_sq - local_mean**2, 0))
        local_std = local_std.mean()
        self.mlstd_results.append(local_std)
        return local_std

    def GLCMContrast(self, image): # from google ai result
        # 2. Compute GLCM 
        # distances: distance between pixel pairs (e.g., 1 pixel)
        # angles: direction (0=horizontal, np.pi/2=vertical, etc.)
        glcm = graycomatrix(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                            levels=256, symmetric=True, normed=True)

        # 3. Calculate Contrast
        contrast = graycoprops(glcm, 'contrast').mean()

        self.GLCM_results.append(contrast)
        return contrast

    def visualize(self):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        # laplacian varienace histogram
        axes[0][0].hist(self.vl_results, bins=10, color='steelblue', edgecolor='white')
        axes[0][0].axvline(np.mean(self.vl_results), color='red', label=f'mean={np.mean(self.vl_results):.3f}')
        axes[0][0].axvline(np.median(self.vl_results), color='orange', label=f'median={np.median(self.vl_results):.3f}')
        axes[0][0].set_title("Laplacian Variance")
        axes[0][0].set_xlabel('Value')
        axes[0][0].set_ylabel('Count')
        axes[0][0].legend(fontsize=7)

        # tenengrad histogram
        axes[0][1].hist(self.tenengrad_results, bins=10, color='steelblue', edgecolor='white')
        axes[0][1].axvline(np.mean(self.tenengrad_results), color='red', label=f'mean={np.mean(self.tenengrad_results):.3f}')
        axes[0][1].axvline(np.median(self.tenengrad_results), color='orange', label=f'median={np.median(self.tenengrad_results):.3f}')
        axes[0][1].set_title("Tenengrad Criterion")
        axes[0][1].set_xlabel('Value')
        axes[0][1].set_ylabel('Count')
        axes[0][1].legend(fontsize=7)

        # FDHF histogram
        axes[0][2].hist(self.FDHF_results, bins=10, color='steelblue', edgecolor='white')
        axes[0][2].axvline(np.mean(self.FDHF_results), color='red', label=f'mean={np.mean(self.FDHF_results):.3f}')
        axes[0][2].axvline(np.median(self.FDHF_results), color='orange', label=f'median={np.median(self.FDHF_results):.3f}')
        axes[0][2].set_title("Frequency-Domain High-Frequency Energy Ratio")
        axes[0][2].set_xlabel('Value')
        axes[0][2].set_ylabel('Count')
        axes[0][2].legend(fontsize=7)

        # mlstd histogram
        axes[1][0].hist(self.mlstd_results, bins=10, color='steelblue', edgecolor='white')
        axes[1][0].axvline(np.mean(self.mlstd_results), color='red', label=f'mean={np.mean(self.mlstd_results):.3f}')
        axes[1][0].axvline(np.median(self.mlstd_results), color='orange', label=f'median={np.median(self.mlstd_results):.3f}')
        axes[1][0].set_title("Mean Local Standard Deviation")
        axes[1][0].set_xlabel('Value')
        axes[1][0].set_ylabel('Count')
        axes[1][0].legend(fontsize=7)

        # GCLM histogram
        axes[1][1].hist(self.GLCM_results, bins=10, color='steelblue', edgecolor='white')
        axes[1][1].axvline(np.mean(self.GLCM_results), color='red', label=f'mean={np.mean(self.GLCM_results):.3f}')
        axes[1][1].axvline(np.median(self.GLCM_results), color='orange', label=f'median={np.median(self.GLCM_results):.3f}')
        axes[1][1].set_title("GLCM Contrast")
        axes[1][1].set_xlabel('Value')
        axes[1][1].set_ylabel('Count')
        axes[1][1].legend(fontsize=7)

        plt.suptitle('Generated Image Evaluation Metrics', fontsize=14)
        plt.tight_layout()

        fname = "VAE_images/GenModelEval_metrics.pdf"
        plt.savefig(fname)
        print(f"Evaluation metrics saved to {fname}")
        return fig

#--------------------------------------------------------------------------------------------------------------------
#
#           GAN model
#
#--------------------------------------------------------------------------------------------------------------------


# class Generator(nn.Module):
#     def __init__(self, latent_dim=100, height  = 28, width = 28):
#         super(Generator, self).__init__()
#         self.latent_dim = latent_dim
#         self.height = height
#         self.width = width
        
#         # Build a simple feedforward network (MLP)
#         # Goodfellow's original paper used fully connected layers
#         self.model = nn.Sequential(
#             # Input: latent_dim (100) -> Hidden 1: 256
#             nn.Linear(latent_dim, 256),
#             nn.LeakyReLU(0.2),
            
#             # Hidden 1 -> Hidden 2: 512
#             nn.Linear(256, 512),
#             nn.LeakyReLU(0.2),
            
#             # Hidden 2 -> Hidden 3: 1024
#             nn.Linear(512, 1024),
#             nn.LeakyReLU(0.2),
            
#             # Hidden 3 -> Output: 784 (28*28 flattened image)  for MNIST
#             nn.Linear(1024, self.width*self.height),
#             nn.Tanh()  # Output in [-1, 1] to match normalized data
#         )
    
#     def forward(self, z):
#         # z shape: [batch_size, latent_dim]
#         img = self.model(z)
#         img = img.view(img.size(0), 1, self.height, self.width)  # Reshape to image
#         return img


# class Discriminator(nn.Module):
#     def __init__(self, height  = 28, width = 28):
#         super(Discriminator, self).__init__()
        
#         self.height = height
#         self.width = width
        
#         self.model = nn.Sequential(
#             # Input: 784 (flattened 28*28 image) -> Hidden 1: 1024 for MNIST
#             nn.Linear(self.height*self.width, 1024),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(0.3),  # Regularization
            
#             # Hidden 1 -> Hidden 2: 512
#             nn.Linear(1024, 512),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(0.3),
            
#             # Hidden 2 -> Hidden 3: 256
#             nn.Linear(512, 256),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(0.3),
            
#             # Hidden 3 -> Output: 1 (probability)
#             nn.Linear(256, 1),
#             nn.Sigmoid()  # Output probability [0, 1]
#         )
    
#     def forward(self, img):
#         # img shape: [batch_size, 1, 28, 28]
#         img_flat = img.view(img.size(0), -1)  # Flatten
#         validity = self.model(img_flat)
#         return validity


# class GANModel():
#     def __init__(self,
#         latent_dim=100,
#         lr=0.0002,
#         batch_size=64
#         epoch=50,
#         h=28,
#         w=28,
#         device=(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
#         loss_funt=None,
#         optimizer=(None,None), #(generator,discriminator)
#         scheduler=None):

#         self.latent_dim=latent_dim
#         self.lr=lr
#         self.batch_size=batch_size
#         self.epochs=epoch
#         self.height=h
#         self.width=w
#         self.device=device

#         # Initialize networks
#         self.generator = Generator(self.latent_dim, self.height, self.width).to(self.device)
#         self.discriminator = Discriminator(self.height, self.width).to(self.device)      

#         if lossfn is None:
#             # Loss function: Binary Cross Entropy, adersarial loss
#             lossfn = nn.BCELoss()  
#         self.loss = lossfn
        
#         opt_G, opt_D = optimizer
#         # Optimizers (Adam as suggested in the paper)
#         if opt_G is None:
#             opt_G = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
#         if opt_D is None:
#             opt_D = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))
#         self.optimizer_G = opt_G
#         self.optimizer_D = opt_D

#         if scheduler is None:
#             scheduler=optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)    
#         self.scheduler = scheduler  

#         self.fitted = False
#         self.D_losses = None
#         self.G_losses = None

#     def train(self): #need to pass data?
#         G_losses = []
#         D_losses = []
        
#         for epoch in range(self.epochs):
#             for i, (real_imgs, _) in enumerate(train_loader):
#                 batch_size = real_imgs.size(0)
#                 real_imgs = real_imgs.to(device)
                
#                 # Create labels
#                 real_labels = torch.ones(batch_size, 1).to(device)
#                 fake_labels = torch.zeros(batch_size, 1).to(device)
                
#                 # ====================
#                 # Train Discriminator
#                 # ====================
#                 optimizer_D.zero_grad()
                
#                 # Loss on real images
#                 real_output = discriminator(real_imgs)
#                 d_loss_real = adversarial_loss(real_output, real_labels)
                
#                 # Generate fake images
#                 z = torch.randn(batch_size, latent_dim).to(device)
#                 fake_imgs = generator(z)
                
#                 # Loss on fake images
#                 fake_output = discriminator(fake_imgs.detach())  # detach() so G doesn't get gradients
#                 d_loss_fake = adversarial_loss(fake_output, fake_labels)
                
#                 # Total discriminator loss
#                 d_loss = (d_loss_real + d_loss_fake) / 2
#                 d_loss.backward()
#                 optimizer_D.step()
                
#                 # ====================
#                 # Train Generator
#                 # ====================
#                 optimizer_G.zero_grad()
                
#                 # Generate fake images again (or reuse, but fresh noise often works better)
#                 z = torch.randn(batch_size, latent_dim).to(device)
#                 fake_imgs = generator(z)
                
#                 # Try to fool the discriminator (want D to output 1 for fakes)
#                 output = discriminator(fake_imgs)
#                 g_loss = adversarial_loss(output, real_labels)  # Trick D with real_labels
                
#                 g_loss.backward()
#                 optimizer_G.step()
                
#                 # Store losses
#                 G_losses.append(g_loss.item())
#                 D_losses.append(d_loss.item())
                
#                 # Print progress
#                 if i % 100 == 0:
#                     print(f"Epoch [{epoch}/{epochs}] Batch [{i}/{len(train_loader)}] "
#                         f"D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")
            
#             # Save sample images every few epochs
#             if epoch % 5 == 0:
#                 save_samples(generator, epoch)
        
#         self.G_losses = G_losses
#         self.D_losses = D_losses
#         self.fitted = True

#         return G_losses, D_losses

#     def save_samples(self, generator, epoch, n_samples=16): #from github code - update for class impl
#         """Generate and save sample images"""
#         z = torch.randn(n_samples, latent_dim).to(device)
#         gen_imgs = generator(z).cpu().detach()
        
#         # Denormalize from [-1, 1] to [0, 1]
#         gen_imgs = (gen_imgs + 1) / 2
        
#         fig, axes = plt.subplots(4, 4, figsize=(6, 6))
#         for i, ax in enumerate(axes.flat):
#             ax.imshow(gen_imgs[i].squeeze(), cmap='gray')
#             ax.axis('off')
#         plt.suptitle(f'Generated Samples - Epoch {epoch}')
#         plt.tight_layout()
#         plt.savefig(f'gan_epoch_{epoch}.png')
#         plt.close()

#     def save(self, filename:str='GANModel.onnx'):
#         if not self.fitted:
#             raise ValueError("Model must be trained before saving")
        
#         self.model.eval()
#         self.model.to('cpu')
#         for x,y in self.train_loader:
#             x = x.to(self.device)
#             y = y.to(self.device)
#             dummy_input = x[0].unsqueeze(0).to('cpu')  # Adds batch dimension
#             break

#         # Export the model to ONNX format
#         torch.onnx.export(
#             self.model,              # Model to export
#             dummy_input,            # Example model input
#             filename,           # Output file name
#             export_params=True,     # Store trained parameters
#             opset_version=11,       # ONNX opset version
#             do_constant_folding=True,  # Optimize constants
#             input_names=['input'],  # Input tensor name
#             output_names=['output'], # Output tensor name
#             dynamic_axes={          # Allow dynamic batch size
#                 'input': {0: 'batch_size'},
#                 'output': {0: 'batch_size'}
#             }
#         )
#         self.model.to(self.device)
#         self.model.train()

    
#     def evaluation(self):
#         plt.figure(figsize=(10, 5))
#         plt.plot(G_losses, label='Generator Loss')
#         plt.plot(D_losses, label='Discriminator Loss')
#         plt.xlabel('Iterations')
#         plt.ylabel('Loss')
#         plt.legend()
#         plt.title('GAN Training Losses')
#         plt.show()

#     def generate_samples(n=25): #taken from github code - update for class impl
#         if not self.fitted:
#             raise ValueError("GANModel must be trained before saving")
        
#         generator.eval()
#         z = torch.randn(n, latent_dim).to(device)
#         with torch.no_grad():
#             gen_imgs = generator(z).cpu()
        
#         # Denormalize
#         gen_imgs = (gen_imgs + 1) / 2
        
#         fig, axes = plt.subplots(5, 5, figsize=(8, 8))
#         for i, ax in enumerate(axes.flat):
#             ax.imshow(gen_imgs[i].squeeze(), cmap='gray')
#             ax.axis('off')
#         plt.suptitle('Final Generated MNIST Digits')
#         plt.tight_layout()
#         plt.show()


#--------------------------------------------------------------------------------------------------------------------
#
#           Diffusion model
#
#--------------------------------------------------------------------------------------------------------------------
# def sinusoidal_embedding(n, d):
#     # Returns the standard positional embedding 
#     embedding = torch.zeros(n, d)
#     wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
#     wk = wk.reshape((1, d))
#     t = torch.arange(n).reshape((n, 1))
#     embedding[:,::2] = torch.sin(t * wk[:,::2])
#     embedding[:,1::2] = torch.cos(t * wk[:,::2])

#     return embedding

# # DDPM (denoising diffusion probablistic model) class
# class DiffusionModel(nn.Module):
#     def __init__(self, 
#         network, 
#         n_steps=200, 
#         min_beta=10 ** -4, 
#         max_beta=0.02, 
#         device=(torch.device('cuda' if torch.cuda.is_available() else 'cpu')), 
#         image_chw=(3, 64, 64),
#         batch_size=128, #ADDED DEFAULTS FROM FIRST PART OF GITHUB CODE HERE....
#         epochs=20,
#         lr=0.001):
#         super(DiffusionModel, self).__init__()

#         self.n_steps = n_steps
#         self.device = device
#         self.image_chw = image_chw
#         self.network = network.to(device)
#         self.betas = torch.linspace(min_beta, max_beta, n_steps).to(
#             device)  # Number of steps is typically in the order of thousands
#         self.alphas = 1 - self.betas
#         self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(device)
       
#         self.batch_size=batch_size
#         self.epochs=epochs
#         self.lr=lr


#     def forward(self, x0, t, eta=None):
#         # Make input image more noisy (we can directly skip to the desired step)
#         n, c, h, w = x0.shape
#         a_bar = self.alpha_bars[t]

#         if eta is None:
#             eta = torch.randn(n, c, h, w).to(self.device)

#         noisy = a_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eta
#         return noisy

#     def backward(self, x, t):
#         # Run each image through the network for each timestep t in the vector t.
#         # The network returns its estimation of the noise that was added.
#         return self.network(x, t)

#     def generate_samples(self, n_samples=16, device=None, frames_per_gif=100, gif_name="sampling.gif", c=1, h=28, w=28):
#         """Given a DDPM model, a number of samples to be generated and a device, returns some newly generated samples"""
#         frame_idxs = np.linspace(0, self.n_steps, frames_per_gif).astype(np.uint)
#         frames = []

#         with torch.no_grad():
#             if device is None:
#                 device = self.device

#             # Starting from random noise
#             x = torch.randn(n_samples, c, h, w).to(device)

#             for idx, t in enumerate(list(range(self.n_steps))[::-1]):
#                 # Estimating noise to be removed
#                 time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()
#                 eta_theta = self.backward(x, time_tensor)

#                 alpha_t = self.alphas[t]
#                 alpha_t_bar = self.alpha_bars[t]

#                 # Partially denoising the image
#                 x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

#                 if t > 0:
#                     z = torch.randn(n_samples, c, h, w).to(device)

#                     # Option 1: sigma_t squared = beta_t
#                     beta_t = self.betas[t]
#                     sigma_t = beta_t.sqrt()

#                     # Option 2: sigma_t squared = beta_tilda_t
#                     # prev_alpha_t_bar = self.alpha_bars[t-1] if t > 0 else self.alphas[0]
#                     # beta_tilda_t = ((1 - prev_alpha_t_bar)/(1 - alpha_t_bar)) * beta_t
#                     # sigma_t = beta_tilda_t.sqrt()

#                     # Adding some more noise like in Langevin Dynamics fashion
#                     x = x + sigma_t * z

#                 # Adding frames to the GIF
#                 if idx in frame_idxs or t == 0:
#                     # Putting digits in range [0, 255]
#                     normalized = x.clone()
#                     for i in range(len(normalized)):
#                         normalized[i] -= torch.min(normalized[i])
#                         normalized[i] *= 255 / torch.max(normalized[i])

#                     # Reshaping batch (n, c, h, w) to be a (as much as it gets) square frame
#                     frame = einops.rearrange(normalized, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=int(n_samples ** 0.5))
#                     frame = frame.cpu().numpy().astype(np.uint8)

#                     # Rendering frame
#                     frames.append(frame)

#         # Storing the gif
#         with imageio.get_writer(gif_name, mode="I") as writer:
#             for idx, frame in enumerate(frames):
#                 rgb_frame = np.repeat(frame, 3, axis=2)
#                 writer.append_data(rgb_frame)

#                 # Showing the last frame for a longer time
#                 if idx == len(frames) - 1:
#                     last_rgb_frame = np.repeat(frames[-1], 3, axis=2)
#                     for _ in range(frames_per_gif // 3):
#                         writer.append_data(last_rgb_frame)
#         return x

#     def show_images(images, title=""):
#         """Shows the provided images as sub-pictures in a square"""

#         # Converting images to CPU numpy arrays
#         if type(images) is torch.Tensor:
#             images = images.detach().cpu().numpy()

#         # Defining number of rows and columns
#         fig = plt.figure(figsize=(8, 8))
#         rows = int(len(images) ** (1 / 2))
#         cols = round(len(images) / rows)

#         # Populating figure with sub-plots
#         idx = 0
#         for r in range(rows):
#             for c in range(cols):
#                 fig.add_subplot(rows, cols, idx + 1)

#                 if idx < len(images):
#                     plt.imshow(images[idx][0], cmap="gray")
#                     idx += 1
#         fig.suptitle(title, fontsize=30)

#         # Showing the figure
#         plt.show()


# class MyBlock(nn.Module):
#     def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, padding=1, activation=None, normalize=True):
#         super(MyBlock, self).__init__()
#         self.ln = nn.LayerNorm(shape)
#         self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
#         self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
#         self.activation = nn.SiLU() if activation is None else activation
#         self.normalize = normalize

#     def forward(self, x):
#         out = self.ln(x) if self.normalize else x
#         out = self.conv1(out)
#         out = self.activation(out)
#         out = self.conv2(out)
#         out = self.activation(out)
#         return out


# class MyUNet(nn.Module):
#     def __init__(self, n_steps=1000, time_emb_dim=100):
#         super(MyUNet, self).__init__()

#         # Sinusoidal embedding
#         self.time_embed = nn.Embedding(n_steps, time_emb_dim)
#         self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
#         self.time_embed.requires_grad_(False)

#         # First half
#         self.te1 = self._make_te(time_emb_dim, 1)
#         self.b1 = nn.Sequential(
#             MyBlock((1, 28, 28), 1, 10),
#             MyBlock((10, 28, 28), 10, 10),
#             MyBlock((10, 28, 28), 10, 10)
#         )
#         self.down1 = nn.Conv2d(10, 10, 4, 2, 1)

#         self.te2 = self._make_te(time_emb_dim, 10)
#         self.b2 = nn.Sequential(
#             MyBlock((10, 14, 14), 10, 20),
#             MyBlock((20, 14, 14), 20, 20),
#             MyBlock((20, 14, 14), 20, 20)
#         )
#         self.down2 = nn.Conv2d(20, 20, 4, 2, 1)

#         self.te3 = self._make_te(time_emb_dim, 20)
#         self.b3 = nn.Sequential(
#             MyBlock((20, 7, 7), 20, 40),
#             MyBlock((40, 7, 7), 40, 40),
#             MyBlock((40, 7, 7), 40, 40)
#         )
#         self.down3 = nn.Sequential(
#             nn.Conv2d(40, 40, 2, 1),
#             nn.SiLU(),
#             nn.Conv2d(40, 40, 4, 2, 1)
#         )

#         # Bottleneck
#         self.te_mid = self._make_te(time_emb_dim, 40)
#         self.b_mid = nn.Sequential(
#             MyBlock((40, 3, 3), 40, 20),
#             MyBlock((20, 3, 3), 20, 20),
#             MyBlock((20, 3, 3), 20, 40)
#         )

#         # Second half
#         self.up1 = nn.Sequential(
#             nn.ConvTranspose2d(40, 40, 4, 2, 1),
#             nn.SiLU(),
#             nn.ConvTranspose2d(40, 40, 2, 1)
#         )

#         self.te4 = self._make_te(time_emb_dim, 80)
#         self.b4 = nn.Sequential(
#             MyBlock((80, 7, 7), 80, 40),
#             MyBlock((40, 7, 7), 40, 20),
#             MyBlock((20, 7, 7), 20, 20)
#         )

#         self.up2 = nn.ConvTranspose2d(20, 20, 4, 2, 1)
#         self.te5 = self._make_te(time_emb_dim, 40)
#         self.b5 = nn.Sequential(
#             MyBlock((40, 14, 14), 40, 20),
#             MyBlock((20, 14, 14), 20, 10),
#             MyBlock((10, 14, 14), 10, 10)
#         )

#         self.up3 = nn.ConvTranspose2d(10, 10, 4, 2, 1)
#         self.te_out = self._make_te(time_emb_dim, 20)
#         self.b_out = nn.Sequential(
#             MyBlock((20, 28, 28), 20, 10),
#             MyBlock((10, 28, 28), 10, 10),
#             MyBlock((10, 28, 28), 10, 10, normalize=False)
#         )

#         self.conv_out = nn.Conv2d(10, 1, 3, 1, 1)

#     def forward(self, x, t):
#         # x is (N, 2, 28, 28) (image with positional embedding stacked on channel dimension)
#         t = self.time_embed(t)
#         n = len(x)
#         out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1))  # (N, 10, 28, 28)
#         out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))  # (N, 20, 14, 14)
#         out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))  # (N, 40, 7, 7)

#         out_mid = self.b_mid(self.down3(out3) + self.te_mid(t).reshape(n, -1, 1, 1))  # (N, 40, 3, 3)

#         out4 = torch.cat((out3, self.up1(out_mid)), dim=1)  # (N, 80, 7, 7)
#         out4 = self.b4(out4 + self.te4(t).reshape(n, -1, 1, 1))  # (N, 20, 7, 7)

#         out5 = torch.cat((out2, self.up2(out4)), dim=1)  # (N, 40, 14, 14)
#         out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1))  # (N, 10, 14, 14)

#         out = torch.cat((out1, self.up3(out5)), dim=1)  # (N, 20, 28, 28)
#         out = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1))  # (N, 1, 28, 28)

#         out = self.conv_out(out)

#         return out

#     def _make_te(self, dim_in, dim_out):
#         return nn.Sequential(
#             nn.Linear(dim_in, dim_out),
#             nn.SiLU(),
#             nn.Linear(dim_out, dim_out)
#         )


#     def training_loop(ddpm, loader, n_epochs, optim, device, display=False, store_path="ddpm_model.pt"):
#         mse = nn.MSELoss()
#         best_loss = float("inf")
#         n_steps = ddpm.n_steps

#         for epoch in tqdm(range(n_epochs), desc=f"Training progress", colour="#00ff00"):
#             epoch_loss = 0.0
#             for step, batch in enumerate(tqdm(loader, leave=False, desc=f"Epoch {epoch + 1}/{n_epochs}", colour="#005500")):
#                 # Loading data
#                 x0 = batch[0].to(device)
#                 n = len(x0)

#                 # Picking some noise for each of the images in the batch, a timestep and the respective alpha_bars
#                 eta = torch.randn_like(x0).to(device)
#                 t = torch.randint(0, n_steps, (n,)).to(device)

#                 # Computing the noisy image based on x0 and the time-step (forward process)
#                 noisy_imgs = ddpm(x0, t, eta)

#                 # Getting model estimation of noise based on the images and the time-step
#                 eta_theta = ddpm.backward(noisy_imgs, t.reshape(n, -1))

#                 # Optimizing the MSE between the noise plugged and the predicted noise
#                 loss = mse(eta_theta, eta)
#                 optim.zero_grad()
#                 loss.backward()
#                 optim.step()

#                 epoch_loss += loss.item() * len(x0) / len(loader.dataset)

#             # Display images generated at this epoch
#             if display:
#                 show_images(generate_new_images(ddpm, device=device), f"Images generated at epoch {epoch + 1}")

#             log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}"

#             # Storing the model
#             if best_loss > epoch_loss:
#                 best_loss = epoch_loss
#                 torch.save(ddpm.state_dict(), store_path)
#                 log_string += " --> Best model ever (stored)"

#             print(log_string)

#     def test(self):
#         # Loading the trained model
#         best_model = DiffusionModel(MyUNet(), n_steps=n_steps, device=device)
#         best_model.load_state_dict(torch.load(store_path, map_location=device))
#         best_model.eval()
#         print("Model loaded")
#         print("Generating new images")
#         generated = generate_new_images(
#                 best_model,
#                 n_samples=100,
#                 device=device,
#                 gif_name="fashion.gif"
#             )
#         show_images(generated, "Final result")


# n_steps, min_beta, max_beta = 1000, 10 ** -4, 0.02  # Originally used by the authors
# ddpm = DiffusionModel(MyUNet(n_steps), n_steps=n_steps, min_beta=min_beta, max_beta=max_beta, device=device)

# training_loop(ddpm, loader, n_epochs, optim=Adam(ddpm.parameters(), lr), device=device, store_path=store_path)

# # from IPython.display import Image
# # Image(open('fashion.gif' ,'rb').read())
