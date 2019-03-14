#first model
#encoder: four conv layers 3-16-32-64-128-FC
#decoder: four deconv layers FC-128-64-32-16-3
import torch
import torch.nn as nn
import torch.nn.functional as F

class DAE(nn.Module):
    def __init__(self, IMAGE_SIZE):
        super(DAE, self).__init__()

        self.image_dim = IMAGE_SIZE # a 28x28 image corresponds to 4 on the FC layer, a 64x64 image corresponds to 13
                            # can calculate this using output_after_conv() in utils.py
        self.latent_dim = 100
        self.noise_scale = 0
        self.batch_size = 50
        
        self.del1_size = int(IMAGE_SIZE / 4)
        self.del2_size = int(IMAGE_SIZE / 2)

        self.encoder_l1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.encoder_l2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.encoder_l3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU())
        self.encoder_l4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU())
        self.fc1 = nn.Linear(128*self.del1_size*self.del1_size, self.latent_dim)
        self.fc2 = nn.Linear(self.latent_dim, 128*self.del1_size*self.del1_size)
        self.decoder_l1 = nn.Sequential(
            nn.ConvTranspose2d(129, 64, kernel_size=3, stride=2, padding=1, output_padding = 1),
            nn.ReLU())
        #self.m1 = nn.MaxPool2d(3, stride=2)
        self.decoder_l2 = nn.Sequential(
            nn.ConvTranspose2d(65, 32, kernel_size=3, stride=2, padding=1, output_padding = 1),
            nn.ReLU())
        #self.m1 = nn.MaxPool2d(3, stride=2)
        self.decoder_l3 = nn.Sequential(
            nn.ConvTranspose2d(33, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        #self.m1 = nn.MaxPool2d(3, stride=2)
        self.decoder_l4 = nn.Sequential(
            nn.ConvTranspose2d(17, 3, kernel_size=3, stride=1, padding=1))
	    #nn.Sigmoid())
        
        
    def forward(self, x, mask):
        n = x.size()[0]

        #x = torch.add(x, noise)
        #print("****")
        #print(x.shape)
        x = self.encoder_l1(x)
        #print(x.shape)
        x = self.encoder_l2(x)
        #print(x.shape)
        x = self.encoder_l3(x)
        #print(x.shape)
        z = self.encoder_l4(x)
        #print(x.shape)
        #z = self.encoder(x)
        z = z.view(-1, 128*self.del1_size*self.del1_size)
        z = self.fc1(z)
        x_hat = self.fc2(z)
        x_hat = x_hat.view(-1, 128, self.del1_size, self.del1_size)
        
        #print(x_hat.shape)
        
        m1 = nn.MaxPool2d(4, stride = 4)
        mask_l1 = m1(mask)
        #print(mask_l1.shape)
        #print(type(x_hat), type(mask_l1))
        x_hat = torch.cat((x_hat, mask_l1), dim = 1)
        x_hat = self.decoder_l1(x_hat)
        #print(x_hat.shape)
        m2 = nn.MaxPool2d(2, stride = 2)
        mask_l2 = m2(mask)
        x_hat = torch.cat((x_hat, mask_l2), dim = 1)
        #print(mask_l2.shape)
        x_hat = self.decoder_l2(x_hat)
        x_hat = torch.cat((x_hat, mask), dim = 1)
        #print(x_hat.shape)
        x_hat = self.decoder_l3(x_hat)
        x_hat = torch.cat((x_hat, mask), dim = 1)
        #print(x_hat.shape)
        x_hat = self.decoder_l4(x_hat)

        return z, x_hat

    def encode(self, x):
        #x = x.unsqueeze(0)
        z, x_hat = self.forward(x)

        return z