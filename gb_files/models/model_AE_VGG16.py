#first model
#encoder: four conv layers 3-16-32-64-128-FC
#decoder: four deconv layers FC-128-64-32-16-3
import torch
import torch.nn as nn
import torch.nn.functional as F

class AE_VGG16(nn.Module):
    def __init__(self, IMAGE_SIZE):
        super(AE_VGG16, self).__init__()

        self.image_dim = IMAGE_SIZE # a 28x28 image corresponds to 4 on the FC layer, a 64x64 image corresponds to 13
                            # can calculate this using output_after_conv() in utils.py
        self.latent_dim = 200
        self.noise_scale = 0
        self.batch_size = 50
        
        self.del1_size = int(IMAGE_SIZE / 16)
        self.del2_size = int(IMAGE_SIZE / 8)

        self.encoder_l1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            #nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            #nn.MaxPool2d(3, stride=2, padding = 1),
            nn.ReLU())
        self.encoder_l2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            #nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            #nn.MaxPool2d(3, stride=2, padding = 1),
            nn.ReLU())
        self.encoder_l3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            #nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            #nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            #nn.MaxPool2d(3, stride=2, padding = 1),
            nn.ReLU())
        self.encoder_l4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            #nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            #nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            #nn.MaxPool2d(3, stride=2, padding = 1),
            nn.ReLU())
        self.encoder_fc1 = nn.Sequential(
            nn.Linear(512*self.del1_size*self.del1_size, 1000),
            nn.Tanh()
        )
        self.latent_layer = nn.Linear(1000, self.latent_dim)
        self.decoder_fc1 = nn.Sequential(
            nn.Linear(self.latent_dim, 1000),
            nn.Tanh()
        )
        self.decoder_fc2 = nn.Sequential(
            nn.Linear(1000, 512*self.del1_size*self.del1_size),
            nn.Tanh()
        )
        self.decoder_l1 = nn.Sequential(
            nn.ConvTranspose2d(513, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            #nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),
            #nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1),
            #nn.MaxUnpool2d(3, stride=2),
            nn.ReLU())
        #self.m1 = nn.MaxPool2d(3, stride=2)
        self.decoder_l2 = nn.Sequential(
            nn.ConvTranspose2d(257, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            #nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),
            #nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
            #nn.MaxUnpool2d(3, stride=2),
            nn.ReLU())
        #self.m1 = nn.MaxPool2d(3, stride=2)
        self.decoder_l3 = nn.Sequential(
            nn.ConvTranspose2d(129, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            #nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            #nn.MaxUnpool2d(3, stride=2),
            nn.ReLU())
        #self.m1 = nn.MaxPool2d(3, stride=2)
        self.decoder_l4 = nn.Sequential(
            nn.ConvTranspose2d(65, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            #nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),
            #nn.MaxUnpool2d(3, stride=2)
        )
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
        x = self.encoder_l4(x)
        #print(z.shape)
        x = x.view(-1, 512*self.del1_size*self.del1_size)
        #print(x.shape)
        x = self.encoder_fc1(x)
        
        z = self.latent_layer(x)
        
        x_hat = self.decoder_fc1(z)
        x_hat = self.decoder_fc2(x_hat)
        x_hat = x_hat.view(-1, 512, self.del1_size, self.del1_size)
        m1 = nn.MaxPool2d(16, stride = 16)
        mask_l1 = m1(mask)
        #print(mask_l1.shape)
        #print(x_hat.shape)
        #print(type(x_hat), type(mask_l1))
        x_hat = torch.cat((x_hat, mask_l1), dim = 1)
        #print(x_hat.shape)
        x_hat = self.decoder_l1(x_hat)
        #print(x_hat.shape)
        m2 = nn.MaxPool2d(8, stride = 8)
        mask_l2 = m2(mask)
        x_hat = torch.cat((x_hat, mask_l2), dim = 1)
        #print(mask_l2.shape)
        x_hat = self.decoder_l2(x_hat)
        
        m3 = nn.MaxPool2d(4, stride = 4)
        mask_l3 = m3(mask)
        x_hat = torch.cat((x_hat, mask_l3), dim = 1)
        #print(x_hat.shape)
        x_hat = self.decoder_l3(x_hat)
        
        m4 = nn.MaxPool2d(2, stride = 2)
        mask_l4 = m4(mask)
        x_hat = torch.cat((x_hat, mask_l4), dim = 1)
        #print(x_hat.shape)
        x_hat = self.decoder_l4(x_hat)

        return z, x_hat

    def encode(self, x):
        #x = x.unsqueeze(0)
        z, x_hat = self.forward(x)

        return z