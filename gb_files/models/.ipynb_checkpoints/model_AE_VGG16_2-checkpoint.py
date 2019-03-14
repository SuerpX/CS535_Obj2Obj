#first model
#encoder: four conv layers 3-16-32-64-128-FC
#decoder: four deconv layers FC-128-64-32-16-3
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, IMAGE_SIZE):
        super(Encoder, self).__init__()
        self.image_dim = IMAGE_SIZE # a 28x28 image corresponds to 4 on the FC layer, a 64x64 image corresponds to 13
                            # can calculate this using output_after_conv() in utils.py
        self.latent_dim = 200
        self.noise_scale = 0
        self.batch_size = 50
        
        self.del1_size = int(IMAGE_SIZE / 32)
        self.del2_size = int(IMAGE_SIZE / 16)

        self.encoder_l1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            #nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            #nn.MaxPool2d(3, stride=2, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(64))
        self.encoder_l2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            #nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            #nn.MaxPool2d(3, stride=2, padding = 1),
            
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.encoder_l3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            #nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            #nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            #nn.MaxPool2d(3, stride=2, padding = 1),
            
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        self.encoder_l4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            #nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            #nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            #nn.MaxPool2d(3, stride=2, padding = 1),
            
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )
        self.encoder_fc1 = nn.Sequential(
            nn.Linear(512*self.del1_size*self.del1_size, 1000),
            nn.Tanh(),
            nn.BatchNorm1d(1000)
        )
        self.latent_layer = nn.Linear(1000, self.latent_dim)
        
    def forward(self, x):

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

        return z

        
class Decoder(nn.Module):
    def __init__(self, IMAGE_SIZE):
        self.image_dim = IMAGE_SIZE # a 28x28 image corresponds to 4 on the FC layer, a 64x64 image corresponds to 13
                            # can calculate this using output_after_conv() in utils.py
        self.latent_dim = 200
        self.noise_scale = 0
        self.batch_size = 50
        
        self.del1_size = int(IMAGE_SIZE / 32)
        self.del2_size = int(IMAGE_SIZE / 16)
        super(Decoder, self).__init__()
        self.decoder_fc1 = nn.Sequential(
            nn.Linear(self.latent_dim, 1000),
            nn.Tanh(),
            nn.BatchNorm1d(1000)
        )
        self.decoder_fc2 = nn.Sequential(
            nn.Linear(1000, 512*self.del1_size*self.del1_size),
            nn.Tanh(),
            nn.BatchNorm1d(512*self.del1_size*self.del1_size)
        )
        self.decoder_l1 = nn.Sequential(
            nn.ConvTranspose2d(513, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            #nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),
            #nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1),
            #nn.MaxUnpool2d(3, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        #self.m1 = nn.MaxPool2d(3, stride=2)
        self.decoder_l2 = nn.Sequential(
            nn.ConvTranspose2d(257, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            #nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),
            #nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
            #nn.MaxUnpool2d(3, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        #self.m1 = nn.MaxPool2d(3, stride=2)
        self.decoder_l3 = nn.Sequential(
            nn.ConvTranspose2d(129, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            #nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            #nn.MaxUnpool2d(3, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        #self.m1 = nn.MaxPool2d(3, stride=2)
        self.decoder_l4 = nn.Sequential(
            nn.ConvTranspose2d(65, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32)
            #nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),
            #nn.MaxUnpool2d(3, stride=2)
        )
        self.decoder_l5 = nn.Sequential(
            nn.ConvTranspose2d(33, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        
    def forward(self, z, mask):
        x_hat = self.decoder_fc1(z)
        x_hat = self.decoder_fc2(x_hat)
        x_hat = x_hat.view(-1, 512, self.del1_size, self.del1_size)
        m1 = nn.MaxPool2d(32, stride = 32)
        mask_l1 = m1(mask)
        #print(mask_l1.shape)
        #print(x_hat.shape)
        #print(type(x_hat), type(mask_l1))
        x_hat = torch.cat((x_hat, mask_l1), dim = 1)
        #print(x_hat.shape)
        x_hat = self.decoder_l1(x_hat)
        #print(x_hat.shape)
        m2 = nn.MaxPool2d(16, stride = 16)
        mask_l2 = m2(mask)
        x_hat = torch.cat((x_hat, mask_l2), dim = 1)
        #print(mask_l2.shape)
        x_hat = self.decoder_l2(x_hat)
        
        m3 = nn.MaxPool2d(8, stride = 8)
        mask_l3 = m3(mask)
        x_hat = torch.cat((x_hat, mask_l3), dim = 1)
        #print(x_hat.shape)
        x_hat = self.decoder_l3(x_hat)
        
        m4 = nn.MaxPool2d(4, stride = 4)
        mask_l4 = m4(mask)
        x_hat = torch.cat((x_hat, mask_l4), dim = 1)
        #print(x_hat.shape)
        x_hat = self.decoder_l4(x_hat)
        
        m5 = nn.MaxPool2d(2, stride = 2)
        mask_l5 = m5(mask)
        x_hat = torch.cat((x_hat, mask_l5), dim = 1)
        #print(x_hat.shape)
        x_hat = self.decoder_l5(x_hat)

        return x_hat
    
class AE_VGG16_2(nn.Module):
    def __init__(self, IMAGE_SIZE):
        super(AE_VGG16_2, self).__init__()

        self.encoder = Encoder(IMAGE_SIZE)
        self.decoder = Decoder(IMAGE_SIZE)
        
        
    def forward(self, x, mask):
        z = self.encoder(x)
        x_hat = self.decoder(z, mask)
        return z, x_hat

    def encode(self, x):
        #x = x.unsqueeze(0)
        z, x_hat = self.forward(x)

        return z