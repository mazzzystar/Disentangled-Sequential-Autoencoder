import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


class VAE(nn.Module):
    
    def __init__(self):
        super(VAE,self).__init__()
        self.latent_dim = 128 
        #Encoder architecture
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                stride=1,padding=1,bias=False) #3*32*32 to 16*32*32
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16,16,kernel_size=3,
                stride=1,padding=1,bias=False) #16*32*32 to 16*32*32
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16,32,kernel_size=3,
                stride=2,padding=1,bias=False) #16*32*32 to 32*16*16
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32,32,kernel_size=3,
                stride=1,padding=1,bias=False) #32*16*16 to 32*16*16
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32,64,kernel_size=3,
                stride=2,padding=1,bias=False) #32*16*16 to 64*8*8
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64,64,kernel_size=3,
                stride=1,padding=1,bias=False) #64*8*8 to 64*8*8
        self.bn6 = nn.BatchNorm2d(64)
        self.conv7 = nn.Conv2d(64,64,kernel_size=3,
                stride=2,padding=1,bias=False) #64*8*8 to 64*4*4
        self.bn7 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(4*4*64,512,bias=False)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512,128,bias=False)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.fc_mean = nn.Linear(128,128)
        self.fc_logvar = nn.Linear(128,128)
        
        #Decoder Architecture
        self.d_fc2 = nn.Linear(128,512,bias=False)
        self.d_bn_fc2 = nn.BatchNorm1d(512)
        self.d_fc1 = nn.Linear(512,4*4*64,bias=False)
        self.d_bn_fc1 = nn.BatchNorm1d(4*4*64)
        self.d_conv7 = nn.ConvTranspose2d(64,64,kernel_size=3,
                stride=2,padding=1,output_padding=1) # 64*4*4 to 64*8*8 
        self.d_bn7 = nn.BatchNorm2d(64)
        self.d_conv6 = nn.ConvTranspose2d(64,64,kernel_size=3,
                stride=1,padding=1)
        self.d_bn6 = nn.BatchNorm2d(64)
        self.d_conv5 = nn.ConvTranspose2d(64,32,kernel_size=3,
                stride=2,padding=1,output_padding=1) # 64*8*8 to 32*16*16
        self.d_bn5 = nn.BatchNorm2d(32)
        self.d_conv4 = nn.ConvTranspose2d(32,32,kernel_size=3,
                stride=1,padding=1)
        self.d_bn4 = nn.BatchNorm2d(32)
        self.d_conv3 = nn.ConvTranspose2d(32,16,kernel_size=3,
                stride=2,padding=1,output_padding=1)  
        self.d_bn3 = nn.BatchNorm2d(16)
        self.d_conv2 = nn.ConvTranspose2d(16,16,kernel_size=3,
                stride=1,padding=1)
        self.d_bn2 = nn.BatchNorm2d(16)
        self.d_conv1 = nn.ConvTranspose2d(16,3,kernel_size=3,
                stride=1,padding=1)
        self.d_bn1 = nn.BatchNorm2d(3)
        
        #Initialization
        for m in self.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.ConvTranspose2d) or isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
        nn.init.xavier_normal_(self.fc_mean.weight)
        nn.init.xavier_normal_(self.fc_logvar.weight)

    def encode(self,x):
        conv1 = F.relu(self.bn1(self.conv1(x)))
        conv2 = F.relu(self.bn2(self.conv2(conv1)))
        conv3 = F.relu(self.bn3(self.conv3(conv2)))
        conv4 = F.relu(self.bn4(self.conv4(conv3)))
        conv5 = F.relu(self.bn5(self.conv5(conv4)))
        conv6 = F.relu(self.bn6(self.conv6(conv5)))
        conv7 = F.relu(self.bn7(self.conv7(conv6)))
        fc1 = F.relu(self.bn_fc1(self.fc1(conv7.view(-1,4*4*64))))
        fc2 = F.relu(self.bn_fc2(self.fc2(fc1)))
        mean,logvar = self.fc_mean(fc2),self.fc_logvar(fc2)
        return mean,logvar
    
    def reparametrize(self,mean,logvar):
        if self.training:
            eps = torch.randn_like(logvar) #During testing, representation should be spread out
            std = torch.exp(0.5*logvar) #hence the name variational autoencoder
            z = mean + eps*std
            return z
        else:
            return mean #During testing you want to return the peak of the normal to minimise loss

    def decode(self,z):
        d_fc2 = F.relu(self.d_bn_fc2(self.d_fc2(z)))
        d_fc1 = F.relu(self.d_bn_fc1(self.d_fc1(d_fc2)))
        d_conv7 = F.relu(self.d_bn7(self.d_conv7(d_fc1.view(-1,64,4,4)))) 
        d_conv6 = F.relu(self.d_bn6(self.d_conv6(d_conv7)))
        d_conv5 = F.relu(self.d_bn5(self.d_conv5(d_conv6)))
        d_conv4 = F.relu(self.d_bn4(self.d_conv4(d_conv5)))
        d_conv3 = F.relu(self.d_bn3(self.d_conv3(d_conv4)))
        d_conv2 = F.relu(self.d_bn2(self.d_conv2(d_conv3)))
        d_conv1 = F.relu(self.d_bn1(self.d_conv1(d_conv2)))
        return d_conv1

    def forward(self,x):
        mean,logvar = self.encode(x)
        z = self.reparametrize(mean,logvar)
        return self.decode(z),mean,logvar

    def generate_random(self):
        z = torch.randn((1,self.latent_dim))
        return self.decode(z)

def kl_crossentropy(original,recon,mean,logvar):
    bce = F.binary_cross_entropy(recon.view(-1,32*32*3),original.view(-1,32*32*3))
    kld = -0.5 * torch.sum(1 + logvar - torch.pow(mu,2) - torch.exp(logvar))
    return bce+kld

def kl_meansquare(original,recon,mean,logvar):
    kld = -0.5 * torch.sum(1 + logvar - torch.pow(mu,2) - torch.exp(logvar))
    mse = F.mse_loss(recon.view(-1,32*32*3),original.view(-1,32*32*3))
    return mse+kld


class Trainer(object):
    def __init__(self,model,device,trainloader,testloader,epochs,batch_size,learning_rate,checkpoints):
        self.trainloader = trainloader
        self.testloader = testloader
        self.start_epoch = 0
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = model
        self.model.to(device)
        self.learning_rate = learning_rate
        self.checkpoints = checkpoints
        self.optimizer = optim.Adam(self.model.parameters(),self.learning_rate)
        
    def save_checkpoint(epoch):
        torch.save({
            'epoch' : epoch+1,
            'state_dict' : self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict()},
            self.checkpoints)
        
    def load_checkpoint(self):
        try:
            print("Loading Checkpoint from '{}'".format(self.checkpoints))
            checkpoint = torch.load(self.checkpoints)
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Resuming Training From Epoch {}".format(self.start_epoch))
        except:
            print("No Checkkkpoint Exists At '{}'.Start Fresh Training".format(self.checkpoints))
            self.start_epoch = 0

    def train(self):
       self.model.train()
       for epoch in range(self.start_epoch,self.epochs):
           losses = []
           print("Running Epoch : {}".format(epoch))
           for i,(data,_) in enumerate(self.trainloader):
               data = data.to(device)
               self.optimizer.zero_grad()
               #this part is VAE specific
               recon_x,mean,logvar = self.model(data)
               loss = kl_crossentropy(data,recon_x,mean,logvar)
               loss.backward()
               self.optimizer.step()
               losses.append(loss.item())
           print("Epoch {} : Average Loss: {}".format(epoch,np.mean(losses)))
           self.save_checkpoint(epoch) 
       print("Training is complete")

batch_size = 64
epochs = 40

preprocess = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.0,0.0,0.0),(1.0,1.0,1.0))])
trainset = torchvision.datasets.CIFAR10(root='./cifar10',train=True,download=True,transform=preprocess)
testset = torchvision.datasets.CIFAR10(root='./cifar10',train=False,download=True,transform=preprocess)
trainloader = torch.utils.data.DataLoader(trainset,batch_size,shuffle=True,num_workers=4)
testloader = torch.utils.data.DataLoader(trainset,batch_size,shuffle=True,num_workers=4)

vae = VAE()
device = torch.device('cuda')
trainer = Trainer(vae,device,trainloader,testloader,epochs,batch_size,0.001,'/cnn-vae-cifar.model')
trainer.load_checkpoint()
trainer.train()

