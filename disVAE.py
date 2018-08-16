import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Sprites(torch.utils.data.Dataset):


class FullQDisentangledVAE(nn.Module):
        super(DisentangledVAE,self).__init__(self)
        self.f_dim = params['f_dim']
        self.z_dim = params['z_dim']
        self.frames = params['frames']
        self.conv_dim = params['conv_dim']
        self.hidden_dim = params['hidden_dim']
        self.conv_params = params['conv']
        self.deconv_params = params['deconv']
        
        self.f_lstm = nn.LSTM(self.conv_dim, self.hidden_dim, 1,
                bidirectional=True,batch_first=True)
        self.f_mean = nn.Linear(self.hidden_dim*2, self.f_dim)
        self.f_logvar = nn.Linear(self.hidden_dim*2, self.f_dim)

        self.z_lstm = nn.LSTM(self.conv_dim+self.f_dim, self.hidden_dim, 1,
                 bidirectional=True,batch_first=True)
        self.z_rnn = nn.RNN(self.hidden_dim*2, self.hidden_dim,batch_first=True) 
        self.z_mean = nn.Linear(self.hidden_dim, self.z_dim)
        self.z_logvar = nn.Linear(self.hidden_dim, self.z_dim)
        
        self.conv1 = nn.Conv2d(3,256,kernel_size=3,stride=2,padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256,256,kernel_size=3,stride=2,padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256.256.kernel_size=3,stride=2,padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256,256,kernel_size=3,stride=2,padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv_fc = nn.Linear(4*4*256,self.conv_dim)
        self.bnf = nn.BatchNorm1d(self.conv_dim) 

        self.deconv_fc = nn.Linear(self.f_dim+self.z_dim,4*4*256)
        self.deconv_bnf = nn.BatchNorm1d(4*4*256)
        self.deconv4 = nn.ConvTranspose2d(256,256,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.dbn4 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256,256,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.dbn3 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256,256,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.dbn2 = nn.BatchNorm2d(256)
        self.deconv1 = nn.ConvTranspose2d(256,3,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.dbn1 = nn.BatchNorm2d(3)
        
    def encode_frames(self,x):
        x = x.view(-1,3,64,64) #Batchwise stack the 8 images for applying convolutions parallely
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(-1,4*4*256)
        x = F.relu(self.bnf(self.conv_fc(x))) 
        return x.view(-1,self.frames,self.conv_dim) #Convert the stack batches back into frames

    def decode_frames(self,zf):
        x = F.relu(self.deconv_bnf(self.deconv_fc(zf)))
        x = x.view(-1,256,4,4) #The 8 frames are stacked batchwise
        x = F.relu(self.dbn4(self.deconv4(x)))
        x = F.relu(self.dbn3(self.deconv3(x)))
        x = F.relu(self.dbn2(self.deconv2(x)))
        x = F.relu(self.dbn1(self.deconv1(x))) #If images are in 0,1 range use ReLU otherwise use tanh
        return x.view(-1.self.frames,3,64,64) #Convert the stacked batches back into frames

    def reparameterize(self,mean,logvar):
        if self.training:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5*logvar)
            z = mean + eps*std
            return z
        else:
            return mean

    def encode_f(self,x):
        lstm_out,_ = self.f_lstm(x)
        mean = self.f_mean(lstm_out[:,self.frames-1]) #The forward and the reverse are already concatenated
        logvar = self.f_logvar(lstm_out[:,self.frames-1]) # TODO: Check if its the correct forward and reverse
        return mean,logvar,self.reparameterize(mean,logvar)
    
    def encode_z(self,x,f):
        f_expand = f.unsqueeze(1).expand(-1,self.frames,self.f_dim)
        lstm_out,_ = self.z_lstm(torch.cat((x, f_expand), dim=2))
        rnn_out,_ = self.z_rnn(lstm_out)
        mean = self.z_mean(rnn_out)
        logvar = self.z_logvar(rnn_out)
        return mean,logvar,self.reparameterize(mean,logvar)

    def forward(self,x):
        conv_x = self.encode_frames(x)
        f_mean,f_logvar,f = self.encode_f(conv_x)
        z_mean,z_logvar,z = self.encode_z(conv_x,f)
        f_expand = f.unsqueeze(1).expand(-1,self.frames,self.f_dim)
        recon_x = self.decode_frames(torch.cat((z,f_expand),dim=2))
        return f_mean,f_logvar,f,z_mean,z_logvar,z,recon_x

def loss_fn(original_seq,recon_seq,f_mean,f_logvar,z_mean,z_logvar):
    mse = F.mse_loss(recon_seq,original_seq,size_average=False);
    kld_f = -0.5 * torch.sum(1 + f_logvar - torch.pow(f_mean,2) - torch.exp(f_logvar))
    kld_z = -0.5 * torch.sum(1 + z_logvar - torch.pow(z_mean,2) - torch.exp(z_logvar))
    return mse + kld_f + kld_z
  

#Necessary changes will be made to trainer after exact CNN architecture is finalised
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
        
    def save_checkpoint(self,epoch):
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
            print("No Checkpoint Exists At '{}'.Start Fresh Training".format(self.checkpoints))
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
               loss = kl_meansquare(data,recon_x,mean,logvar)
               loss.backward()
               self.optimizer.step()
               losses.append(loss.item())
           print(len(losses) == self.batch_size)
           print("Epoch {} : Average Loss: {}".format(epoch,np.mean(losses)))
           self.save_checkpoint(epoch) 
       print("Training is complete")
