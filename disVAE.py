import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DisentangledVAE(nn.Module):
    def __init__(self,params):
        super(DisentangledVAE,self).__init__(self)
        self.f_dim = params['f_dim']
        self.z_dim = params['z_dim']
        self.input_dim = params['input_dim']
        self.conv_dim = params['conv_dim']
        self.hidden_dim = params['hidden_dim']
        self.conv = params['conv']
        self.deconv = params['deconv']
        
        self.f_lstm = nn.LSTM(self.conv_dim, self.hidden_dim, 1,
                bidirectional=True)
        self.f_mean = nn.Linear(self.hidden_dim*2, self.f_dim)
        self.f_logvar = nn.Linear(self.hidden_dim*2, self.f_dim)

        self.z_lstm = nn.LSTM(self.conv_dim+self.f_dim, self.hidden_dim, 1,
                 bidirectional=True)
        self.z_rnn = nn.RNN(self.hidden_dim*2, self.hidden_dim) 
        self.z_mean = nn.Linear(self.hidden_dim, self.z_dim)
        self.z_logvar = nn.Linear(self.hidden_dim, self.z_dim)

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
        seq_len = lstm_out.size(0)
        mean = self.f_mean(lstm_out[seq_len])
        logvar = self.f_logvar(lstm_out[seq_len])
        return mean,logvar,self.reparameterize(mean,logvar)
    
    def encode_z(self,x,f):
        f_expand = f.expand(x.size(0),f.size(0),f.size(1))
        lstm_out,_ = self.z_lstm(torch.cat((x, f_expand), dim=2))
        rnn_out,_ = self.z_rnn(lstm_out)
        mean = self.z_mean(rnn_out)
        logvar = self.z_logvar(rnn_out)
        return mean,logvar,self.reparameterize(mean,logvar)

    def forward(self,x):
        conv_x = self.conv(x)
        f_mean,f_logvar,f = self.encode_f(conv_x)
        z_mean,z_logvar,z = self.encode_z(conv_x,f)
        f_expand = f.expand(z.size(0),f.size(0),f.size(1))
        recon_x = self.deconv(torch.cat((z,f_expand),dim=2))
        return f_mean,f_logvar,f,z_mean,z_logvar,z,recon_x

def loss_fn(original,recon,f_mean,f_logvar,z_mean,z_logvar,unroll_size):
    mse = F.mse_loss(recon.view(-1,unroll_size),original.view(-1,unroll_size),size_average=False)
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
