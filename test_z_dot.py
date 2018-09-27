from disVAE import FullQDisentangledVAE
import torch
import torchvision
vae = FullQDisentangledVAE(frames=8, f_dim=64, z_dim=32, hidden_dim=512, conv_dim=1024)
device = torch.device('cuda:0')
vae.to(device)
checkpoint = torch.load('disentangled-vae.model')
vae.load_state_dict(checkpoint['state_dict'])
vae.eval()

for imageset in ('set1', 'set2', 'set3'):
    print(imageset)
    path = './test/style-transfer/'+imageset+'/'
    image1 = torch.load(path + 'image1.sprite')
    image2 = torch.load(path + 'image2.sprite')
    image1 = image1.to(device)
    image2 = image2.to(device)
    image1 = torch.unsqueeze(image1,0)
    image2= torch.unsqueeze(image2,0)
    with torch.no_grad():
        conv1 = vae.encode_frames(image1)
        conv2 = vae.encode_frames(image2)

        _,_,image1_f = vae.encode_f(conv1)
        image1_f_expand = image1_f.unsqueeze(1).expand(-1, vae.frames, vae.f_dim)
        _,_,image1_z = vae.encode_z(conv1,image1_f)

        _,_,image2_f = vae.encode_f(conv2)
        image2_f_expand = image2_f.unsqueeze(1).expand(-1,vae.frames,vae.f_dim)
        _,_,image2_z = vae.encode_z(conv2,image2_f)
        
        print(image2_z.shape)



