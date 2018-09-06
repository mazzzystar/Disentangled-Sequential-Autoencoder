from disVAE import FullQDisentangledVAE
import torch
import torchvision
vae = FullQDisentangledVAE(frames=8,f_dim=256,z_dim=32,hidden_dim=512,conv_dim=1024)
device = torch.device('cuda:0')
vae.to(device)
vae.train()
checkpoints = torch.load('disentangled-vae.model')
vae.load_state_dict(checkpoints['state_dict'])

image1 = torch.load('image1.sprite')
image2 = torch.load('image2.sprite')
image1 = image1.to(device)
image2 = image2.to(device)
image1 = torch.unsqueeze(image1,0)
image2= torch.unsqueeze(image2,0)
vae.eval()
conv1 = vae.encode_frames(image1)
conv2 = vae.encode_frames(image2)

_,_,image1_f = vae.encode_f(conv1)
image1_f_expand = image1_f.unsqueeze(1).expand(-1,vae.frames,vae.f_dim)
_,_,image1_z = vae.encode_z(conv1,image1_f)

_,_,image2_f = vae.encode_f(conv2)
image2_f_expand = image2_f.unsqueeze(1).expand(-1,vae.frames,vae.f_dim)
_,_,image2_z = vae.encode_z(conv2,image2_f)

image1swap_zf = torch.cat((image2_z,image1_f_expand),dim=2)
image1_body_image2_motion = vae.decode_frames(image1swap_zf)
image1_body_image2_motion = torch.squeeze(image1_body_image2_motion,0)

image2swap_zf = torch.cat((image1_z,image2_f_expand),dim=2)
image2_body_image1_motion = vae.decode_frames(image1swap_zf)
image2_body_image1_motion = torch.squeeze(image2_body_image1_motion,0)

torchvision.utils.save_image(image1_body_image2_motion,'image1_body_image2_motion.png')
torchvision.utils.save_image(image2_body_image1_motion,'image2_body_image1_motion.png')





