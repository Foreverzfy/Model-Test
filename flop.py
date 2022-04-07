import torchvision.models as models
import torch
from ptflops import get_model_complexity_info

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True).to(device)
#model = torch.hub.load('PingoLH/Pytorch-HarDNet', 'hardnet68', pretrained=True).to(device) #68 85 68ds 39ds
#model = torch.hub.load('rwightman/pytorch-dpn-pretrained', 'dpn131', pretrained=True).to(device) #68 92 98 107 131
model.eval()

with torch.cuda.device(0):
  net = model
  macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))