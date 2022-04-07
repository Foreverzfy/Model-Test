import torch
import torchvision

device = "cuda" #if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

dummy_input = torch.randn(1, 3, 224, 224, device=device)
model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True).to(device)
#model = torch.hub.load('PingoLH/Pytorch-HarDNet', 'hardnet68', pretrained=True).to(device) #68 85 68ds 39ds
#model = torch.hub.load('rwightman/pytorch-dpn-pretrained', 'dpn131', pretrained=True).to(device) #68 92 98 107 131
model.load_state_dict(torch.load('vgg11.pth')) ####
model.eval()


input_names = ["input_1"]
output_names = ["output_1"]

torch.onnx.export(model, 
                  dummy_input, 
                  "vgg11.onnx", ###
                  verbose=True, 
                  input_names=input_names, 
                  output_names=output_names,
                  dynamic_axes={"input" : {0 : "batch_size"},
                                "output" : {0 : "batch_size"}})