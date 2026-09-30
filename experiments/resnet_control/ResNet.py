import torch
import urllib
import torchvision

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

url, filename = () #load the data from the data set
## Examples from pytorch documentation to load a file from a url
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)


# Execution
preprocess = transforms.Compose([
    
])
