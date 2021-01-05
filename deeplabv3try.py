import torch
import torchvision.models as models
deeplabv3=models.segmentation.deeplabv3_resnet101(pretrained=True)
print ('good')
model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
model.eval()



print('finished')