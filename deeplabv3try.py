import torch
import torchvision.models as models
import matplotlib.pyplot as plt
import time
base=time.time()
model=models.segmentation.deeplabv3_resnet101(pretrained=True)
model.eval()
print('{:.1f} model loaded'.format(time.time()-base))



from PIL import Image
from torchvision import transforms

input_image = Image.open('a.jpg')
input_image=input_image.resize((120,160))

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model


print('{:.1f} image processed'.format(time.time()-base))
with torch.no_grad():
    output = model(input_batch)['out'][0]

output_predictions = output.argmax(0)
print('{:.1f} model predicted'.format(time.time()-base))
# create a color pallette, selecting a color for each class
palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
colors = (colors % 255).numpy().astype("uint8")

# plot the semantic segmentation predictions of 21 classes in each color
r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
r.putpalette(colors)

plt.imshow(r)
plt.show()