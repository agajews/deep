from torchvision import transforms
import torchsample
from scipy.misc import imread
import matplotlib.pyplot as plt
from PIL import Image

tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    # transforms.RandomResizedCrop(224),
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.2),
    transforms.ToTensor(),
    torchsample.transforms.RandomAffine(
        rotation_range=20,
        translation_range=(0.2, 0.2),
        zoom_range=(0.7, 1.0),
        shear_range=20),
    # torchsample.transforms.RandomChoiceBlur([0.1]),
    transforms.ToPILImage()
])

img = Image.fromarray(imread('/mnt/data/pigs/imgs/0/1.jpg'))
# plt.imshow(img)
# plt.show()
while True:
    plt.imshow(tf(img))
    plt.show()
