from torchvision import transforms
import torchsample
from scipy.misc import imread
import matplotlib.pyplot as plt
from PIL import Image
import os

tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    # transforms.RandomResizedCrop(224),
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.2),
    transforms.ToTensor(),
    torchsample.transforms.RandomAffine(
        rotation_range=30,
        translation_range=(0.4, 0.4),
        zoom_range=(0.5, 1.5),
        shear_range=20),
    # torchsample.transforms.RandomChoiceBlur([0.1]),
    transforms.ToPILImage()
])

base = '/Users/alexgajewski/Downloads/pigs_1'
for fnm in os.listdir(base):
    img = Image.fromarray(imread(os.path.join(base, fnm)))
    # plt.imshow(img)
    # plt.show()
    plt.imshow(tf(img))
    plt.show()
