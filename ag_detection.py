import torch
import torch.autograd.variable as Variable
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as utils
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from PIL import Image
from shutil import copyfile

transforms_list = [
    transforms.Resize(256),
    transforms.CenterCrop(227),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.RandomCrop(227)
]

transforms_dict = {
    'train': {
        0: list(transforms_list[i] for i in [0, 1, 3]),
        1: list(transforms_list[i] for i in [0, 1, 2, 3]),
        2: list(transforms_list[i] for i in [0, 4, 2, 3])
    },
    'val': {
        0: list(transforms_list[i] for i in [0, 1, 3])
    },
    'test': {
        0: list(transforms_list[i] for i in [0, 1, 3])
    }
}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

PATH_TO_MODELS = "/content/gdrive/My Drive/age-and-gender-classification/models"

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 96, 7, stride = 4, padding = 1)
        self.pool1 = nn.MaxPool2d(3, stride = 2, padding = 1)
        self.norm1 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv2 = nn.Conv2d(96, 256, 5, stride = 1, padding = 2)
        self.pool2 = nn.MaxPool2d(3, stride = 2, padding = 1)
        self.norm2 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv3 = nn.Conv2d(256, 384, 3, stride = 1, padding = 1)
        self.pool3 = nn.MaxPool2d(3, stride = 2, padding = 1)
        self.norm3 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.fc1 = nn.Linear(18816, 512)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 512)
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(512, 10)

        self.apply(weights_init)


    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.pool1(x)
        x = self.norm1(x)

        x = F.leaky_relu(self.conv2(x))
        x = self.pool2(x)
        x = self.norm2(x)

        x = F.leaky_relu(self.conv3(x))
        x = self.pool3(x)
        x = self.norm3(x)

        x = x.view(-1, 18816)

        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.dropout2(x)

        x = F.log_softmax(self.fc3(x), dim=1)

        return x


def weights_init(m):
  if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    nn.init.normal_(m.weight, mean=0, std=1e-2)

PATH_TO_AGE_MODEL = f'{PATH_TO_MODELS}/age.pt'
PATH_TO_GENDER_MODEL = f'{PATH_TO_MODELS}/gender.pt'

mapping = {
    0: '0-2 years',
    1: '4-6 years',
    2: '8-13 years',
    3: '15-20 years',
    4: '25-32 years',
    5: '38-43 years',
    6: '48-53 years',
    7: '60 years and above',
    8: 'male',
    9: 'female'
}

def test_on_a_class(c, image_tensor):
    with torch.no_grad():
        net = Net().to(device)
        net.load_state_dict(torch.load(f'{PATH_TO_MODELS}/{c}.pt'))
        net.eval()
        output = net(image_tensor)
        output = torch.max(output, 1)[1].to(device)
        result = f'{c} = {mapping[output.item()]}'

    return result

def test(path):
    image = Image.open(path)
    plt.imshow(image)
    image = transforms.Compose(transforms_dict['test'][0])(image)
    image.unsqueeze_(0)
    image = image.to(device)
    print(test_on_a_class('age', image))
    print(test_on_a_class('gender', image))
