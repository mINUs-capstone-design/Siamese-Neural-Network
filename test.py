import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

class Config():
    testing_dir = "/content/drive/MyDrive/testing" 

def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()

def compute_accuracy(model, test_loader, threshold=0.5): # margin 1.0 -> 0.74
    correct = 0
    total = 0

    model.eval()  # 평가 모드로 설정

    with torch.no_grad():
        for data1, data2, labels in test_loader:

            data1, data2, labels = data1.cuda(), data2.cuda(), labels.cuda()
            output1, output2 = model(data1, data2)

            euclidean_distance = F.pairwise_distance(output1, output2)
            predicted = (euclidean_distance > threshold).float()  # 유사성을 확인하기 위한 임계값 적용
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    accuracy *= 100
    return accuracy


class SiameseNetworkDataset(Dataset):

    def __init__(self,imageFolderDataset,transform=None,should_invert=False):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        
        should_get_same_class = random.randint(0,1)
        if should_get_same_class:
            while True:
                
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1]==img1_tuple[1]:
                    break
        else:
            while True:

                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] !=img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))
        # 두 이미지가 서로 다른 클래스면 1
        # 두 이미지가 서로 같은 클래스면 0

    def __len__(self):
        return len(self.imageFolderDataset.imgs)


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1), # 가장자리의 특징들까지 고려
            nn.Conv2d(3, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),


            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),


        )

        self.fc1 = nn.Sequential(
            nn.Linear(8*99*250, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1) # flatten
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load("siamese_net_v4.pt", map_location=device)
print(model)

folder_dataset_test = dset.ImageFolder(root=Config.testing_dir)
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                        transform=transforms.Compose([transforms.Resize((99,250)),
                                                                      transforms.ToTensor()
                                                                      ])
                                       ,should_invert=False)

test_dataloader = DataLoader(siamese_dataset,num_workers=1,batch_size=1,shuffle=True) # 1장씩 test


dataiter = iter(test_dataloader)
# x0,x1,label1 = next(dataiter) # test의 기준이 되는 img.
# 0 : same class , 1 : other class

for i in range(10):
    x0,x1,label1 = next(dataiter)
    concatenated = torch.cat((x0,x1),0)

    output1,output2 = model(Variable(x0).cuda(),Variable(x1).cuda())
    euclidean_distance = F.pairwise_distance(output1, output2)
    imshow(torchvision.utils.make_grid(concatenated),'isNotSame : {:.0f}\nDissimilarity: {:.2f}'.format(label1.item(),euclidean_distance.item()))




result = 0
for i in range(0,100):
  acc = int(compute_accuracy(model, test_dataloader))
  result += acc
print("accuracy : {}%".format(result/100))
