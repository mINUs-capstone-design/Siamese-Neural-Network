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


def compute_accuracy(model, test_loader, threshold=1.0): # margin 1.0 -> 0.74
    correct = 0
    total = 0

    model.eval()  # 평가 모드로 설정

    with torch.no_grad():
        for data1, data2, labels in test_loader:

            # data1, data2, labels = data1.cuda(), data2.cuda(), labels.cuda()
            output1, output2 = model(data1, data2)

            euclidean_distance = F.pairwise_distance(output1, output2)
            predicted = (euclidean_distance > threshold).float()  # 유사성을 확인하기 위한 임계값 적용
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    accuracy *= 100
    return accuracy

result = 0
for i in range(0,100):
  acc = int(compute_accuracy(net, test_dataloader))
  result += acc
print("accuracy : {}%".format(result/100))
