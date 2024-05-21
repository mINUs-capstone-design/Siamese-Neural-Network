import torchvision.transforms as transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
    
def getScore(dissimilarity):
  
  if dissimilarity >= 2.0:
    score = 0
  else:
    score = 100 - dissimilarity*50
    score = round(score)
  
  return score
  
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
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

# 모델 이름 경로
model = torch.load("./model/siamese_net_v4.pt", map_location=device)


# 비교 할 이미지들의 경로
x0 = Image.open("kr/Mel_spectrum_VAD_KsponSpeech_000098.jpg")
x1 = Image.open("data/testing/korean/man.jpg")

convert_tensor = transforms.Compose([transforms.Resize((99,250)),transforms.ToTensor()])
x0 = convert_tensor(x0).unsqueeze(0)
x1 = convert_tensor(x1).unsqueeze(0)

output1, output2 = model(x0, x1)
euclidean_distance = F.pairwise_distance(output1, output2)
print(f"score : {getScore(euclidean_distance.item())}")
    
    
