import json
import torch
from torch import nn
from torch.nn import Softmax

from processor import ImageProcessor
from torchvision.models import vgg16

class ILSVRCPredictor:
    def __init__(self, class_index):
        self.class_index = class_index

    def predict(self, logits):
        # softmax를 적용하여 확률을 계산
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        # 가장 확률이 높은 클래스 인덱스
        class_idx = torch.argmax(probabilities, dim=1).item()
        # 클래스 인덱스를 클래스 이름으로 변환
        class_name = self.class_index[str(class_idx)]
        return class_name, probabilities[0, class_idx].item()

class CustomVgg16(nn.Module):
    def __init__(self, transforms=ImageProcessor()):
        super().__init__()
        self.vgg16_model = vgg16(pretrained=True)
        self.transforms = transforms
        self.softmax = nn.Softmax(dim=1)

    def forward(self, img):
        self.vgg16_model.eval()
        img_transform = self.transforms(img).unsqueeze(0)
        out = self.vgg16_model(img_transform)
        probabilities = self.softmax(out)
        class_idx = str(torch.argmax(probabilities, dim=1).item())

        return class_idx