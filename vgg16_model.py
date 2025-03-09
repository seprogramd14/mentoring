import json
import torch
from torch import nn
from torchvision.models import vgg16

from torchvision import transforms

class ImageProcessor: # TIL 정리
    def __init__(self, size=(224, 224), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img):
        return self.transform(img)

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
        ilsvrc_class_index = json.load(open('./imagenet_class_index.json', 'r'))
        self.predictor = ILSVRCPredictor(ilsvrc_class_index)

    def forward(self, img):
        self.vgg16_model.eval()
        img_transform = self.transforms(img).unsqueeze(0)
        out = self.vgg16_model(img_transform)

        result = self.predictor.predict(out)
        return result