# 이미지 전처리를 위한 클래스 생성
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