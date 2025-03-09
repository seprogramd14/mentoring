from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models import vgg16
from custom_dataset import CustomDataset
from processor import ImageProcessor

# 1. 학습시킬 데이터 불러오기
# dataset = CustomDataset("", "", ImageProcessor())
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# 2. 전이학습을 시킬 모델 불러오기
vgg = vgg16(pretrained=True)
vgg.features[6] = nn.Linear(in_features=4096, out_features=10, bias=True) # out 개수 수정 필요
vgg.train()

# 3. 손실함수 설정하기
criterion = nn.CrossEntropyLoss()
opt = optim.Adam(vgg.classifier.parameters(), lr=0.001)

# 4. 학습 루프 구성하기
num_epoch = 100
