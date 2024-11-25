import os
import re
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
from torchvision import transforms
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import time
import numpy as np
from sklearn.model_selection import KFold

# 데이터셋 폴더 경로 설정
data_dir = r"C:\computervision\CUB_200_2011\CUB_200_2011\images"

# 사용할 클래스와 클래스 ID 정의
selected_classes = {
    "House_Sparrow": 146,
    "Chipping_Sparrow": 145,
    "Song_Sparrow": 148,
    "Tree_Sparrow": 150,
    "Field_Sparrow": 147,
    "American_Crow": 122,
    "Fish_Crow": 124,
    "California_Gull": 72,
    "Western_Gull": 79,
    "Herring_Gull": 74,
    "Ring_billed_Gull": 77,
}

# 이미지 경로와 클래스 라벨 로드
image_paths = []
label_text = []
for root, _, files in os.walk(data_dir):
    for file in files:
        if file.endswith(".jpg"):
            try:
                class_id = int(re.findall(r'\d+', os.path.basename(root))[0])
                if class_id in selected_classes.values():
                    image_paths.append(os.path.join(root, file))
                    label_text.append(class_id)
            except IndexError:
                print(f"Warning: Could not find class_id in directory name {root}")

# 라벨을 0부터 시작하도록 변환
label_map = {class_id: idx for idx, class_id in enumerate(selected_classes.values())}
label_text = [label_map[class_id] for class_id in label_text]

# 이미지와 텍스트 라벨을 리스트 형태로 저장
data = list(zip(image_paths, label_text))
print(f"Total samples: {len(data)}")

# CLIP 모델과 프로세서 로드
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 추가 분류 레이어 정의 (더 깊은 분류기 사용)
class CLIPWithClassifier(nn.Module):
    def __init__(self, clip_model, num_classes=len(selected_classes)):
        super(CLIPWithClassifier, self).__init__()
        self.clip = clip_model
        self.fc1 = nn.Linear(self.clip.config.projection_dim, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, images, input_ids, attention_mask):
        image_outputs = self.clip.get_image_features(pixel_values=images)
        x = self.fc1(image_outputs)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)

# 데이터 증강 및 이미지 전처리 함수 (강화된 증강)
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
])

# 전처리 함수
def preprocess_data(image_path, label_text):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    
    # 텍스트 전처리
    text_input = processor(text=[str(label_text)], return_tensors="pt", padding=True, truncation=True)
    return image, text_input

# 데이터셋 클래스 정의
class BirdDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label_text = self.data[idx]
        image, text_input = preprocess_data(image_path, label_text)
        return image, text_input, label_text

# collate_fn 정의
def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    max_input_len = max([item[1]["input_ids"].size(1) for item in batch])
    input_ids = torch.stack([torch.cat([item[1]["input_ids"], torch.zeros(1, max_input_len - item[1]["input_ids"].size(1))], dim=1) for item in batch]).squeeze(1)
    attention_mask = torch.stack([torch.cat([item[1]["attention_mask"], torch.zeros(1, max_input_len - item[1]["attention_mask"].size(1))], dim=1) for item in batch]).squeeze(1)
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
    return images, {"input_ids": input_ids.long(), "attention_mask": attention_mask.long()}, labels

# 검증 함수 정의
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, text_inputs, labels in val_loader:
            images = images.to(device)
            input_ids = text_inputs["input_ids"].to(device)
            attention_mask = text_inputs["attention_mask"].to(device)
            labels = labels.to(device)
            
            outputs = model(images=images, input_ids=input_ids, attention_mask=attention_mask)
            val_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_accuracy = 100 * correct / total
    return val_loss / len(val_loader), val_accuracy

# 5-fold 교차 검증 설정
k_folds = 5
kfold = KFold(n_splits=k_folds, shuffle=True)
dataset = BirdDataset(data)

# fold_results와 best_val_accuracies 리스트 초기화
fold_results = []
best_val_accuracies = []

# 데이터셋을 5개의 폴드로 나누기
all_splits = list(kfold.split(dataset))

# 2개는 학습, 2개는 검증, 1개는 테스트용으로 사용
for fold in range(k_folds):
    print(f'FOLD {fold+1}')
    print('--------------------------------')
    
    # 현재 폴드를 테스트 세트로 사용
    test_ids = all_splits[fold][1]
    
    # 나머지 4개의 폴드에서 2개는 학습, 2개는 검증용으로 사용
    remaining_folds = list(range(k_folds))
    remaining_folds.remove(fold)
    
    train_ids = []
    val_ids = []
    
    # 남은 4개의 폴드 중 2개는 학습, 2개는 검증용으로 분할
    for i, remaining_fold in enumerate(remaining_folds):
        if i < 2:  # 처음 2개는 학습용
            train_ids.extend(all_splits[remaining_fold][1])
        else:  # 나머지 2개는 검증용
            val_ids.extend(all_splits[remaining_fold][1])
    
    # 데이터로더 설정
    train_subsampler = SubsetRandomSampler(train_ids)
    val_subsampler = SubsetRandomSampler(val_ids)
    test_subsampler = SubsetRandomSampler(test_ids)
    
    train_loader = DataLoader(dataset, batch_size=64, sampler=train_subsampler, collate_fn=collate_fn)
    val_loader = DataLoader(dataset, batch_size=64, sampler=val_subsampler, collate_fn=collate_fn)
    test_loader = DataLoader(dataset, batch_size=64, sampler=test_subsampler, collate_fn=collate_fn)
    
    # 모델 초기화
    model = CLIPWithClassifier(CLIPModel.from_pretrained("openai/clip-vit-base-patch32"), num_classes=len(selected_classes))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 전이 학습 설정
    for name, param in model.clip.named_parameters():
        if "visual.transformer" in name and "layer.11" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    # 옵티마이저와 손실 함수 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-5, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Early stopping 파라미터
    best_val_accuracy = 0.0
    epochs_no_improve = 0
    patience = 15
    model_save_path = f"best_clip_model_fold_{fold+1}.pth"
    
    # 학습 시작
    for epoch in range(100):
        model.train()
        total_loss = 0
        
        for batch_idx, (images, text_inputs, labels) in enumerate(train_loader):
            images = images.to(device)
            input_ids = text_inputs["input_ids"].to(device)
            attention_mask = text_inputs["attention_mask"].to(device)
            labels = labels.to(device)
            
            outputs = model(images=images, input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 검증
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Fold {fold+1}, Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            print("BestValAccuracySaved\n")
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print("Early stopping triggered")
            break
    
    # 최종 테스트 성능 평가
    model.load_state_dict(torch.load(model_save_path))
    test_loss, test_accuracy = validate(model, test_loader, criterion, device)
    print(f'Fold {fold+1} Test Accuracy: {test_accuracy:.2f}%')
    print(f'Fold {fold+1} Best Validation Accuracy : {best_val_accuracy:.2f}%')
    fold_results.append(test_accuracy)  # 테스트 정확도 저장
    best_val_accuracies.append(best_val_accuracy)  # 최고 검증 정확도 저장

# 결과 출력
print("\nResults for all folds:")
for fold in range(len(fold_results)):
    print(f'Fold {fold+1}: Test Accuracy = {fold_results[fold]:.2f}%, Best Validation Accuracy = {best_val_accuracies[fold]:.2f}%')