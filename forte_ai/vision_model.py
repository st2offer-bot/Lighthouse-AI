import cv2
import torch

from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)

# 1. Выбираем GPU MacBook
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

# 2. Загружаем предобученную модель
weights = MobileNet_V3_Small_Weights.DEFAULT

model = mobilenet_v3_small(weights=weights)
model = model.to(device)
model.eval()

# 3. Открываем видео
cap = cv2.VideoCapture("X-1.mov")

if not cap.isOpened():
    raise RuntimeError("Cannot open X-1.mov")

# 4. Переходим к 22-й секунде
cap.set(cv2.CAP_PROP_POS_MSEC, 22 * 1000)

success, frame = cap.read()
cap.release()

if not success:
    raise RuntimeError("Cannot read frame")

# 5. Преобразуем BGR -> RGB
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# 6. Подготавливаем изображение для MobileNet
image = weights.transforms()( 
    torch.from_numpy(frame).permute(2, 0, 1)
)

image = image.unsqueeze(0).to(device)

# 7. Запускаем нейросеть
with torch.inference_mode():
    output = model(image)

# 8. Показываем результаты
print("Device:", device)
print("Frame time: 22 seconds")
print("Input shape:", image.shape)
print("Output shape:", output.shape)