# Nhận Biết Độ Chín Trái Cây (Fruit Ripeness Classifier)

Hệ thống phân loại mức độ chín của trái cây sử dụng mạng **MobileNetV2** (Transfer Learning) được huấn luyện bằng PyTorch.

## 🍉 12 Lớp phân loại
Mô hình phân loại 3 loại quả phổ biến với 4 mức độ chín:
- **Chuối (Banana):** Xanh (`banana_green`), Ươm (`banana_semi_ripe`), Chín (`banana_ripe`), Quá chín/Hỏng (`banana_damaged`)
- **Xoài (Mango):** Xanh (`mango_green`), Ươm (`mango_semi_ripe`), Chín (`mango_ripe`), Quá chín/Hỏng (`mango_damaged`)
- **Cà chua (Tomato):** Xanh (`tomato_green`), Ươm (`tomato_semi_ripe`), Chín (`tomato_ripe`), Quá chín/Hỏng (`tomato_damaged`)

---

## 📁 Cấu trúc thư mục

```
fruit-ripeness-classifier/
├── dataset/                    # Thư mục dataset (tự động chia train / val)
│   ├── train/
│   └── val/
├── best_model.pth              # Checkpoint mô hình tốt nhất sau khi train
├── prepare_dataset.py          # Script tải & tổ chức dữ liệu từ Kaggle
├── train.py                    # Script huấn luyện MobileNetV2
├── requirements.txt            # Thư viện cần cài đặt
└── README.md
```

---

## 🚀 Cài đặt & Sử dụng

### 1. Cài đặt môi trường
```bash
pip install -r requirements.txt
```

### 2. Chuẩn bị Dataset
Đặt API key `kaggle.json` vào `~/.kaggle/kaggle.json`, sau đó chạy:
```bash
python prepare_dataset.py
```

### 3. Huấn luyện mô hình
```bash
python train.py
```

Mô hình tốt nhất sẽ tự động được lưu tại `best_model.pth`.

---

## 🧠 Load mô hình dự đoán

```python
import torch
from torchvision import models

model = models.mobilenet_v2()
model.classifier[1] = torch.nn.Linear(1280, 12)

checkpoint = torch.load("best_model.pth", map_location="cpu")
model.load_state_dict(checkpoint["state_dict"])
model.eval()
print(f"Val Accuracy: {checkpoint.get('val_acc', 0):.4f}")
```
