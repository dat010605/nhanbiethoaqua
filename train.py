"""
=============================================================
  HỆ THỐNG PHÂN LOẠI MỨC ĐỘ CHÍN TRÁI CÂY
  Mô hình: MobileNetV2 + Transfer Learning
  Framework: PyTorch + torchvision
  
  12 Lớp phân loại:
    - Chuối: Xanh, Ươm, Chín, Quá chín/Hư hỏng
    - Xoài:  Xanh, Ươm, Chín, Quá chín/Hư hỏng
    - Cà chua: Xanh, Ươm, Chín, Quá chín/Hư hỏng

  Cấu trúc thư mục dataset cần có:
    dataset/
    ├── train/
    │   ├── banana_green/
    │   ├── banana_semi_ripe/
    │   ├── banana_ripe/
    │   ├── banana_damaged/
    │   ├── mango_green/
    │   ├── mango_semi_ripe/
    │   ├── mango_ripe/
    │   ├── mango_damaged/
    │   ├── tomato_green/
    │   ├── tomato_semi_ripe/
    │   ├── tomato_ripe/
    │   └── tomato_damaged/
    └── val/
        └── (cấu trúc tương tự train/)
=============================================================
"""

import os
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# ─────────────────────────────────────────────
# 1. CẤU HÌNH CHUNG (Hyperparameters & Paths)
# ─────────────────────────────────────────────

# Thư mục chứa dataset
DATA_DIR = "./dataset"

# Thư mục lưu model tốt nhất
SAVE_PATH = "./best_model.pth"

# Số lớp phân loại (3 loại quả × 4 mức độ chín)
NUM_CLASSES = 12

# Kích thước ảnh đầu vào
IMG_SIZE = 224

# Kích thước batch
BATCH_SIZE = 32

# Số epoch huấn luyện
NUM_EPOCHS = 30

# Tốc độ học (learning rate)
LEARNING_RATE = 0.001

# Sử dụng GPU nếu có, ngược lại dùng CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Đang sử dụng thiết bị: {DEVICE}")


# ─────────────────────────────────────────────
# 2. DATA AUGMENTATION & TRANSFORMS
# ─────────────────────────────────────────────

# Giá trị mean và std chuẩn của ImageNet (dùng cho MobileNetV2 pretrained)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Transform cho tập TRAIN (có augmentation để tránh overfitting)
train_transforms = transforms.Compose([
    # Resize ảnh về kích thước lớn hơn rồi crop ngẫu nhiên
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),

    # Lật ngang ảnh ngẫu nhiên (50% xác suất)
    transforms.RandomHorizontalFlip(p=0.5),

    # Lật dọc ảnh ngẫu nhiên (30% xác suất)
    transforms.RandomVerticalFlip(p=0.3),

    # Xoay ảnh ngẫu nhiên trong khoảng [-30°, +30°]
    transforms.RandomRotation(degrees=30),

    # Điều chỉnh màu sắc ngẫu nhiên: độ sáng, độ tương phản, bão hòa, sắc độ
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.1
    ),

    # Chuyển ảnh sang Tensor và chuẩn hóa về [0, 1]
    transforms.ToTensor(),

    # Chuẩn hóa theo mean/std của ImageNet
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Transform cho tập VAL (không augmentation, chỉ resize + normalize)
val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ─────────────────────────────────────────────
# 3. LOAD DATASET & DATALOADER
# ─────────────────────────────────────────────

def create_dataloaders(data_dir, batch_size):
    """
    Tạo DataLoader cho tập train và validation.
    Sử dụng ImageFolder — tên thư mục con = tên class.

    Args:
        data_dir  (str): Đường dẫn đến thư mục dataset gốc.
        batch_size (int): Số ảnh trên mỗi batch.

    Returns:
        dataloaders (dict): {'train': ..., 'val': ...}
        dataset_sizes (dict): Số lượng ảnh mỗi tập.
        class_names (list): Danh sách tên các lớp.
    """
    # ImageFolder tự động gán nhãn dựa trên tên thư mục con
    image_datasets = {
        "train": datasets.ImageFolder(
            root=os.path.join(data_dir, "train"),
            transform=train_transforms
        ),
        "val": datasets.ImageFolder(
            root=os.path.join(data_dir, "val"),
            transform=val_transforms
        ),
    }

    dataloaders = {
        "train": DataLoader(
            image_datasets["train"],
            batch_size=batch_size,
            shuffle=True,          # Trộn dữ liệu mỗi epoch
            num_workers=4,         # Số luồng song song để load ảnh
            pin_memory=True        # Tăng tốc transfer lên GPU
        ),
        "val": DataLoader(
            image_datasets["val"],
            batch_size=batch_size,
            shuffle=False,         # Không cần shuffle khi validate
            num_workers=4,
            pin_memory=True
        ),
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    class_names = image_datasets["train"].classes

    print(f"[INFO] Số lớp: {len(class_names)}")
    print(f"[INFO] Tên các lớp: {class_names}")
    print(f"[INFO] Số ảnh train: {dataset_sizes['train']}")
    print(f"[INFO] Số ảnh val:   {dataset_sizes['val']}")

    return dataloaders, dataset_sizes, class_names


# ─────────────────────────────────────────────
# 4. KHỞI TẠO MÔ HÌNH MobileNetV2
# ─────────────────────────────────────────────

def build_model(num_classes):
    """
    Khởi tạo MobileNetV2 với trọng số pretrained từ ImageNet.
    Thay thế lớp classifier cuối cùng để phù hợp với bài toán.

    Args:
        num_classes (int): Số lớp cần phân loại.

    Returns:
        model: MobileNetV2 đã được chỉnh sửa.
    """
    # Tải MobileNetV2 với trọng số pretrained (ImageNet)
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # ── Fine-tuning chiến lược: đóng băng toàn bộ backbone ──
    # Các lớp features sẽ không được cập nhật trọng số (tiết kiệm thời gian & tránh overfitting)
    for param in model.features.parameters():
        param.requires_grad = False

    # ── Thay thế lớp Classifier cuối cùng ──
    # MobileNetV2 mặc định: Classifier[-1] = Linear(1280, 1000)
    # Ta thay bằng: Linear(1280, num_classes)
    in_features = model.classifier[1].in_features  # = 1280
    model.classifier[1] = nn.Linear(in_features, num_classes)

    print(f"[INFO] Mô hình MobileNetV2 đã sẵn sàng.")
    print(f"[INFO] Lớp classifier cuối: Linear({in_features}, {num_classes})")

    return model


# ─────────────────────────────────────────────
# 5. VÒNG LẶP HUẤN LUYỆN (Training Loop)
# ─────────────────────────────────────────────

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs, save_path):
    """
    Huấn luyện và validate mô hình qua các epoch.
    Lưu trọng số tốt nhất (theo val accuracy) vào file .pth.

    Args:
        model         : Mô hình cần huấn luyện.
        dataloaders   : DataLoader cho train và val.
        dataset_sizes : Số lượng mẫu mỗi tập.
        criterion     : Hàm loss (CrossEntropyLoss).
        optimizer     : Bộ tối ưu (Adam).
        scheduler     : Bộ điều chỉnh learning rate.
        num_epochs    : Số epoch huấn luyện.
        save_path     : Đường dẫn lưu model tốt nhất.

    Returns:
        model         : Mô hình với trọng số tốt nhất.
        history       : Dict chứa lịch sử loss/accuracy.
    """
    # Lưu trọng số tốt nhất vào bộ nhớ
    best_model_weights = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0

    # Lịch sử huấn luyện để vẽ đồ thị sau
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
    }

    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"  Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")

        # Mỗi epoch gồm 2 giai đoạn: train → val
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()   # Bật chế độ huấn luyện (dropout, batchnorm hoạt động)
            else:
                model.eval()    # Bật chế độ đánh giá (không cập nhật gradient)

            running_loss     = 0.0
            running_corrects = 0

            # ── Duyệt qua từng batch ──
            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                # Chuyển dữ liệu lên device (GPU/CPU)
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # Reset gradient về 0 trước mỗi batch
                optimizer.zero_grad()

                # ── Forward pass ──
                # Chỉ tính gradient khi đang train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)          # Dự đoán
                    loss = criterion(outputs, labels)  # Tính loss
                    _, preds = torch.max(outputs, 1)   # Lấy class có xác suất cao nhất

                    # ── Backward pass & cập nhật trọng số ──
                    if phase == "train":
                        loss.backward()    # Tính gradient
                        optimizer.step()   # Cập nhật trọng số

                # Tích lũy loss và số dự đoán đúng
                running_loss     += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # In tiến trình mỗi 10 batch
                if (batch_idx + 1) % 10 == 0 and phase == "train":
                    print(f"    Batch [{batch_idx + 1}/{len(dataloaders[phase])}] "
                          f"Loss: {loss.item():.4f}")

            # Điều chỉnh learning rate sau mỗi epoch train
            if phase == "train":
                scheduler.step()

            # ── Tính loss và accuracy trung bình của epoch ──
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc  = running_corrects.double() / dataset_sizes[phase]

            # Lưu vào lịch sử
            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc.item())

            print(f"\n  [{phase.upper():5s}] Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f} ({epoch_acc*100:.2f}%)")

            # ── Lưu mô hình tốt nhất ──
            if phase == "val" and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())

                # Lưu checkpoint đầy đủ: trọng số + thông tin epoch
                checkpoint = {
                    "epoch":      epoch + 1,
                    "state_dict": best_model_weights,
                    "val_acc":    best_val_acc.item(),
                    "optimizer":  optimizer.state_dict(),
                }
                torch.save(checkpoint, save_path)
                print(f"  ✅ Đã lưu mô hình tốt nhất! Val Accuracy: {best_val_acc:.4f}")

    # Tổng thời gian huấn luyện
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  Huấn luyện hoàn tất trong {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
    print(f"  Val Accuracy tốt nhất: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"  Mô hình đã lưu tại: {save_path}")
    print(f"{'='*60}\n")

    # Nạp lại trọng số tốt nhất vào mô hình
    model.load_state_dict(best_model_weights)
    return model, history


# ─────────────────────────────────────────────
# 6. ĐIỂM KHỞI CHẠY CHÍNH (Main Entry Point)
# ─────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  PHÂN LOẠI MỨC ĐỘ CHÍN TRÁI CÂY — MobileNetV2")
    print("="*60 + "\n")

    # ── Bước 1: Tạo DataLoader ──
    dataloaders, dataset_sizes, class_names = create_dataloaders(DATA_DIR, BATCH_SIZE)

    # Kiểm tra số lớp khớp với cấu hình
    assert len(class_names) == NUM_CLASSES, (
        f"Số lớp không khớp! Tìm thấy {len(class_names)} lớp, "
        f"cấu hình yêu cầu {NUM_CLASSES} lớp."
    )

    # ── Bước 2: Khởi tạo mô hình ──
    model = build_model(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    # ── Bước 3: Hàm Loss & Optimizer ──

    # CrossEntropyLoss: phù hợp cho bài toán phân loại đa lớp
    criterion = nn.CrossEntropyLoss()

    # Adam optimizer: chỉ cập nhật các tham số có requires_grad=True
    # (tức là chỉ lớp classifier cuối, vì backbone đã bị đóng băng)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-4   # L2 regularization để tránh overfitting
    )

    # StepLR: giảm learning rate đi 10 lần mỗi 7 epoch
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # ── Bước 4: Huấn luyện mô hình ──
    trained_model, history = train_model(
        model        = model,
        dataloaders  = dataloaders,
        dataset_sizes= dataset_sizes,
        criterion    = criterion,
        optimizer    = optimizer,
        scheduler    = scheduler,
        num_epochs   = NUM_EPOCHS,
        save_path    = SAVE_PATH,
    )

    # ── Bước 5: In tóm tắt kết quả ──
    print("📊 LỊCH SỬ HUẤN LUYỆN:")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>10} | {'Val Loss':>10} | {'Val Acc':>10}")
    print("-" * 55)
    for i in range(NUM_EPOCHS):
        print(
            f"{i+1:>6} | "
            f"{history['train_loss'][i]:>10.4f} | "
            f"{history['train_acc'][i]:>10.4f} | "
            f"{history['val_loss'][i]:>10.4f} | "
            f"{history['val_acc'][i]:>10.4f}"
        )

    print(f"\n✅ Hoàn tất! Mô hình tốt nhất đã được lưu tại: '{SAVE_PATH}'")
    print("   Để sử dụng mô hình, nạp lại bằng lệnh:")
    print("   checkpoint = torch.load('best_model.pth')")
    print("   model.load_state_dict(checkpoint['state_dict'])\n")


# ─────────────────────────────────────────────
# CHẠY CHƯƠNG TRÌNH
# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()

