"""
Script tổ chức dataset từ dataset_raw → dataset/train và dataset/val
Mapping:
  Train/Ripe/banana_ripe_*     → dataset/train/banana_ripe/
  Train/Unripe/banana_unripe_* → dataset/train/banana_green/
  Train/Overipe/banana_over*   → dataset/train/banana_damaged/
  (tương tự cho mango, tomato)

Lưu ý: Dataset không có "semi_ripe" riêng
  → Dùng ảnh Unripe làm "green", chia đôi Ripe làm "semi_ripe" và "ripe"
  → Dùng Overripe làm "damaged"

Tỉ lệ split: 80% train, 20% val
"""

import os
import shutil
import random
from pathlib import Path

# ─── Cấu hình ───
RAW_DIR   = Path("./dataset_raw")
OUT_DIR   = Path("./dataset")
SPLIT     = 0.8   # 80% train, 20% val
SEED      = 42

# Mapping: (loại quả, trạng thái raw) → (class_name, split_ratio_for_semi_ripe)
# Với Ripe: chia đôi → 50% vào *_semi_ripe, 50% vào *_ripe
FRUITS = ["banana", "mango", "tomato"]

RIPENESS_MAP = {
    "Ripe":    ["semi_ripe", "ripe"],   # chia đôi
    "Unripe":  ["green"],
    "Overipe": ["damaged"],
}

random.seed(SEED)

def copy_images(src_files, dst_dir):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in src_files:
        shutil.copy2(f, dst_dir / f.name)

def organize():
    # Xóa .gitkeep để tránh lỗi
    for gk in OUT_DIR.rglob(".gitkeep"):
        gk.unlink()

    total_copied = 0

    for split_folder in ["Train", "Test"]:
        for ripeness_folder, class_suffixes in RIPENESS_MAP.items():
            folder = RAW_DIR / split_folder / ripeness_folder
            if not folder.exists():
                print(f"[WARN] Không tìm thấy: {folder}")
                continue

            for fruit in FRUITS:
                # Lọc ảnh theo tên fruit
                all_imgs = sorted([
                    f for f in folder.iterdir()
                    if f.is_file()
                    and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
                    and f.stem.startswith(fruit)
                ])

                if not all_imgs:
                    print(f"[WARN] Không có ảnh {fruit} trong {folder}")
                    continue

                random.shuffle(all_imgs)

                if len(class_suffixes) == 2:
                    # Ripe → chia đôi thành semi_ripe + ripe
                    mid = len(all_imgs) // 2
                    chunks = [all_imgs[:mid], all_imgs[mid:]]
                else:
                    chunks = [all_imgs]

                for suffix, chunk in zip(class_suffixes, chunks):
                    class_name = f"{fruit}_{suffix}"

                    # Train/val split
                    split_at = int(len(chunk) * SPLIT)
                    train_imgs = chunk[:split_at]
                    val_imgs   = chunk[split_at:]

                    copy_images(train_imgs, OUT_DIR / "train" / class_name)
                    copy_images(val_imgs,   OUT_DIR / "val"   / class_name)

                    n = len(train_imgs) + len(val_imgs)
                    total_copied += n
                    print(f"  ✅ {class_name:25s} train={len(train_imgs):4d}  val={len(val_imgs):4d}  (tổng {n})")

    print(f"\n🎉 Hoàn tất! Tổng {total_copied} ảnh đã được tổ chức.")

    # In tóm tắt cuối
    print("\n📊 TÓM TẮT:")
    for split in ["train", "val"]:
        total = 0
        print(f"\n  [{split.upper()}]")
        for cls in sorted((OUT_DIR / split).iterdir()):
            count = len(list(cls.glob("*.*")))
            total += count
            print(f"    {cls.name:25s}: {count:4d} ảnh")
        print(f"    {'TOTAL':25s}: {total:4d} ảnh")

if __name__ == "__main__":
    print("🚀 Bắt đầu tổ chức dataset...\n")
    organize()

