# 👂 Ear Recognition with VGG16

**Status:** Experimental / Educational

This project implements **ear recognition** by fine-tuning a **VGG16** model on the **Ears dataset** and **EarVN1 dataset**.  

---

## ✨ Features

- Trains a pre-trained VGG16 network on ear images
- Works with multiple datasets (Ears, EarVN1)
- Outputs classification accuracy and loss metrics
- Easy to adapt to new ear image datasets

---

## 🛠️ Tech Stack

- Python 3.x
- PyTorch
- Torchvision
- NumPy, Pandas, Matplotlib

---

## ⚙️ Usage

1. Install dependencies:

```bash
pip install torch torchvision numpy pandas matplotlib

📊 Notes

The model uses transfer learning from ImageNet pre-trained weights.

Dataset preprocessing includes resizing, normalization, and optional data augmentation.

Results may vary depending on dataset split and hardware.

⚠️ Disclaimer

This project is for research and educational purposes only.
