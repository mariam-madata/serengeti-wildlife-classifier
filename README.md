# 🦁 Serengeti African Wildlife Classifier

**Deep learning model for automated wildlife species identification — Serengeti National Park, Tanzania**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-96.3%25-brightgreen)

## 🌍 Overview

This project applies EfficientNet-B0 transfer learning to automatically identify African wildlife species from camera trap images — enabling real-time conservation monitoring and anti-poaching support in Tanzania's Serengeti National Park.

## 📊 Results

| Metric | Score |
|---|---|
| Test Accuracy | **96.3%** |
| Macro F1 Score | **0.96** |
| Training Time | ~4 minutes |
| Epochs | 10 |

## Per-Class Performance

| Species | Precision | Recall | F1 |
|---|---|---|---|
| 🐃 Buffalo | 0.97 | 0.92 | 0.95 |
| 🐘 Elephant | 0.94 | 0.99 | 0.97 |
| 🦏 Rhino | 0.95 | 0.96 | 0.95 |
| 🦓 Zebra | 1.00 | 0.98 | 0.99 |

> Zebra achieved perfect precision (1.00) — the model never misidentified another animal as a zebra.

## 🧠 Model

- **Architecture:** EfficientNet-B0 (pretrained on ImageNet)
- **Technique:** Transfer Learning
- **Framework:** PyTorch
- **Dataset:** 1,504 African wildlife camera trap images

## 🌱 Impact

- Anti-poaching automated alerts
- Real-time species population monitoring
- Conservation research acceleration
- Tanzania generates $2.5B annually from wildlife tourism

## 🔮 Future Work

- Grad-CAM visualisation for explainability
- Expand to 10+ species
- Deploy as ranger field tablet API
- Human intrusion detection for anti-poaching

## 👤 Author

**Mariam Khamis Madata** — Dar es Salaam, Tanzania

🌐 [Portfolio](https://mariam-madata.github.io) · 💻 [GitHub](https://github.com/mariam-madata) · 🔗 [LinkedIn](https://www.linkedin.com/in/mariam-madata-183aa7187)

## 🔗 Related Projects

[Diabetic Retinopathy Detection](https://github.com/mariam-madata/diabetic-retinopathy-detection) — Medical AI · CNN · PyTorch · 83% accuracy
