🦁 Serengeti African Wildlife Classifier
Deep learning model for automated wildlife species identification from camera trap images — Serengeti National Park, Tanzania
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
[![Accuracy](https://img.shields.io/badge/Accuracy-96.3%25-brightgreen)]()
![License](https://img.shields.io/badge/License-MIT-yellow)
---
🌍 Project Overview
Serengeti National Park in Tanzania is one of the world's most biodiverse ecosystems — home to lions, elephants, rhinos, zebras, wildebeest, and over 60 other mammalian species. Conservation rangers deploy hundreds of camera traps across 1,125 km² to monitor animal populations and detect poaching activity.
The problem: Rangers manually review thousands of camera trap images daily — a slow, resource-intensive process that delays conservation responses and diverts time from active fieldwork.
The solution: This project applies deep learning and transfer learning (EfficientNet-B0) to automatically identify wildlife species in camera trap images — enabling real-time species monitoring and faster anti-poaching response.
---
📊 Results
Metric	Score
Test Accuracy	96.3%
Macro F1 Score	0.96
Training Time	~4 minutes (CPU)
Training Epochs	10
Per-Class Performance
Species	Precision	Recall	F1 Score	Support
🐃 Buffalo	0.97	0.92	0.95	76
🐘 Elephant	0.94	0.99	0.97	85
🦏 Rhino	0.95	0.96	0.95	74
🦓 Zebra	1.00	0.98	0.99	66
Overall	0.97	0.96	0.96	301
> **Zebra achieved perfect precision (1.00)** — the model never once misidentified another animal as a zebra.
---
📈 Training Charts
![Training Results](training_results.png)
---
🔵 Confusion Matrix
![Confusion Matrix](confusion_matrix.png)
The model shows strong diagonal dominance — most predictions fall on the correct class. Buffalo and Rhino occasionally confuse each other (both are large, dark-coloured animals with similar body shapes in camera trap lighting), which is expected and consistent with how even human annotators sometimes struggle with these species.
---
🧠 Model Architecture
Base Model: EfficientNet-B0 (pretrained on ImageNet)
Technique: Transfer Learning — fine-tuned on African wildlife camera trap images
Custom Head: Dropout(0.3) → Linear(1280, 512) → ReLU → Dropout(0.2) → Linear(512, 4)
Framework: PyTorch + TorchVision
Why EfficientNet-B0?
EfficientNet-B0 achieves state-of-the-art accuracy with significantly fewer parameters than ResNet or VGG — making it faster to train and more suitable for eventual edge deployment on solar-powered camera trap devices in remote conservation areas.
Pipeline
```
Camera Trap Image
        ↓
Resize (224×224) + Augmentation (Flip, Rotation, Colour Jitter)
        ↓
Normalise (ImageNet mean & std)
        ↓
EfficientNet-B0 (Pretrained Backbone — Transfer Learning)
        ↓
Custom Classification Head (4 species)
        ↓
Species Prediction + Confidence Score
```
---
📁 Dataset
Species: Buffalo, Elephant, Rhino, Zebra (4 classes)
Total Images: 1,504
Train/Val Split: 80/20
Source: African wildlife camera trap dataset
---
⚙️ Installation & Usage
```bash
git clone https://github.com/mariam-madata/serengeti-wildlife-classifier
cd serengeti-wildlife-classifier
pip install torch torchvision scikit-learn matplotlib seaborn
```
Training
```python
# Open the notebook in Google Colab
# Runtime → Run all
```
Inference
```python
import torch
from torchvision import models, transforms
from PIL import Image

# Load model
model = models.efficientnet_b0()
model.classifier[1] = torch.nn.Linear(1280, 4)
model.load_state_dict(torch.load('wildlife_classifier.pth', map_location='cpu'))
model.eval()

# Predict
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

classes = ['buffalo', 'elephant', 'rhino', 'zebra']
image = Image.open('your_image.jpg').convert('RGB')
tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(tensor)
    pred = output.argmax(1).item()
    confidence = torch.softmax(output, dim=1).max().item()

print(f"Predicted: {classes[pred]} ({confidence*100:.1f}% confidence)")
```
---
🌱 Real-World Impact
This project directly supports:
Anti-poaching: Automated alerts when rare species are detected near park boundaries
Population monitoring: Track species counts and movement patterns over time
Conservation research: Faster data processing frees rangers for active fieldwork
Climate monitoring: Long-term species distribution data informs ecosystem health
Tanzania's wildlife generates over $2.5 billion in tourism revenue annually — protecting it is both an ecological and economic imperative.
---
🔮 Future Work
[ ] Add Grad-CAM visualisation to highlight which image regions drove predictions
[ ] Expand to 10+ species using full Snapshot Serengeti dataset (1.2M images)
[ ] Deploy as lightweight API for ranger field tablets
[ ] Extend to detect human intrusion for anti-poaching alerts
[ ] Integrate with GPS metadata for spatial distribution mapping
---
👤 Author
Mariam Khamis Madata — Data Scientist & ML Researcher, Dar es Salaam, Tanzania
🌐 Portfolio
💻 GitHub
🔗 LinkedIn
---
🔗 Related Projects
Diabetic Retinopathy Detection — Medical AI · CNN · PyTorch · 83% accuracy
---
📄 License
MIT License — free to use and build upon.
