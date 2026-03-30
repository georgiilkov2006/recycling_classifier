# ♻️ Smart Recycling Classifier

An AI-powered web application that classifies waste into recycling categories using computer vision.

## What it does
Upload a photo or use your webcam — the app identifies the type of waste and tells you exactly which bin it belongs in, with a confidence score.

## Demo
![App Screenshot](https://i.imgur.com/placeholder.png)

## 🌐 Live Demo
👉 [Try it here](https://huggingface.co/spaces/georgiilkov2006/recycling-classifier)

## Model
- Architecture: MobileNetV2 (transfer learning)
- Dataset: TrashNet (2,527 images, 6 classes)
- Training accuracy: ~90%
- Validation accuracy: ~77%
- Classes: cardboard, glass, metal, paper, plastic, trash

## Tech Stack
- Python, TensorFlow 2.21 / Keras
- MobileNetV2 (pretrained on ImageNet, fine-tuned on TrashNet)
- Flask (REST API backend)
- HTML / CSS / JavaScript (frontend)
- OpenCV (image preprocessing)

## Features
- Real-time webcam classification
- Image upload support
- Confidence score with visual bar
- Low-confidence warning (< 60%)
- Dark mode UI

## How to Run
1. Clone the repository
```
   git clone https://github.com/georgiilkov2006/recycling_classifier.git
```
2. Create a virtual environment with Python 3.11
```
   py -3.11 -m venv venv
   venv\Scripts\activate
```
3. Install dependencies
```
   pip install tensorflow opencv-python flask numpy pillow
```
4. Train the model (or download a pretrained one)
```
   python model/train.py
```
5. Start the app
```
   python app/app.py
```
6. Open `http://127.0.0.1:5000` in your browser

## Project Structure
```
recycling-classifier/
├── dataset/          ← TrashNet images (not included, download separately)
├── model/
│   └── train.py      ← Training script (MobileNetV2 + transfer learning)
├── app/
│   ├── app.py        ← Flask server
│   └── templates/
│       └── index.html ← Frontend UI
└── README.md
```

## Dataset
[TrashNet](https://github.com/garythung/trashnet) — 2,527 images across 6 categories.
