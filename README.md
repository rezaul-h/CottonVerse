
# CottonVerse 🌿

CottonVerse is a multi-model Flask-based web application for image classification tasks in the agricultural and textile domains. It enables interpretable prediction using Grad-CAM and supports four independent models:

- 🍃 Cotton Leaf Disease (CottonLeafNet)
- 🌿 Cotton Leaf Disease (SAR-CLD-2024)
- 🧵 Fabric Texture Percentage (CottonFabricImageBD)
- 🩸 Fabric Stain Type (FabricSpotDefect)

## 🚀 Features

- 📁 Drag-and-drop file upload
- ✅ Prediction probabilities with visual bar chart
- 🔥 Grad-CAM visualization for interpretability
- 📸 Upload preview and CAM download
- 🖼️ Multiple datasets and models supported
- 🧠 Powered by LEViT model (via PyTorch + timm)
- 🎨 Responsive, modern, and interactive UI

## 📂 Project Structure

```
cottonverse/
├── app.py                     # Flask app
├── utils.py                   # Preprocessing & Grad-CAM logic
├── models/                    # Trained .pth models
├── static/
│   ├── uploads/               # Uploaded images
│   └── cams/                  # Grad-CAM outputs
├── templates/                 # HTML templates
└── requirements.txt           # Dependencies
└── README.md
```

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/rezaul-h/CottonVerse.git
cd cottonverse
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

6. Open `http://127.0.0.1:5000` in your browser 🌐

## 📊 Models & Datasets

| Model Name | Dataset                | Classes |
|------------|------------------------|---------|
| LEViT      | CottonLeafNet          | 8       |
| LEViT      | SAR-CLD-2024           | 9       |
| LEViT      | CottonFabricImageB     | 13      |
| LEViT      | FabricSpotDefect       | 12      |

All models are trained using the **LEViT** architecture with PyTorch, saved in `.pth` format.

## 🧪 How to Use

1. Click on a model on the landing page.
2. Upload an image for classification.
3. View prediction probabilities and Grad-CAM.
4. Download the CAM if desired.

## 📦 Dependencies

- Flask
- PyTorch
- timm
- torchvision
- OpenCV
- matplotlib
- pytorch-grad-cam

## 🙏 Acknowledgements

- LEViT model: Facebook AI
- GradCAM: Jacob Gildenblat (pytorch-grad-cam)
