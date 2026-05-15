# 🔍 Crack Detection Using CNN

This project detects **surface cracks** in images using a **Convolutional Neural Network (CNN)** built with TensorFlow. It supports model training, evaluation, and real-time crack detection via webcam. Ideal for infrastructure monitoring or industrial applications.

--------------------------------------------------

## 📁 Project Structure

Crack-Detection/
├── train\_model.py           # Trains the CNN on crack images
├── evaluate\_model.py        # Evaluates model, creates graphs & reports
├── crack\_cam\_realtime.py    # Detects cracks in real-time using webcam
├── my\_model.h5              # Saved trained model
├── accuracy\_graph.png       # Accuracy plot for training vs validation
├── confusion\_matrix.png     # Confusion matrix visualization
└── Crack/
└── Model/
├── Positive/        # Images with cracks
└── Negative/        # Images without cracks

--------------------------------------------------

## 🧠 Model Details

A lightweight CNN with:
- `Conv2D` → `MaxPooling` → `Conv2D` → `MaxPooling` → `GlobalAveragePooling` → `Dense`
- Activation: ReLU + Sigmoid (for binary classification)
- Loss: `binary_crossentropy`
- Optimizer: `adam`
- Image size: `120x120x3`

---

## 🛠️ Setup Instructions

### 1. **Install Requirements**
Run this once to set up your environment:


pip install tensorflow numpy pandas matplotlib seaborn plotly scikit-learn opencv-python


### 2. **Dataset Format**

Your images should be organized like this:

--------------------------------------------------
Crack/
└── Model/
    ├── Positive/        # Cracked surfaces
    └── Negative/        # Non-cracked surfaces


Ensure all images are  .jpg.

--------------------------------------------------

# How to Run

# Train the Model


python model.py


This will:

* Preprocess and augment the dataset
* Train the CNN
* Save the model as: my_model.h5
* Output: accuracy_graph.png

--------------------------------------------------

# Evaluate the Model


python evaluate_model.py

This will:

* Load: my_model.h5
* Evaluate performance on the test set
* Print:

  * Test Loss & Accuracy
  * Classification Report
* Save:

  * confusion_matrix.png
  * accuracy_graph.png

--------------------------------------------------

# Run Real-Time Crack Detection

python crack_cam_realtime.py

This opens your webcam and overlays live predictions:

* 🟥 "Crack" if detected
* 🟩 "No Crack" otherwise
  Press `q` to quit.

--------------------------------------------------

## 📊 Output Visuals

* 📈 accuracy_graph.png – Training vs Validation Accuracy over epochs
* 🔥 confusion_matrix.png – Heatmap showing model performance

---

## ⚙️ Customization Tips

Want better accuracy? Try:

* Increasing epochs
* Using deeper Conv layers
* Fine-tuning batch size or image dimensions
* Using advanced data augmentation

---

## 💡 Future Upgrades

* Deploy model as a Flask/Streamlit web app
* Use edge devices like Raspberry Pi / Rock Pi for real-time detection
* Add segmentation for precise crack localization
* Integrate drone feed for large-scale infrastructure scanning

--------------------------------------------------

## 🧑‍💻 Author

**Sahil B Pillai**  
Engineer | Robotics & AI Enthusiast  
[LinkedIn](your-linkedin-url) | [GitHub](https://github.com/Sahilpillai006)

---

## 📄 License

This project is open-source and available under the MIT License.