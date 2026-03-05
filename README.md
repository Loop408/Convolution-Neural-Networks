# Pneumonia Detection from Chest X-Ray

An AI-powered web application for detecting pneumonia from chest X-ray images using deep learning.

## 📋 Description

This project provides a simple web interface built with Streamlit that allows users to upload chest X-ray images and get predictions on the likelihood of pneumonia. The model is trained on a dataset of chest X-rays and uses convolutional neural networks (CNNs) to classify images as either normal or indicative of pneumonia.

**⚠️ Disclaimer:** This tool is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult with qualified healthcare professionals for medical advice.

## 🚀 Features

- **Image Upload:** Upload chest X-ray images in JPG or PNG format
- **Real-time Prediction:** Get instant predictions with confidence scores
- **Visual Feedback:** View uploaded and processed images
- **Risk Assessment:** Categorized risk levels (Low, Moderate, High)
- **Probability Visualization:** Bar chart showing prediction probabilities

## 📁 Project Structure

- `app.py` - Main Streamlit application
- `pneumonia_model.h5` - Pre-trained Keras model for pneumonia detection
- `xray.ipynb` - Jupyter notebook for data exploration and model development

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/pneumonia-detection.git
   cd pneumonia-detection
   ```

2. **Install dependencies:**
   ```bash
   pip install streamlit tensorflow pillow numpy matplotlib
   ```

3. **Ensure the model file is present:**
   The `pneumonia_model.h5` file should be in the same directory as `app.py`.

## ▶️ Usage

1. **Run the application:**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser:**
   Navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

3. **Upload an X-ray image:**
   - Click on the file uploader
   - Select a chest X-ray image (JPG or PNG)
   - Click "Analyze Image" to get predictions

## 📊 Model Details

- **Framework:** TensorFlow/Keras
- **Input Size:** 150x150 pixels (RGB)
- **Output:** Binary classification (Normal vs Pneumonia)
- **Architecture:** Convolutional Neural Network (CNN)

The model expects images to be resized to 150x150 pixels and normalized to [0,1] range.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

- Dataset: Chest X-Ray Images (Pneumonia) from Kaggle
- Model trained using transfer learning techniques

## 🆘 Support

If you encounter any issues or have questions, please open an issue on GitHub.