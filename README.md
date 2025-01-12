# PlateCam

PlateCam is an **Automated Number Plate Recognition (ANPR)** system that combines deep learning and computer vision techniques to detect and recognize vehicle number plates. The project includes a web-based application built using Django, enabling you to interact with the ANPR system through an intuitive interface.

![demo](https://github.com/rahulogoel/PlateCam/blob/main/demo-gif.gif)

## Features

- **Data Preprocessing**: Efficient handling and annotation of dataset images using XML files.
- **Deep Learning Model**: Implementation of You Only Look Once (YOLO) v5 model for real time plate detection and recognition.
- **OCR Integration**: Integration with Pytesseract for extracting text from detected plates.
- **Visualization**: Use of tools like Matplotlib and Plotly for visualizing data and results.
- **TensorBoard**: Evaluated the model with the help of tensorflow's tensorboard.
- **Pipeline**: Well defined steps and modular approach which makes it easier to manage and maintain the workflow.
- **Web Application**: A Django-based web app to provide a user-friendly interface for uploading images and obtaining results.

## Dataset
Check out the dataset that I used for Automated Number Plate Recognition:

https://www.kaggle.com/datasets/andrewmvd/car-plate-detection/data

## Getting Started

### Prerequisites

1. Python 3.7+
2. Required Python libraries:
   - TensorFlow
   - Keras
   - OpenCV
   - Pytesseract
   - Plotly
   - Matplotlib
   - Scikit-learn
   - Django
3. Tesseract OCR installed and added to the system path. [Guide](https://tesseract-ocr.github.io/tessdoc/Installation.html)

### Steps

1. Set up new Django project;

   ```
   django-admin startproject myproject
   cd myproject
   ```

2. Clone the repository inside `myproject` folder:

   ```bash
   git clone https://github.com/your-username/PlateCam.git
   cd webapp
   ```

3. Make sure to configure your own  `pytesseract` path in `deeplearn.py`. Like here is mine:

   ```
   pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
   ```

4. Set up the Django web app server:

   ```bash
   cd.. 
   python manage.py runserver
   ```

5. Access the application at `http://127.0.0.1:8000/`.

## Future Work

- Enhance OCR accuracy by fine-tuning parameters.
- Expand dataset for improved model generalization.
- Try different OCR models like Keras-OCR or EasyOCR as while testing the model I came to know that Tesseract is only good for high-resolution images but has poor precision for low-resolution images. So we can try other models too.[Check comparison blog post](https://thangasami.medium.com/tesseract-vs-keras-ocr-vs-easyocr-ec8500b9455b)

## License

MIT
