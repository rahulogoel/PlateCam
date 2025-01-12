import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
import tensorflow as tf
import os
from django.conf import settings

# Load your trained model
MODEL_PATH = r"C:\vscode\Projects\Automatic Number Plate Recognition\myproject\webapp\my_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(image, target_size=(224, 224)):
    image_resized = cv2.resize(image, target_size)  # resize the image
    image_normalized = image_resized / 255.0  # normalize pixel values

    return np.expand_dims(image_normalized, axis=0)  # add batch dimension

def process_predictions(predictions, image_width, image_height):
    x_min, x_max, y_min, y_max = predictions[0] 
    # converting normalized coordinates to pixel values
    xmin = int(x_min * image_width)
    xmax = int(x_max * image_width)
    ymin = int(y_min * image_height)
    ymax = int(y_max * image_height)

    return xmin, ymin, xmax, ymax

def detect_license_plate(image):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)  # calling out our model
    # image dimensions
    image_height, image_width, _ = image.shape

    return process_predictions(predictions, image_width, image_height)

def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    if brightness != 0:
        shadow = brightness if brightness > 0 else 0
        highlight = 255 + brightness if brightness < 0 else 255
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

def OCR(image_path):
    img = cv2.imread(image_path)
    xmin, ymin, xmax, ymax = detect_license_plate(img)
    # placing bounding box on top of the image
    img_with_box = img.copy()
    cv2.rectangle(img_with_box, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2) 

    # save the image with bounding box
    bounded_image_path = os.path.join(settings.MEDIA_ROOT, 'bounded', os.path.basename(image_path))
    os.makedirs(os.path.dirname(bounded_image_path), exist_ok = True)
    cv2.imwrite(bounded_image_path, img_with_box)
    # convert the saved path to a URL
    bounded_plate_image_url = os.path.join(settings.MEDIA_URL, 'bounded', os.path.basename(image_path))

    # crop the license plate region
    plate_img = img[ymin:ymax, xmin:xmax]
    # apply preprocessing for OCR
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    enhanced_plate = apply_brightness_contrast(gray, brightness=40, contrast=70)
    # extract text using Tesseract
    text = pytesseract.image_to_string(enhanced_plate)
    extracted_text = text.strip()

    # save the cropped license plate image
    plate_image_path = os.path.join(settings.MEDIA_ROOT, 'roi', os.path.basename(image_path))
    os.makedirs(os.path.dirname(plate_image_path), exist_ok = True)
    cv2.imwrite(plate_image_path, plate_img)
    # convert the saved path to a URL
    plate_image_url = os.path.join(settings.MEDIA_URL, 'roi', os.path.basename(image_path))

    return extracted_text, plate_image_url, bounded_plate_image_url
