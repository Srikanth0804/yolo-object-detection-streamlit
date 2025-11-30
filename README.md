# yolo-object-detection-streamlit
Real-time object detection web app using YOLOv8 and Streamlit

.

🧠 **YOLOv8 Object Detection Web App**

Real-Time Object Detection using YOLOv8 and Streamlit

📌 **Project Overview**

This project implements a real-time object detection web application using YOLOv8 and Streamlit.
Users can upload any image and the model automatically detects and labels objects with bounding boxes and confidence scores.

The project demonstrates the practical application of deep learning in computer vision, making object detection accessible through a simple user interface.

🎯 **Objective**

To build an interactive web-based object detection app that allows users to upload an image and instantly visualize detected objects using a pretrained YOLOv8 model.

🛠 **Tech Stack**

Python
OpenCV
NumPy
Streamlit
Ultralytics YOLOv8
PIL (Image Processing)

⚙️ **Workflow**

Load YOLOv8 model

Upload image via Streamlit UI

Run inference on uploaded image

Display annotated image with bounding boxes

Allow users to download detected output

🔍 **Key Features**

Upload images in JPG / PNG formats

Real-time object detection

Annotated output visualization

Downloadable result image

Lightweight yolov8n model for fast inference

📁 **Project Structure**

yolo-object-detection/

│── app.py

│── yolov8n.pt

│── README.md

▶ **How to Run Locally**

Clone the repository:

git clone <repository_url>
cd yolo-object-detection


Install dependencies:

pip install -r requirements.txt


Run the app:

streamlit run streamlit_yolo_app.py

✅ requirements.txt

Create a file named requirements.txt and add:

streamlit
opencv-python
numpy
ultralytics
pillow

🚀 **Future Enhancements**

Add webcam support

Enable video detection

Deploy on Hugging Face Spaces

Implement confidence filtering

Add class-specific filtering

👤 **Author**

Srikanth Gunti
📧 Email: srikanthgunti11@gmail.com

🔗 LinkedIn: https://www.linkedin.com/in/srikanth-gunti-

⭐ Support

If you like this project, please ⭐ star the repository!
