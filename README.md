# Object-Detection-with-YOLOv5 Workshop

This repo contains all files for the **Object Detection with YOLOv5** workshop. Presented at **DataFest Africa 2024**.

## Overview

This workshop focuses on building, training, and deploying object detection models using YOLOv5. The goal is to equip participants with the skills to leverage computer vision for industry-specific tasks.

Participants will:
- Create and annotate custom datasets.
- Configure and optimize YOLOv5 for performance.
- Train, validate, and deploy object detection models.

By the end of the workshop, you will have built a functional YOLOv5 model for real-world applications.

## Requirements

- Laptop
- Stable Internet Connection
- Google Account (for Google Colab and Drive)
- GitHub Account (optional)

### Technical Requirements
- Python 3.7+
- [YOLOv5 repository](https://github.com/ultralytics/yolov5)
- PyTorch 1.7+ (for local training)
- GPU (optional but recommended)

## Workshop Outline

1. **Setup YOLOv5**: Clone the YOLOv5 repository, install dependencies, and configure the environment.
2. **Dataset Preparation**: Download, unzip, and preprocess the custom dataset.
3. **Model Training**: Train the YOLOv5 model on the prepared dataset.
4. **Evaluation**: Test the model's performance on unseen images.
5. **Deployment**: Simulate real-time applications by running the model in inference mode.

## Getting Started

### 1. Clone the YOLOv5 Repository

```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

### 2. Download Dataset
Next, we download our dataset to the colab session. The dataset we will be working with is the Garbage Detection Dataset on [Roboflow ](https://universe.roboflow.com/thai-pham-y5sob/garbage-classify-9ndxx)
```bash
gdown --id <dataset_google_drive_id> -O ./dataset.zip
unzip -q dataset.zip -d ./data/
```
### 3. Train the Model

Now, we prepare to train our model. By modfiying the `data.yaml` file in our dataset's folder. Replace the content of the file with the code below. 

#### Modify the config file 
```yaml
# Class names for the object detection task
# Each number corresponds to a class label for different types of garbage
names:
  0: can     # Class 0 represents cans (e.g., soda cans, tin cans)
  1: plastic # Class 1 represents plastic items (e.g., bottles, bags)
  2: glass   # Class 2 represents glass items (e.g., glass bottles, jars)
  3: paper   # Class 3 represents paper items (e.g., cardboard, paper sheets)

# Number of classes
nc: 4  # We have 4 classes in total: can, plastic, glass, and paper

# Paths to training and validation image datasets
train: /content/garbage-detection-dataset/garbage-detection-dataset/train/images
# Specifies the path to the folder containing training images

val: /content/garbage-detection-dataset/garbage-detection-dataset/valid/images
# Specifies the path to the folder containing validation images


```
#### Train Model
Lastly, we specify our model's training parameters and paths as seen below. 
```bash
python train.py --img 640 --batch 16 --epochs 50 --data ./data/custom_dataset.yaml --weights yolov5s.pt
```
### 4. Test the Model
```bash

python detect.py --source ./data/test_images/ --weights runs/train/exp/weights/best.pt
```
##  What's next?

### Is performance Satisfactory? If No,
- Re-training with more data 
- Augmentation?
- Experimenting with different set of hyperparameters. 

### If satisfactory, 
Export trained model, can be deployed on the web, locally, on mircrocontrollers like the raspberry pi, NIVIDIA Jetson Nano,etc. Some of the popular export formats are listed below. 
- `.pt` -  pytorch model(default) 
- `.tflite` tensorflow lite model for deployment on mobile and low-end devices. 
- `.onnx`- popularly used for low end devices. 

The object detection model has different applications: 
- Object counting
- Defect Monitoring production lines. 
- Object Blurring. 
- Line Passage Counter
- First stage for Image recogniton algorithm, e.g Number Plate Detector.


## Additional Resources
- [Krish Naik's Computer Vision Playlist](https://www.youtube.com/watch?v=jLcuVu5xdDo&t=16s)
- [Kaggle's Introduction to Computer Vision](https://www.kaggle.com/learn/computer-vision)
- [Training a Custom YOLO Model - Step-by-Step Tutorial](https://www.youtube.com/watch?v=GRtgLlwxpc4)

## Credits
- Ultralytics [YOLOv5](https://github.com/ultralytics/yolov5)
- Dataset provided by [Roboflow](https://universe.roboflow.com/)
