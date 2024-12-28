# Crack Detection

## Overview
This project implements crack detection using the YOLOv8 model. We specifically use the **instance segmentation model** from the `ultralytics` library to detect and analyze cracks in images. The project provides training, validation, and inference workflows, with detailed property extraction of detected cracks.

---

## Getting Started

### Clone the Repository
```bash
git clone <repository-url>
cd crack_detection
```
### Install Dependencies
```bash
pip install -r requirements.txt
```
### Model 
The YOLOv8 instance segmentation model is used for detecting cracks in images. This model provides pixel-level segmentation, enabling precise detection of cracks with their regions outlined.

### Custom Training
Data Preparation
Images:

1.Prepare a dataset of images containing cracks.
2.Organize the dataset into two folders: train and valid.

Annotations:

1.One text file per image: Each image in the dataset has a corresponding text file with the same name as the image file and the ".txt" extension.
2.One row per object: Each row in the text file corresponds to one object instance in the image.
3.Object information per row: Each row contains the following information about the object instance:
Object class index: An integer representing the class of the object (e.g., 0 for person, 1 for car, etc.).
Object bounding coordinates: The bounding coordinates around the mask area, normalized to be between 0 and 1.
The format for a single row in the segmentation dataset file is as follows:

```bash
<class-index> <x1> <y1> <x2> <y2> ... <xn> <yn>
```
### data yml formate
Create a data.yaml file to specify the dataset details:

```bash
train: /full/path/to/train/images
val: /full/path/to/valid/images
nc: <number_of_classes>
names: [class1, class2, ...]
```

### Training the Model

Use the train.py script for training. The script includes imports, model loading, and training routines.
To run training:

```bash
python train.py
```
For CPU Training
Replace the device setting in train.py with "cpu".

Outputs of train.py are 

Results Folder: Contains training loss, validation metrics, confusion matrix, and other logs.
Weights: Saved in the specified location during training.

### Inference

Use the predict.py script for inference.

To run inference:

```bash
python predict.py
```

This script performs crack detection on input images and saves result.jpg which is the detected bounding boxand segmented image, result_mask.jpg gives the mast region of the crack details, result_mask1.jpg is the detected crack region overlayed on the original image . at end exports properties of detected cracks to a CSV file.

Crack Properties Extracted:

1.Area:  The number of pixels in the region. It represents the size of the crack detected in the image.

2.Perimeter:  The total length of the boundary around the detected crack region.

3.Major Axis Length:The length of the major axis of an ellipse fitted to the region.

4.Minor Axis Length: The length of the minor axis of an ellipse fitted to the region.

5.Aspect Ratio: Ratio of the major axis to the minor axis.

6.Bounding Box Aspect Ratio: Ratio of the width to the height of the bounding box around the crack

7.Circularity: Measures how regular or irregular the region

8.Centroid: The center of mass of the crack region

9.Bounding Box: The smallest rectangle enclosing the region

10.Eccentricity: A measure of how much the region deviates from being circular. It is the ratio of the distance between the foci of the ellipse and its major axis length.Closer to 1 for elongated shapes.

11.Solidity: The ratio of the area of the crack region to the area of its convex hull.Less than 1 for regions with irregular boundaries.

12.Convex Area:  The area of the convex hull of the region. The convex hull is the smallest convex shape that can enclose the region. Helps assess the region's irregularity.

### post model Quantization we can export model to onnx formate

```bash
my_new_model.export(format='onnx', imgsz=[800,800])
```