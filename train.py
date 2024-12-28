from ultralytics import YOLO
from matplotlib import pyplot as plt
from PIL import Image

#Instance
model = YOLO('yolov8m-seg.yaml')  # build a new model from YAML
model = YOLO('yolov8m-seg.pt')  #

# define number of classes based on YAML
import yaml
with open("/content/drive/MyDrive/POC2/data.yaml", 'r') as stream:
    num_classes = str(yaml.safe_load(stream)['nc'])

#Define a project --> Destination directory for all results
project = "results"
#Define subdirectory for this specific training
name = "500_epochs-" 

# Train the model
results = model.train(data='/content/drive/MyDrive/POC2/data.yaml',
                      project=project,
                      name=name,
                      device=0,
                      epochs=500,
                      patience=30,
                      batch=4,
                      imgsz=800)