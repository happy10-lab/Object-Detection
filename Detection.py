#install necessary packages
!pip install torch torchvision torchaudio  --quiet
!pip install opencv-python matplotlib   --quiet

#clone yolov5 repository 
!git clone https://github.com/ultralytics/yolov5

#change directory and install yolov5 requirement
%cd yolov5
!pip install -r requirements.txt   --quiet


#importing the libraries
import torch
import cv2
import urllib.request
import numpy as np
# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# Valid image URL
img_url = "https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg"
# Download and decode image
resp = urllib.request.urlopen(img_url)
image = np.asarray(bytearray(resp.read()), dtype="uint8")
img_bgr = cv2.imdecode(image, cv2.IMREAD_COLOR)
# Convert to RGB
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
# Run inference
results = model(img_rgb)
# Print, show, and save results
results.print()
results.show()
results.save()
