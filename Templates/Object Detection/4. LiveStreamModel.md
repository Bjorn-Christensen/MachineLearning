# About

This will be a simplified tutorial that assumes you are already very familiar with your dataset and your model. The intention is to help you quickly set up a system that allows you to live-stream your model using a webcam and run real time object detection. As with the other tutorials in this section we will continue to use PyTorch.

# Starter Imports

```python
# "Real-time" torchvision tutorial imports
from torchvision.models import detection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from imutils.video import VideoStream # Access webcam
from imutils.video import FPS # Track FPS from webcam
from PIL import Image
from io import BytesIO
import pandas as pd
import numpy as np
import torch
import time
import cv2
from tqdm import tqdm
import imagehash
import requests
```

# Set Device Type

```python
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
```