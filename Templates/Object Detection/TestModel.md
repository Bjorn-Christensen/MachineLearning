# About

This tutorial is designed to efficiently walk you through testing an object detection model by examining accuracy scores and visualizing bounding box predictions. It will be treated as a follow-up to [FinetunePyTorchModel.md](FinetunePyTorchModel.md) and as such it will assume that you are familiar with important aspects of your dataset, such as how to properly load data, what kind of labels/classes you are working with, and how to generate bounding boxes and map integer labels to objects. Some sections of this tutorial are also written under the assumption that you are working with data that has already been processed following the template in [PreprocessData.md](PreprocessData.md). Please refer to the table of contents for assistance.

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#starter-imports">Starter Imports</a></li>
    <li><a href="#set-device-type">Set Device Type</a></li>
    <li>
      <a href="#load-dataset">Load Dataset</a>
      <ul>
        <li><a href="#pandas-load">Pandas Dataset</a></li>
        <li><a href="#huggingface-load">HuggingFace Dataset</a></li>
        <li><a href="#xml-load">XML Files</a></li>
        <li><a href="#separted-images-load">Separated Images</a></li>
      </ul>
    </li>
    <li><a href="#determine-classes">Determine Classes</a></li>
    <li><a href="#initialize-model">Initialize Model</a></li>
    <li><a href="#create-custom-dataset">Create Custom Dataset</a></li>
    <li><a href="#create-dataloader">Create Dataloader</a></li>
    <li><a href="#calculate-accuracy">Calculate Accuracy</a></li>
    <li><a href="#visualize-boxes">Visualize Boxes</a></li>
  </ol>
</details>

# Starter Imports

```python
# Model Building
import torch
from torchvision import transforms
from torchvision.models import detection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Data Extraction
from PIL import Image # Image editing
import io
from datasets import load_dataset

# Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
```

# Set Device Type

```python
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
```

# Load Dataset

For examples in loading various dataset types, please refer to [FinetunePyTorchModel.md](FinetunePyTorchModel.md).

# Determine Classes

For examples in determining classes, please refer to [FinetunePyTorchModel.md](FinetunePyTorchModel.md).

# Initialize Model

This method creates a dictionary to allow the developer to quickly switch between testing various pretrained PyTorch models. 

```python
# Fill dictionary with models you wish to test
models = {
    "frcnn-resnet": detection.fasterrcnn_resnet50_fpn,
    "frcnn-mobilenet": detection.fasterrcnn_mobilenet_v3_large_320_fpn,
}

# Load the desired model
model = models["frcnn-mobilenet"](weights="DEFAULT").to(device)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, number_of_classes)

# NOTE: Make sure that the local weights correspond to the selected model
model.load_state_dict(torch.load('LocalFinetunedWeights.pt'))
```

# Generate Target

During training, generate_target will be called to process data and make any final format changes to prepare it for finetuning the model. This example helper function returns two items:
- target: a dictionary which combines the bounding boxes and classes/labels associated with the input image
- img: The input image loaded and reformatted to a tuple of torch tensors

This example assumes that you are working with a pickle file generated from [PreprocessData.md](PreprocessData.md). 

```python
def generate_target(object):

    # Bounding boxes must be in Pascal VOC format, [xmin, ymin, xmax, ymax]
    boxes = object['annotation']
    boxes = torch.as_tensor(boxes, dtype=torch.float32)

    # Access label dictionary to map metadata to integers
    labels = []
    for label in object['metadata']:
        labels.append(label_dict[label])
    labels = torch.as_tensor(labels, dtype=torch.int64)

    # Convert image from Byte to Tensor
    img = Image.open(io.BytesIO(object['image'])).convert("RGB")
    data_transform = transforms.Compose([transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float),])
    img = data_transform(img).to(device)

    # Create target dictionary
    target = {}
    target["boxes"] = boxes.to(device)
    target["labels"] = labels.to(device)

    return target, img
```

# Create Custom Dataset

PyTorch supports two main types of datasets in its Dataloader object: Map-style and Iterable-style. We create a custom dataset based on our loaded dataset to take advantage of this object in training. Preferably, I work with Map-style datasets.

Map-style Dataset:
- Implements __getitem__() and __len__() methods
- Useful on locally stored datasets which are fully loaded in

Iterable-style Dataset:
- Implements __len__() method
- Useful when streaming data or when a large dataset cannot be fully loaded into memory

```python
# Create Mapstyle dataset assuming that you are workign with Pandas dataframe
class CustomMapStyleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    # Use helper function generated earlier to process data
    def __getitem__(self, idx):
        target, image = generate_target(self.data.loc[idx])
        return target, image

    def __len__(self):
        return len(self.data.index)
```

# Create Dataloader

Initialize a PyTorch Dataloader object to handle data batching and processing functions during training. The Dataloader object recommends using a custom collate function to group batched data. I include one that works with the Pandas dataset we have used throughout the tutorial. Play around with the batch_size:
- Higher batch_size: More memory usage, Faster training, Less accurate training
- Lower batch_size: Less memory, Faster than higher batch_size if gpu speeds are throttled by high memory usage, More accurate training

```python
def collate_fn(batch):
    return tuple(zip(*batch))

test_dl = torch.utils.data.DataLoader(dataset=CustomImageDataset(dataset),
                                       batch_size=8,
                                       collate_fn=collate_fn,
                                       )
```

# Calculate Accuracy

In order to calculate the accuracy of an object detection model I recommend following these two steps:
1. Create a helper function that runs an intersection over union calculation to determine how closely our predicted boxes match the ground truth data
2. Iterate over all testing data, run our model on each batch in evaluation mode, then call our helper function, and finally tally all of the scores to find the average IoU score. Ideally this number is >.5. Anything greater than .75 is a very effective model. An average IoU of 1 would mean that our model is perfect and something has gone horribly wrong.

```python
def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    # Calculate the area of both bounding boxes
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    # Calculate union
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area

# Iterate over testing dataset and calculate Intersection over Union score
total_iou = 0.0
num_boxes = 0
model.eval() # Set model to eval mode
with torch.no_grad():
    for targets, imgs in tqdm(test_dl):
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
        # Generate model predictions
        preds = model(imgs)
        
        for i, output in enumerate(preds):
            predicted_boxes = output['boxes'].cpu().numpy()
            true_boxes = annotations[i]['boxes'].cpu().numpy()

            for true_box in true_boxes:
                best_iou = 0
                for pred_box in predicted_boxes:
                    iou = calculate_iou(true_box, pred_box)
                    if iou > best_iou:
                        best_iou = iou

                total_iou += best_iou
                num_boxes += 1
    
    avg_iou = total_iou / num_boxes if num_boxes > 0 else 0
    print(f"Average IoU: {avg_iou}")
```

# Visualize Boxes

In order to visualize our images with box predictions drawn around the objects, I recommend this helper function plot_image(). Using this function we can take any batch of data from our test dataset, run our model in eval() mode with torch.no_grad() to predict bounding boxes, and then visualize them compared to the ground truth.

```python
def plot_image(img_tensor, annotation, isPred):

    fig,ax = plt.subplots(1)
    img = img_tensor.cpu().data

    # Display the image
    ax.imshow(img.permute(1, 2, 0))

    i = 0
    for box in annotation["boxes"]:
        
        # This block allows us to only visualize predictions that are given a confidence score >= 0.5 or 50%
        if isPred:
            if annotation['scores'][i].item() < 0.5:
                i += 1
                continue
            i += 1
                
        box = box.cpu() # Use if running cuda
        xmin, ymin, xmax, ymax = box.detach().numpy()

        # Create a Rectangle patch
        rect = patches.Rectangle((xmin.item(),ymin.item()),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='r',facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()

# Make sure to set the model to evaluation mode and run predictions with torch.no_grad()
model.eval()
with torch.no_grad():
    for targets, imgs in tqdm(test_dl):
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in targets]
        break # Simplify visualization by only grabbing the first batch
    preds = model(imgs) # Get predictions

# Output images with boxes drawn
IMG_NUM = 3 # [0, batch_size - 1]
print("Prediction")
plot_image(imgs[IMG_NUM], preds[IMG_NUM], True)
print("Target")
plot_image(imgs[IMG_NUM], annotations[IMG_NUM], False)
```