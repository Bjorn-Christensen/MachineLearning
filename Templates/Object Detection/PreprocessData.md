# About

This template is designed to efficiently preprocess large datasets, particularly images, with the goal of minimizing memory usage while training models. In addition to images, metadata (bounding boxes, labels, etc.) can be preprocessed and converted to the desired format.
All processed data is then stored in a Pandas DataFrame and saved as a .pkl file to preserve Python data structures and data types.

By preprocessing data in this manner, you can significantly reduce training time and enable the use of larger datasets on older or less powerful hardware. Additionally, this approach allows you to standardize input data so that model training code can be reused with various datasets.

In order to simplify this tutorial, it is written under the assumption you are loading in a dataset from a .pkl file that stores images as Bytes data and bounding boxes in Pascal VOC format with normalized points. For examples of working with other file types, refer to <a href='#notable-scenarios'>Notable Scenarios.</a>

# Program Outline

## Starter Imports:

```python
# Data Extraction
from PIL import Image
import io
import pandas as pd

# QoL
from tqdm import tqdm
```

## Create your Preprocessing class:

- Initialize your dataframe with desired columns such as 'images', 'annotations', 'metadata'
- Load your dataset
- Create methods to transform data for each of the desired columns

```python
class Preprocess():

    def __init__(self, DESIRED_IMAGE_SIZE=(400, 400), DESIRED_FILE_NAME='transformed_data.pkl'):
        self.dataframe = pd.DataFrame(columns = ['images', 'annotations']) # Initialize your Dataframe
        self.dataset = pd.read_pickle('pickleddata.pkl') # Load your Dataset
        self.DESIRED_IMAGE_SIZE = DESIRED_IMAGE_SIZE
        self.DESIRED_FILE_NAME = DESIRED_FILE_NAME
    
    
    """
    Load images and resize them before converting (back) to bytes data in JPEG format
    """
    def transform_image(self, row):
        img = Image.open(io.BytesIO(row['images'])).convert('RGB')
        img = img.resize(self.DESIRED_IMAGE_SIZE, Image.Resampling.LANCZOS)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')

        return img_byte_arr.getvalue()


    """
    Resize bounding boxes in relation to desired image size
    """
    def transform_annotation(self, row):
        boxes = []
        for box in row['annotations']:
            xmin = box[0] * self.DESIRED_IMAGE_SIZE[0]
            ymin = box[1] * self.DESIRED_IMAGE_SIZE[1]
            xmax = box[2] * self.DESIRED_IMAGE_SIZE[0]
            ymax = box[3] * self.DESIRED_IMAGE_SIZE[1]
            boxes.append([xmin, ymin, xmax, ymax])

        return boxes


    """
    Call transform methods and build dataframe
    """
    def preprocess(self):
        for idx, row in tqdm(self.dataset):
            image = self.transform_image(row)
            annotation = self.transform_annotation(row)
            transformed_data = [image, annotation]
            data_to_add = pd.DataFrame([transformed_data], columns=self.dataframe.columns)
            self.dataframe = pd.concat([self.dataframe, data_to_add], ignore_index=True)
        
        return self.dataframe
    

    """
    Save data as .pkl file 
    """
    def pickle_data(self):
        self.dataframe.to_pickle(self.DESIRED_FILE_NAME)
```   

## Run the program:

```python
DESIRED_IMAGE_SIZE = (360, 500)
DESIRED_FILE_NAME = 'transformed_data.pkl'
pp = Preprocess(DESIRED_IMAGE_SIZE, DESIRED_FILE_NAME)
pp.preprocess()
pp.pickle_data()
```
# Notable Scenarios

## Images and Metadata are separated:

- Create list of image file pathways and use idx value from .iterrows() to parse this list

```python
images = list(os.listdir("archive/train_images/"))
```

## Data stored in XML File:

- Use BeautifulSoup to parse through xml file data

```python
from bs4 import BeautifulSoup

with open(file) as f:
        data = f.read()
        soup = BeautifulSoup(data, 'xml')
        objects = soup.find_all('object')

        for obj in objects:
            xmin = int(obj.find('xmin').text) # Extract bbox point
            label = obj.find('name') # Extract label
```

## Using HuggingFace:

- Use HuggingFace's load_dataset function

```python
from datasets import load_dataset

# Example with parquet file
data_files = {"train": "train*", "test": "test*"}
dataset = load_dataset("parquet", data_dir="C:\\Users\\User\\dataset_name\\data\\", data_files=data_files, split="train[:20%]")
```