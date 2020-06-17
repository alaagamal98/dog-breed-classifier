# Convolutional Neural Networks

## Project: Write an Algorithm for a Dog Identification App 

---

In this notebook, some template code has already been provided for you, and you will need to implement additional functionality to successfully complete this project. You will not need to modify the included code beyond what is requested. Sections that begin with **'(IMPLEMENTATION)'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section, and the specifics of the implementation are marked in the code block with a 'TODO' statement. Please be sure to read the instructions carefully! 

> **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the Jupyter Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to **File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.

>**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut.  Markdown cells can be edited by double-clicking the cell to enter edit mode.

The rubric contains _optional_ "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. If you decide to pursue the "Stand Out Suggestions", you should include the code in this Jupyter notebook.



---
### Why We're Here 

In this notebook, you will make the first steps towards developing an algorithm that could be used as part of a mobile or web app.  At the end of this project, your code will accept any user-supplied image as input.  If a dog is detected in the image, it will provide an estimate of the dog's breed.  If a human is detected, it will provide an estimate of the dog breed that is most resembling.  The image below displays potential sample output of your finished project (... but we expect that each student's algorithm will behave differently!). 

![Sample Dog Output](images/sample_dog_output.png)

In this real-world setting, you will need to piece together a series of models to perform different tasks; for instance, the algorithm that detects humans in an image will be different from the CNN that infers dog breed.  There are many points of possible failure, and no perfect algorithm exists.  Your imperfect solution will nonetheless create a fun user experience!

### The Road Ahead

We break the notebook into separate steps.  Feel free to use the links below to navigate the notebook.

* [Step 0](#step0): Import Datasets
* [Step 1](#step1): Detect Humans
* [Step 2](#step2): Detect Dogs
* [Step 3](#step3): Create a CNN to Classify Dog Breeds (from Scratch)
* [Step 4](#step4): Create a CNN to Classify Dog Breeds (using Transfer Learning)
* [Step 5](#step5): Write your Algorithm
* [Step 6](#step6): Test Your Algorithm

---
<a id='step0'></a>
## Step 0: Import Datasets

Make sure that you've downloaded the required human and dog datasets:

**Note: if you are using the Udacity workspace, you *DO NOT* need to re-download these - they can be found in the `/data` folder as noted in the cell below.**

* Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in this project's home directory, at the location `/dog_images`. 

* Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).  Unzip the folder and place it in the home directory, at location `/lfw`.  

*Note: If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder.*

In the code cell below, we save the file paths for both the human (LFW) dataset and dog dataset in the numpy arrays `human_files` and `dog_files`.


```python
import numpy as np
from glob import glob

# load filenames for human and dog images
human_files = np.array(glob("/data/lfw/*/*"))
dog_files = np.array(glob("/data/dog_images/*/*/*"))

# print number of images in each dataset
print('There are %d total human images.' % len(human_files))
print('There are %d total dog images.' % len(dog_files))
```

    There are 13233 total human images.
    There are 8351 total dog images.


<a id='step1'></a>
## Step 1: Detect Humans

In this section, we use OpenCV's implementation of [Haar feature-based cascade classifiers](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) to detect human faces in images.  

OpenCV provides many pre-trained face detectors, stored as XML files on [github](https://github.com/opencv/opencv/tree/master/data/haarcascades).  We have downloaded one of these detectors and stored it in the `haarcascades` directory.  In the next code cell, we demonstrate how to use this detector to find human faces in a sample image.


```python
import cv2                
import matplotlib.pyplot as plt                        
%matplotlib inline                               

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# load color (BGR) image
img = cv2.imread(human_files[0])
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = face_cascade.detectMultiScale(gray)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x,y,w,h) in faces:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()
```

    Number of faces detected: 1



![png](images/output_3_1.png)


Before using any of the face detectors, it is standard procedure to convert the images to grayscale.  The `detectMultiScale` function executes the classifier stored in `face_cascade` and takes the grayscale image as a parameter.  

In the above code, `faces` is a numpy array of detected faces, where each row corresponds to a detected face.  Each detected face is a 1D array with four entries that specifies the bounding box of the detected face.  The first two entries in the array (extracted in the above code as `x` and `y`) specify the horizontal and vertical positions of the top left corner of the bounding box.  The last two entries in the array (extracted here as `w` and `h`) specify the width and height of the box.

### Write a Human Face Detector

We can use this procedure to write a function that returns `True` if a human face is detected in an image and `False` otherwise.  This function, aptly named `face_detector`, takes a string-valued file path to an image as input and appears in the code block below.


```python
# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0
```

### (IMPLEMENTATION) Assess the Human Face Detector

__Question 1:__ Use the code cell below to test the performance of the `face_detector` function.  
- What percentage of the first 100 images in `human_files` have a detected human face?  
- What percentage of the first 100 images in `dog_files` have a detected human face? 

Ideally, we would like 100% of human images with a detected face and 0% of dog images with a detected face.  You will see that our algorithm falls short of this goal, but still gives acceptable performance.  We extract the file paths for the first 100 images from each of the datasets and store them in the numpy arrays `human_files_short` and `dog_files_short`.

__Answer:__ 
(You can print out your results and/or write your percentages in this cell)


```python
from tqdm import tqdm

human_files_short = human_files[:100]
dog_files_short = dog_files[:100]

#-#-# Do NOT modify the code above this line. #-#-#
humans=0
dogs=0
for img in human_files_short:
    if face_detector(img) == True:
        humans=humans+1
for img in dog_files_short:
    if face_detector(img) == True:
        dogs=dogs+1
        
## TODO: Test the performance of the face_detector algorithm 
## on the images in human_files_short and dog_files_short.
print('%.1f%% Humans:' % humans)
print('%.1f%% Dogs:' % dogs)
```

    98.0% Humans:
    17.0% Dogs:


We suggest the face detector from OpenCV as a potential way to detect human images in your algorithm, but you are free to explore other approaches, especially approaches that make use of deep learning :).  Please use the code cell below to design and test your own face detection algorithm.  If you decide to pursue this _optional_ task, report performance on `human_files_short` and `dog_files_short`.


```python
### (Optional) 
### TODO: Test performance of anotherface detection algorithm.
### Feel free to use as many code cells as needed.
```

---
<a id='step2'></a>
## Step 2: Detect Dogs

In this section, we use a [pre-trained model](http://pytorch.org/docs/master/torchvision/models.html) to detect dogs in images.  

### Obtain Pre-trained VGG-16 Model

The code cell below downloads the VGG-16 model, along with weights that have been trained on [ImageNet](http://www.image-net.org/), a very large, very popular dataset used for image classification and other vision tasks.  ImageNet contains over 10 million URLs, each linking to an image containing an object from one of [1000 categories](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a).  


```python
import torch
import torchvision.models as models

# define VGG16 model
VGG16 = models.vgg16(pretrained=True)

# check if CUDA is available
use_cuda = torch.cuda.is_available()

# move model to GPU if CUDA is available
if use_cuda:
    VGG16 = VGG16.cuda()
```

    Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" to /root/.torch/models/vgg16-397923af.pth
    100%|██████████| 553433881/553433881 [00:07<00:00, 75677578.40it/s]


Given an image, this pre-trained VGG-16 model returns a prediction (derived from the 1000 possible categories in ImageNet) for the object that is contained in the image.

### (IMPLEMENTATION) Making Predictions with a Pre-trained Model

In the next code cell, you will write a function that accepts a path to an image (such as `'dogImages/train/001.Affenpinscher/Affenpinscher_00001.jpg'`) as input and returns the index corresponding to the ImageNet class that is predicted by the pre-trained VGG-16 model.  The output should always be an integer between 0 and 999, inclusive.

Before writing the function, make sure that you take the time to learn  how to appropriately pre-process tensors for pre-trained models in the [PyTorch documentation](http://pytorch.org/docs/stable/torchvision/models.html).


```python
from PIL import Image
import torchvision.transforms as transforms

def VGG16_predict(img_path):
    '''
    Use pre-trained VGG-16 model to obtain index corresponding to 
    predicted ImageNet class for image at specified path
    
    Args:
        img_path: path to an image
        
    Returns:
        Index corresponding to VGG-16 model's prediction
    '''
    
    ## TODO: Complete the function.
    ## Load and pre-process an image from the given img_path
    ## Return the *index* of the predicted class for that image
    
    img = Image.open(img_path).convert('RGB')
    
    transform = transforms.Compose([
                        transforms.Resize(size=(244, 244)),
                        transforms.ToTensor()])
    img = transform(img)[:3,:,:].unsqueeze(0)
    if use_cuda:
        img = img.cuda()
    ret = VGG16(img)
    return torch.max(ret,1)[1].item() # predicted class index
```

### (IMPLEMENTATION) Write a Dog Detector

While looking at the [dictionary](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a), you will notice that the categories corresponding to dogs appear in an uninterrupted sequence and correspond to dictionary keys 151-268, inclusive, to include all categories from `'Chihuahua'` to `'Mexican hairless'`.  Thus, in order to check to see if an image is predicted to contain a dog by the pre-trained VGG-16 model, we need only check if the pre-trained model predicts an index between 151 and 268 (inclusive).

Use these ideas to complete the `dog_detector` function below, which returns `True` if a dog is detected in an image (and `False` if not).


```python
### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    ## TODO: Complete the function.
    index = VGG16_predict(img_path)
    return index >= 151 and index <= 268 # true/false
```

### (IMPLEMENTATION) Assess the Dog Detector

__Question 2:__ Use the code cell below to test the performance of your `dog_detector` function.  
- What percentage of the images in `human_files_short` have a detected dog?  
- What percentage of the images in `dog_files_short` have a detected dog?

__Answer:__ 



```python
### TODO: Test the performance of the dog_detector function
### on the images in human_files_short and dog_files_short.
def dog_test(files):
    dogs = 0;
    total = len(files)
    for file in files:
        dogs += dog_detector(file)
        percentage = (dogs/total)*100
    return percentage

print("%.1f%% Dogs in human_files:"%dog_test(human_files_short))
print("%.1f%% Dogs in dog_files:"%dog_test(dog_files_short))

```

    0.0% Dogs in human_files:
    99.0% Dogs in dog_files:


We suggest VGG-16 as a potential network to detect dog images in your algorithm, but you are free to explore other pre-trained networks (such as [Inception-v3](http://pytorch.org/docs/master/torchvision/models.html#inception-v3), [ResNet-50](http://pytorch.org/docs/master/torchvision/models.html#id3), etc).  Please use the code cell below to test other pre-trained PyTorch models.  If you decide to pursue this _optional_ task, report performance on `human_files_short` and `dog_files_short`.


```python
### (Optional) 
### TODO: Report the performance of another pre-trained network.
### Feel free to use as many code cells as needed.
```

---
<a id='step3'></a>
## Step 3: Create a CNN to Classify Dog Breeds (from Scratch)

Now that we have functions for detecting humans and dogs in images, we need a way to predict breed from images.  In this step, you will create a CNN that classifies dog breeds.  You must create your CNN _from scratch_ (so, you can't use transfer learning _yet_!), and you must attain a test accuracy of at least 10%.  In Step 4 of this notebook, you will have the opportunity to use transfer learning to create a CNN that attains greatly improved accuracy.

We mention that the task of assigning breed to dogs from images is considered exceptionally challenging.  To see why, consider that *even a human* would have trouble distinguishing between a Brittany and a Welsh Springer Spaniel.  

Brittany | Welsh Springer Spaniel
- | - 
<img src="images/Brittany_02625.jpg" width="100"> | <img src="images/Welsh_springer_spaniel_08203.jpg" width="200">

It is not difficult to find other dog breed pairs with minimal inter-class variation (for instance, Curly-Coated Retrievers and American Water Spaniels).  

Curly-Coated Retriever | American Water Spaniel
- | -
<img src="images/Curly-coated_retriever_03896.jpg" width="200"> | <img src="images/American_water_spaniel_00648.jpg" width="200">


Likewise, recall that labradors come in yellow, chocolate, and black.  Your vision-based algorithm will have to conquer this high intra-class variation to determine how to classify all of these different shades as the same breed.  

Yellow Labrador | Chocolate Labrador | Black Labrador
- | -
<img src="images/Labrador_retriever_06457.jpg" width="150"> | <img src="images/Labrador_retriever_06455.jpg" width="240"> | <img src="images/Labrador_retriever_06449.jpg" width="220">

We also mention that random chance presents an exceptionally low bar: setting aside the fact that the classes are slightly imabalanced, a random guess will provide a correct answer roughly 1 in 133 times, which corresponds to an accuracy of less than 1%.  

Remember that the practice is far ahead of the theory in deep learning.  Experiment with many different architectures, and trust your intuition.  And, of course, have fun!

### (IMPLEMENTATION) Specify Data Loaders for the Dog Dataset

Use the code cell below to write three separate [data loaders](http://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for the training, validation, and test datasets of dog images (located at `dog_images/train`, `dog_images/valid`, and `dog_images/test`, respectively).  You may find [this documentation on custom datasets](http://pytorch.org/docs/stable/torchvision/datasets.html) to be a useful resource.  If you are interested in augmenting your training and/or validation data, check out the wide variety of [transforms](http://pytorch.org/docs/stable/torchvision/transforms.html?highlight=transform)!


```python
import os
from torchvision import datasets
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


### TODO: Write data loaders for training, validation, and test sets
## Specify appropriate transforms, and batch_sizes


num_workers = 0
batch_size = 20
valid_size = 0.2

data_dir ='/data/dog_images/'
train_dir = os.path.join(data_dir,'train/')
valid_dir = os.path.join(data_dir,'valid/')
test_dir = os.path.join(data_dir,'test/')

standard_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

data_transforms = {'train': transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     standard_normalization]),
                   'val': transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     standard_normalization]),
                   'test': transforms.Compose([transforms.Resize(size=(224,224)),
                                     transforms.ToTensor(), 
                                     standard_normalization])
                  }

train_data = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
valid_data = datasets.ImageFolder(valid_dir, transform=data_transforms['val'])
test_data = datasets.ImageFolder(test_dir, transform=data_transforms['test'])

train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=batch_size, 
                                           num_workers=num_workers,
                                           shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data,
                                           batch_size=batch_size, 
                                           num_workers=num_workers,
                                           shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data,
                                           batch_size=batch_size, 
                                           num_workers=num_workers,
                                           shuffle=False)

loaders_scratch = {
    'train': train_loader,
    'valid': valid_loader,
    'test': test_loader
}
```

**Question 3:** Describe your chosen procedure for preprocessing the data. 
- How does your code resize the images (by cropping, stretching, etc)?  What size did you pick for the input tensor, and why?
- Did you decide to augment the dataset?  If so, how (through translations, flips, rotations, etc)?  If not, why not?


**Answer**: I've applied RandomResizedCrop and RandomHorizontalFlip to the training set to help training the model with all sorts of images from different angles and sizes, and on the validation set i only applied resizing and central cropping without augmentation since it will be used for validation check only, and on the test set i only applied resizing.

### (IMPLEMENTATION) Model Architecture

Create a CNN to classify dog breed.  Use the template in the code cell below.


```python
import torch.nn as nn
import torch.nn.functional as F


# define the CNN architecture
class Net(nn.Module):
    ### TODO: choose an architecture, and complete the class
    def __init__(self):
        super(Net, self).__init__()
        ## Define layers of a CNN
            
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        # pool
        self.pool = nn.MaxPool2d(2, 2)
        
        # fully-connected
        self.fc1 = nn.Linear(7*7*128, 500)
        self.fc2 = nn.Linear(500, 133) 
        
        # drop-out
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        ## Define forward behavior
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        x = x.view(-1, 7*7*128)
        
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        
        x = self.dropout(x)
        x = self.fc2(x)
        return x

#-#-# You so NOT have to modify the code below this line. #-#-#

# instantiate the CNN
model_scratch = Net()

# move tensors to GPU if CUDA is available
if use_cuda:
    model_scratch.cuda()
```

__Question 4:__ Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  

__Answer:__ 
(conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

activation: relu

(pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

activation: relu

(conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

activation: relu

(pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

(conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

(pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

(dropout): Dropout(p=0.3)

(fc1): Linear(in_features=6272, out_features=500, bias=True)

(dropout): Dropout(p=0.3)

(fc2): Linear(in_features=500, out_features=133, bias=True)


I've applied the first 2 conv layers with kernal size of 3 and stirde of 2 in order to downsize the input image by 2 and after each one of them i applied max pooling with stirde of 2 to downsize the image by 2, the final conv layer with a kernal size of 3 and a stride of 1 which won't reduce the input size then a final max pooling with a stride of 2, which means the output image size will be downsized by a factor of 2*2*2*2*2(32) and the depth 128. and then appling a dropout of 0.3 to avoid overfitting. then applying two fully connected layers to produce final output size which is resposible for predicting the class of breeds.


### (IMPLEMENTATION) Specify Loss Function and Optimizer

Use the next code cell to specify a [loss function](http://pytorch.org/docs/stable/nn.html#loss-functions) and [optimizer](http://pytorch.org/docs/stable/optim.html).  Save the chosen loss function as `criterion_scratch`, and the optimizer as `optimizer_scratch` below.


```python
import torch.optim as optim

### TODO: select loss function
criterion_scratch =  nn.CrossEntropyLoss()

### TODO: select optimizer
optimizer_scratch = optim.SGD(model_scratch.parameters(), lr = 0.05)
```

### (IMPLEMENTATION) Train and Validate the Model

Train and validate your model in the code cell below.  [Save the final model parameters](http://pytorch.org/docs/master/notes/serialization.html) at filepath `'model_scratch.pt'`.


```python
def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path, last_validation_loss=None):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    if last_validation_loss is not None:
        valid_loss_min = last_validation_loss
    else:
        valid_loss_min = np.Inf
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

            # initialize weights to zero
            optimizer.zero_grad()
            
            output = model(data)
            
            # calculate loss
            loss = criterion(output, target)
            
            # back prop
            loss.backward()
            
            # grad
            optimizer.step()
            
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
            if batch_idx % 100 == 0:
                print('Epoch %d, Batch %d loss: %.6f' %
                  (epoch, batch_idx + 1, train_loss))
            
        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            output = model(data)
            loss = criterion(output, target)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        
        ## TODO: save the model if validation loss has decreased
        if valid_loss < valid_loss_min:
            torch.save(model.state_dict(), save_path)
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            valid_loss_min = valid_loss
            
    # return trained model
    return model

# train the model
model_scratch = train(10, loaders_scratch, model_scratch, optimizer_scratch, 
                      criterion_scratch, use_cuda, 'model_scratch.pt')

# load the model that got the best validation accuracy
model_scratch.load_state_dict(torch.load('model_scratch.pt'))
```

    Epoch 1, Batch 1 loss: 4.233591
    Epoch 1, Batch 101 loss: 4.043358
    Epoch 1, Batch 201 loss: 4.070776
    Epoch 1, Batch 301 loss: 4.070978
    Epoch: 1 	Training Loss: 4.067752 	Validation Loss: 3.873895
    Validation loss decreased (inf --> 3.873895).  Saving model ...
    Epoch 2, Batch 1 loss: 4.255459
    Epoch 2, Batch 101 loss: 4.021764
    Epoch 2, Batch 201 loss: 4.021390
    Epoch 2, Batch 301 loss: 4.024148
    Epoch: 2 	Training Loss: 4.024418 	Validation Loss: 3.874192
    Epoch 3, Batch 1 loss: 4.111045
    Epoch 3, Batch 101 loss: 3.909786
    Epoch 3, Batch 201 loss: 3.944284
    Epoch 3, Batch 301 loss: 3.964598
    Epoch: 3 	Training Loss: 3.963856 	Validation Loss: 3.877568
    Epoch 4, Batch 1 loss: 4.241046
    Epoch 4, Batch 101 loss: 3.933752
    Epoch 4, Batch 201 loss: 3.905021
    Epoch 4, Batch 301 loss: 3.905789
    Epoch: 4 	Training Loss: 3.893104 	Validation Loss: 3.759520
    Validation loss decreased (3.873895 --> 3.759520).  Saving model ...
    Epoch 5, Batch 1 loss: 3.776469
    Epoch 5, Batch 101 loss: 3.848573
    Epoch 5, Batch 201 loss: 3.838690
    Epoch 5, Batch 301 loss: 3.856102
    Epoch: 5 	Training Loss: 3.866128 	Validation Loss: 3.778844
    Epoch 6, Batch 1 loss: 3.974682
    Epoch 6, Batch 101 loss: 3.798939
    Epoch 6, Batch 201 loss: 3.829973
    Epoch 6, Batch 301 loss: 3.835585
    Epoch: 6 	Training Loss: 3.831437 	Validation Loss: 3.582877
    Validation loss decreased (3.759520 --> 3.582877).  Saving model ...
    Epoch 7, Batch 1 loss: 4.244926
    Epoch 7, Batch 101 loss: 3.750094
    Epoch 7, Batch 201 loss: 3.773582
    Epoch 7, Batch 301 loss: 3.787719
    Epoch: 7 	Training Loss: 3.784381 	Validation Loss: 3.692457
    Epoch 8, Batch 1 loss: 4.035285
    Epoch 8, Batch 101 loss: 3.719209
    Epoch 8, Batch 201 loss: 3.750762
    Epoch 8, Batch 301 loss: 3.744146
    Epoch: 8 	Training Loss: 3.739298 	Validation Loss: 3.596233
    Epoch 9, Batch 1 loss: 3.796192
    Epoch 9, Batch 101 loss: 3.663942
    Epoch 9, Batch 201 loss: 3.690634
    Epoch 9, Batch 301 loss: 3.710213
    Epoch: 9 	Training Loss: 3.714258 	Validation Loss: 3.567645
    Validation loss decreased (3.582877 --> 3.567645).  Saving model ...
    Epoch 10, Batch 1 loss: 3.316198
    Epoch 10, Batch 101 loss: 3.606951
    Epoch 10, Batch 201 loss: 3.624535
    Epoch 10, Batch 301 loss: 3.643162
    Epoch: 10 	Training Loss: 3.638007 	Validation Loss: 3.631431


### (IMPLEMENTATION) Test the Model

Try out your model on the test dataset of dog images.  Use the code cell below to calculate and print the test loss and accuracy.  Ensure that your test accuracy is greater than 10%.


```python
def test(loaders, model, criterion, use_cuda):

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))

# call test function    
test(loaders_scratch, model_scratch, criterion_scratch, use_cuda)
```

    Test Loss: 3.714035
    
    
    Test Accuracy: 14% (123/836)


---
<a id='step4'></a>
## Step 4: Create a CNN to Classify Dog Breeds (using Transfer Learning)

You will now use transfer learning to create a CNN that can identify dog breed from images.  Your CNN must attain at least 60% accuracy on the test set.

### (IMPLEMENTATION) Specify Data Loaders for the Dog Dataset

Use the code cell below to write three separate [data loaders](http://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader) for the training, validation, and test datasets of dog images (located at `dogImages/train`, `dogImages/valid`, and `dogImages/test`, respectively). 

If you like, **you are welcome to use the same data loaders from the previous step**, when you created a CNN from scratch.


```python
## TODO: Specify data loaders
loaders_transfer = loaders_scratch.copy()

```

### (IMPLEMENTATION) Model Architecture

Use transfer learning to create a CNN to classify dog breed.  Use the code cell below, and save your initialized model as the variable `model_transfer`.


```python
import torchvision.models as models
import torch.nn as nn

## TODO: Specify model architecture 
model_transfer = models.vgg16(pretrained=True)

for param in model_transfer.features.parameters():
    param.requires_grad = False
    
model_transfer.fc = nn.Linear(2048, 133, bias=True)

fc_parameters = model_transfer.fc.parameters()


for param in fc_parameters:
    param.requires_grad = True
    

if use_cuda:
    model_transfer = model_transfer.cuda()
```

__Question 5:__ Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  Describe why you think the architecture is suitable for the current problem.

__Answer:__ 
I picked VGG16 as a transfer model because it performed greatly on image classification. I've pull out the final fully connected layer and replaced with fully connected layer with output of 133 (dog breed classes)

### (IMPLEMENTATION) Specify Loss Function and Optimizer

Use the next code cell to specify a [loss function](http://pytorch.org/docs/master/nn.html#loss-functions) and [optimizer](http://pytorch.org/docs/master/optim.html).  Save the chosen loss function as `criterion_transfer`, and the optimizer as `optimizer_transfer` below.


```python
criterion_transfer = nn.CrossEntropyLoss()
optimizer_transfer = optim.SGD(model_transfer.classifier.parameters(), lr=0.001)
```

### (IMPLEMENTATION) Train and Validate the Model

Train and validate your model in the code cell below.  [Save the final model parameters](http://pytorch.org/docs/master/notes/serialization.html) at filepath `'model_transfer.pt'`.


```python
# train the model
model_transfer = train(20, loaders_transfer, model_transfer, optimizer_transfer, criterion_transfer, use_cuda, 'model_transfer.pt')

# load the model that got the best validation accuracy (uncomment the line below)
#model_transfer.load_state_dict(torch.load('model_transfer.pt'))

def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()

            # initialize weights to zero
            optimizer.zero_grad()
            
            output = model(data)
            
            # calculate loss
            loss = criterion(output, target)
            
            # back prop
            loss.backward()
            
            # grad
            optimizer.step()
            
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
            if batch_idx % 100 == 0:
                print('Epoch %d, Batch %d loss: %.6f' %
                  (epoch, batch_idx + 1, train_loss))
        
        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            output = model(data)
            loss = criterion(output, target)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        
        ## TODO: save the model if validation loss has decreased
        if valid_loss < valid_loss_min:
            torch.save(model.state_dict(), save_path)
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            valid_loss_min = valid_loss
            
    # return trained model
    return model
```

    Epoch 1, Batch 1 loss: 23.979588
    Epoch 1, Batch 101 loss: 10.355008
    Epoch 1, Batch 201 loss: 8.773997
    Epoch 1, Batch 301 loss: 8.014283
    Epoch: 1 	Training Loss: 7.821214 	Validation Loss: 5.539572
    Validation loss decreased (inf --> 5.539572).  Saving model ...
    Epoch 2, Batch 1 loss: 5.945075
    Epoch 2, Batch 101 loss: 5.836708
    Epoch 2, Batch 201 loss: 5.652678
    Epoch 2, Batch 301 loss: 5.465755
    Epoch: 2 	Training Loss: 5.415541 	Validation Loss: 4.117146
    Validation loss decreased (5.539572 --> 4.117146).  Saving model ...
    Epoch 3, Batch 1 loss: 4.911733
    Epoch 3, Batch 101 loss: 4.676840
    Epoch 3, Batch 201 loss: 4.561588
    Epoch 3, Batch 301 loss: 4.439273
    Epoch: 3 	Training Loss: 4.406626 	Validation Loss: 3.017803
    Validation loss decreased (4.117146 --> 3.017803).  Saving model ...
    Epoch 4, Batch 1 loss: 3.800663
    Epoch 4, Batch 101 loss: 3.867542
    Epoch 4, Batch 201 loss: 3.737675
    Epoch 4, Batch 301 loss: 3.636352
    Epoch: 4 	Training Loss: 3.603680 	Validation Loss: 2.168401
    Validation loss decreased (3.017803 --> 2.168401).  Saving model ...
    Epoch 5, Batch 1 loss: 3.814232
    Epoch 5, Batch 101 loss: 3.234876
    Epoch 5, Batch 201 loss: 3.100875
    Epoch 5, Batch 301 loss: 3.036461
    Epoch: 5 	Training Loss: 3.004214 	Validation Loss: 1.578481
    Validation loss decreased (2.168401 --> 1.578481).  Saving model ...
    Epoch 6, Batch 1 loss: 3.155428
    Epoch 6, Batch 101 loss: 2.738941
    Epoch 6, Batch 201 loss: 2.633916
    Epoch 6, Batch 301 loss: 2.533401
    Epoch: 6 	Training Loss: 2.512091 	Validation Loss: 1.217155
    Validation loss decreased (1.578481 --> 1.217155).  Saving model ...
    Epoch 7, Batch 1 loss: 1.799216
    Epoch 7, Batch 101 loss: 2.290701
    Epoch 7, Batch 201 loss: 2.266810
    Epoch 7, Batch 301 loss: 2.205759
    Epoch: 7 	Training Loss: 2.187128 	Validation Loss: 0.987699
    Validation loss decreased (1.217155 --> 0.987699).  Saving model ...
    Epoch 8, Batch 1 loss: 1.969049
    Epoch 8, Batch 101 loss: 1.991368
    Epoch 8, Batch 201 loss: 1.985619
    Epoch 8, Batch 301 loss: 1.956118
    Epoch: 8 	Training Loss: 1.954924 	Validation Loss: 0.833201
    Validation loss decreased (0.987699 --> 0.833201).  Saving model ...
    Epoch 9, Batch 1 loss: 1.964506
    Epoch 9, Batch 101 loss: 1.889093
    Epoch 9, Batch 201 loss: 1.829471
    Epoch 9, Batch 301 loss: 1.825361
    Epoch: 9 	Training Loss: 1.809436 	Validation Loss: 0.734831
    Validation loss decreased (0.833201 --> 0.734831).  Saving model ...
    Epoch 10, Batch 1 loss: 2.119147
    Epoch 10, Batch 101 loss: 1.722568
    Epoch 10, Batch 201 loss: 1.709262
    Epoch 10, Batch 301 loss: 1.681187
    Epoch: 10 	Training Loss: 1.682872 	Validation Loss: 0.668563
    Validation loss decreased (0.734831 --> 0.668563).  Saving model ...
    Epoch 11, Batch 1 loss: 1.098919
    Epoch 11, Batch 101 loss: 1.669905
    Epoch 11, Batch 201 loss: 1.631730
    Epoch 11, Batch 301 loss: 1.618163
    Epoch: 11 	Training Loss: 1.620528 	Validation Loss: 0.620122
    Validation loss decreased (0.668563 --> 0.620122).  Saving model ...
    Epoch 12, Batch 1 loss: 1.581780
    Epoch 12, Batch 101 loss: 1.561854
    Epoch 12, Batch 201 loss: 1.541202
    Epoch 12, Batch 301 loss: 1.528092
    Epoch: 12 	Training Loss: 1.526284 	Validation Loss: 0.585172
    Validation loss decreased (0.620122 --> 0.585172).  Saving model ...
    Epoch 13, Batch 1 loss: 1.569381
    Epoch 13, Batch 101 loss: 1.432384
    Epoch 13, Batch 201 loss: 1.423305
    Epoch 13, Batch 301 loss: 1.438730
    Epoch: 13 	Training Loss: 1.439412 	Validation Loss: 0.551251
    Validation loss decreased (0.585172 --> 0.551251).  Saving model ...
    Epoch 14, Batch 1 loss: 1.643362
    Epoch 14, Batch 101 loss: 1.404774
    Epoch 14, Batch 201 loss: 1.376819
    Epoch 14, Batch 301 loss: 1.385150
    Epoch: 14 	Training Loss: 1.388030 	Validation Loss: 0.517180
    Validation loss decreased (0.551251 --> 0.517180).  Saving model ...
    Epoch 15, Batch 1 loss: 1.188962
    Epoch 15, Batch 101 loss: 1.373843
    Epoch 15, Batch 201 loss: 1.384334
    Epoch 15, Batch 301 loss: 1.368093
    Epoch: 15 	Training Loss: 1.355856 	Validation Loss: 0.507942
    Validation loss decreased (0.517180 --> 0.507942).  Saving model ...
    Epoch 16, Batch 1 loss: 1.334824
    Epoch 16, Batch 101 loss: 1.293775
    Epoch 16, Batch 201 loss: 1.290215
    Epoch 16, Batch 301 loss: 1.294907
    Epoch: 16 	Training Loss: 1.305526 	Validation Loss: 0.489814
    Validation loss decreased (0.507942 --> 0.489814).  Saving model ...
    Epoch 17, Batch 1 loss: 0.780848
    Epoch 17, Batch 101 loss: 1.254501
    Epoch 17, Batch 201 loss: 1.260103
    Epoch 17, Batch 301 loss: 1.262336
    Epoch: 17 	Training Loss: 1.269784 	Validation Loss: 0.473956
    Validation loss decreased (0.489814 --> 0.473956).  Saving model ...
    Epoch 18, Batch 1 loss: 1.427930
    Epoch 18, Batch 101 loss: 1.221753
    Epoch 18, Batch 201 loss: 1.267032
    Epoch 18, Batch 301 loss: 1.244485
    Epoch: 18 	Training Loss: 1.240398 	Validation Loss: 0.467652
    Validation loss decreased (0.473956 --> 0.467652).  Saving model ...
    Epoch 19, Batch 1 loss: 1.248682
    Epoch 19, Batch 101 loss: 1.218971
    Epoch 19, Batch 201 loss: 1.236947
    Epoch 19, Batch 301 loss: 1.203505
    Epoch: 19 	Training Loss: 1.200258 	Validation Loss: 0.447789
    Validation loss decreased (0.467652 --> 0.447789).  Saving model ...
    Epoch 20, Batch 1 loss: 1.102723
    Epoch 20, Batch 101 loss: 1.237801
    Epoch 20, Batch 201 loss: 1.228645
    Epoch 20, Batch 301 loss: 1.203686
    Epoch: 20 	Training Loss: 1.203212 	Validation Loss: 0.439698
    Validation loss decreased (0.447789 --> 0.439698).  Saving model ...


### (IMPLEMENTATION) Test the Model

Try out your model on the test dataset of dog images. Use the code cell below to calculate and print the test loss and accuracy.  Ensure that your test accuracy is greater than 60%.


```python
test(loaders_transfer, model_transfer, criterion_transfer, use_cuda)
```

    Test Loss: 0.536898
    
    
    Test Accuracy: 83% (696/836)


### (IMPLEMENTATION) Predict Dog Breed with the Model

Write a function that takes an image path as input and returns the dog breed (`Affenpinscher`, `Afghan hound`, etc) that is predicted by your model.  


```python
### TODO: Write a function that takes a path to an image as input
### and returns the dog breed that is predicted by the model.

# list of class names by index, i.e. a name can be accessed like class_names[0]
class_names = [item[4:].replace("_", " ") for item in loaders_transfer['train'].dataset.classes]

def predict_breed_transfer(model, class_names, img_path):
    # load the image and return the predicted breed
    
    img = Image.open(img_path).convert('RGB')
    prediction_transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                     transforms.ToTensor(), 
                                     standard_normalization])

    img = prediction_transform(img)[:3,:,:].unsqueeze(0)    
    model = model.cpu()
    model.eval()
    idx = torch.argmax(model(img))
    return class_names[idx]
```

---
<a id='step5'></a>
## Step 5: Write your Algorithm

Write an algorithm that accepts a file path to an image and first determines whether the image contains a human, dog, or neither.  Then,
- if a __dog__ is detected in the image, return the predicted breed.
- if a __human__ is detected in the image, return the resembling dog breed.
- if __neither__ is detected in the image, provide output that indicates an error.

You are welcome to write your own functions for detecting humans and dogs in images, but feel free to use the `face_detector` and `human_detector` functions developed above.  You are __required__ to use your CNN from Step 4 to predict dog breed.  

Some sample output for our algorithm is provided below, but feel free to design your own user experience!

![Sample Human Output](images/sample_human_output.png)


### (IMPLEMENTATION) Write your Algorithm


```python
### TODO: Write your algorithm.
### Feel free to use as many code cells as needed.

def run_app(img_path):
    ## handle cases for a human face, dog, and neither
    
    img = Image.open(img_path)
    plt.imshow(img)
    plt.show()
    if dog_detector(img_path) is True:
        prediction = predict_breed_transfer(model_transfer, class_names, img_path)
        print("A picture of a dog, It looks like a {0}".format(prediction))  
    elif face_detector(img_path) > 0:
        prediction = predict_breed_transfer(model_transfer, class_names, img_path)
        print("A picture of a human, He looks like  a {0}".format(prediction))
    else:
        print("Error! Can't detect anything")
```

---
<a id='step6'></a>
## Step 6: Test Your Algorithm

In this section, you will take your new algorithm for a spin!  What kind of dog does the algorithm think that _you_ look like?  If you have a dog, does it predict your dog's breed accurately?  If you have a cat, does it mistakenly think that your cat is a dog?

### (IMPLEMENTATION) Test Your Algorithm on Sample Images!

Test your algorithm at least six images on your computer.  Feel free to use any images you like.  Use at least two human and two dog images.  

__Question 6:__ Is the output better than you expected :) ?  Or worse :( ?  Provide at least three possible points of improvement for your algorithm.

__Answer:__ (Three possible points for improvement)


```python
## TODO: Execute your algorithm from Step 6 on
## at least 6 images on your computer.
## Feel free to use as many code cells as needed.
human_files = ['./images/human1.jpg', './images/human2.jpg', './images/human3.jpg' ]
dog_files = ['./images/dog1.jpg', './images/dog2.jpg', './images/dog3.jpg']

## suggested code, below
for file in np.hstack((human_files[:3], dog_files[:3])):
    run_app(file)
```


![png](images/output_55_0.png)


    A picture of a human, He looks like  a German shorthaired pointer



![png](images/output_55_2.png)


    A picture of a human, He looks like  a Pharaoh hound



![png](images/output_55_4.png)


    A picture of a human, He looks like  a Petit basset griffon vendeen



![png](images/output_55_6.png)


    A picture of a dog, It looks like a American staffordshire terrier



![png](images/output_55_8.png)


    A picture of a dog, It looks like a Labrador retriever



![png](images/output_55_10.png)


    A picture of a dog, It looks like a American eskimo dog

