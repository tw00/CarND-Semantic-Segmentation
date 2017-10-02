# Semantic Segmentation

### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).


### Model

The model is based on an architecture from [link](https://people.eecs.berkeley.edu/%7Ejonlong/long_shelhamer_fcn.pdf).
I am using a pre-trained VGG-16 network and replaced the final layer with a fully connected layer:

```python
layer7_conv_out = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding = 'same' )
layer4_conv_decode1 = tf.layers.conv2d_transpose(layer7_conv_out, num_classes, 4, strides = (2, 2), padding = 'same' )
layer4_conv_decode2 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same' )
output_1 = tf.add(layer4_conv_decode1, layer4_conv_decode2)
layer3_conv_decode1 = tf.layers.conv2d_transpose(output_1, num_classes, 4, strides = (2, 2), padding='same' )
layer3_conv_decode2 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same' )
output_2 = tf.add(layer3_conv_decode1, layer3_conv_decode2)
output_3 = tf.layers.conv2d_transpose(output_2, num_classes, 16, strides=(8, 8), padding='same' )
```

### Parameters

Parameters used:

    Loss function:  Cross entropy
    Optimizer:      Adam optimizer
    Learning rate:  0.001
    Keep prob.:     0.4
    Epochs:         50
    Batch size:     12

### Ouput

The trained model gives the following output:

![Semantic Segmentation][img/result.png]


### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
 ## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).
