# Let there be Color Tensorflow

Tensorflow mplementation of [Let there be Color](http://hi.cs.waseda.ac.jp/~iizuka/projects/colorization/en/) paper published in 2016.  
Some of the settings are smaller than the one described in the paper due to lack of computing power.  

The model is still being trained  

## What's different
* no classification network  
* input size is reduced to 63x63 (larger input gives OOM error)  
* conv2d_transpose used in colorization network

## Dataset
* [Places Dataset](http://places2.csail.mit.edu/download.html)  
Validation set is used as a training data (Training data's too large to deal with)  

### Folder setting
```
-data
  -training
    -img1.jpg
    -img2.jpg
    -...  
  -validation
    -val1.jpg
    -val2.jpg
    -...
```

## Requirements
* python 2.7
* Tensorflow 1.3
* cv2

## Network Model
![Alt text](images/network.jpg?raw=true "network")

## Training
```
$ python train.py 
```

To continue training  
```
$ python train.py --continue_train=True
```

## Testing 
```
$ python test.py --test_img=test1.jpg
```


## Training Results
![Alt text](images/training.jpg?raw=true "results")

## Target
![Alt text](images/animated.gif?style=centerme "animation")

## Referenced
* pix2pix
