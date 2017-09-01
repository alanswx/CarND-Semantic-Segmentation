# Semantic Segmentation
### Introduction

This project takes a pretrained VGG network, and converts it into a Fully Convolutional network. The last layer outputs a pixel array that contains the openspace.  It can be overlayed to make nice predictions.

The training data includes an image, plus the output which is a mask of where the open space is:


![roadmask](https://github.com/alanswx/CarND-Semantic-Segmentation/blob/master/writeupimages/uu_road_000097.png)


This is the corrsponding raw image:

![road](https://github.com/alanswx/CarND-Semantic-Segmentation/blob/master/writeupimages/uu_000097.png)

The code simplifies the mask, and just takes the one that corresponds to the road, it ignores the other classes.

We will feed the network images of a street scene, and it will detect the mask.

The code augments the images. It reflects the images, as well as adjusting the brightness. I was hoping the brightness would help fix some problems with shadows on the road.

Here is what the training results look like:
```
 Epoch: 0 | loss: 0.6384100317955017
 Epoch: 1 | loss: 0.5455575585365295
 Epoch: 2 | loss: 0.5127983689308167
 Epoch: 3 | loss: 0.4226115345954895
 Epoch: 4 | loss: 0.37884780764579773
 Epoch: 5 | loss: 0.25250378251075745
 Epoch: 6 | loss: 0.3391118049621582
 Epoch: 7 | loss: 0.4266332983970642
 Epoch: 8 | loss: 0.39331939816474915
 Epoch: 9 | loss: 0.19408324360847473
 Epoch: 10 | loss: 0.22098860144615173
 Epoch: 11 | loss: 0.17417633533477783
Training Finished. Saving test images to: ./runs/1504225588.823267
Model saved in file: model.ckpt
```

### Results

It seems to perform reasonable well.

here is an image in the result set that looks really good:

![good example](https://github.com/alanswx/CarND-Semantic-Segmentation/blob/master/runs/1504225588.823267/um_000000.png)

here is one that it didn't do so well on:

![bad example](https://github.com/alanswx/CarND-Semantic-Segmentation/blob/master/runs/1504225588.823267/uu_000097.png)

### predict.py

At the end of main.py I save a checkpoint of the model so that I can use it in predict.py to process the videos from the advanced lanefinding project.  The harder video works mediocre.  It didn't do well at all on the original highway video. I think it is because the color of the roads in the training dataset is very different.  US highways look a lot different than the German ones in the training set.


### Conclusions, future work

I would like to train it on the cityscape's dataset. Unfortunately there was a bit more of a barrier to download it (since I primarily have a gmail address - and their form says no gmail). I will get around that eventually but not today.  I also think there are probably some more image augmentations that could clean things up.  

It would be interesting to try some different designs other than VGG for the fully convolutional network as well.

I was surprised at how straightforward this project was.  Most of my time was spent learning tensorflow API.
