# Semantic Segmentation

### Update for Second submission:

  * Added weight initialization to each new layer:

      ```
      layer7_onexone=tf.layers.conv2d(vgg_layer7_out, num_classes, 1, strides=(1,1), 
           kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
      ```

  * retrained, and included new loss values
  * updated with a new "run" and video output 


The new model seems to work much better. It fixed the bad image I had in the last submission and the green "carpet" is filled in as a constant value instead of missing a lot of pixels.  The videos also perform much better, and the output are now included.  The videos are created in predict.py

before:

![bad example](https://github.com/alanswx/CarND-Semantic-Segmentation/blob/master/writeupimages/broken_uu_000097.png)

after:

![fixed example](https://github.com/alanswx/CarND-Semantic-Segmentation/blob/master/runs/1504290592.1697896/uu_000097.png)


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

### New Training: (with layers being initialized)

not sure why the loss is jumping around.

```
 Epoch: 0 | loss: 0.3576163649559021
 Epoch: 1 | loss: 0.23299910128116608
 Epoch: 2 | loss: 0.6348710656166077
 Epoch: 3 | loss: 0.2156365066766739
 Epoch: 4 | loss: 0.08985596150159836
 Epoch: 5 | loss: 0.11673294007778168
 Epoch: 6 | loss: 0.172101229429245
 Epoch: 7 | loss: 0.15360024571418762
 Epoch: 8 | loss: 0.055733900517225266
 Epoch: 9 | loss: 0.28553634881973267
 Epoch: 10 | loss: 0.17168495059013367
 Epoch: 11 | loss: 0.12280642241239548
Training Finished. Saving test images to: ./runs/1504290592.1697896
```

### Results

It seems to perform reasonable well.

here is an image in the result set that looks really good:

![good example](https://github.com/alanswx/CarND-Semantic-Segmentation/blob/master/runs/1504290592.1697896/um_000000.png)

here is one that it didn't do so well on:

![bad example](https://github.com/alanswx/CarND-Semantic-Segmentation/blob/master/runs/1504290592.1697896/um_000093.png)

### predict.py

At the end of main.py I save a checkpoint of the model so that I can use it in predict.py to process the videos from the advanced lanefinding project.  The harder video works mediocre.  It didn't do well at all on the original highway video. I think it is because the color of the roads in the training dataset is very different.  US highways look a lot different than the German ones in the training set.


### Conclusions, future work

I would like to train it on the cityscape's dataset. Unfortunately there was a bit more of a barrier to download it (since I primarily have a gmail address - and their form says no gmail). I will get around that eventually but not today.  I also think there are probably some more image augmentations that could clean things up.  

It would be interesting to try some different designs other than VGG for the fully convolutional network as well.

I was surprised at how straightforward this project was.  Most of my time was spent learning tensorflow API.
