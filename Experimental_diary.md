# Experimental diary
Resume of intuitions, ideas nad source used to improve the algorithms performance


## Model 01
Basic model with not many CNN layers, neither many complex features

### Results:
| Kaggle - Private Score  | Kaggle - Public Score  |  
|---|---|
|  0.5683 |  0.5817  |


## Model 02
__More Data__: the assumption made in order to improve the network is to increase the __data augmentation__ procedure.
This will be done by a usage of more training data than the provided, that means using the _data augmentator_
to generate more images than the original number.
 
### Results: 
Results show worst results, thus a further understanting of the __augmentator__ working way is recommended.

| Kaggle - Private Score  | Kaggle - Public Score  |  
|---|---|
|  0.5450 |  0.5582  |


## Model 03
__Deeper CNN__: the assumption made in order to improve the network is to increase the __depth of the convolution__ 
procedure, in order to extract more deep features from the images, before going into the __dense__ layers. 
This has been achieved by including a __second convolutional__ step before doing the _max pooling_ operation.
 
### Results: 
Results show a __strong increase__ from the _initial model_ (model 01), making obvious the necessity of a complexer,
convolutional structure, to __extract meaning from the images__, before going into the dense part. 

| Kaggle - Private Score  | Kaggle - Public Score  |  
|---|---|
|  0.6505 |  0.7593  |


## Model 04
__Regularization__: the assumption is that the NN may be _overfitting_, in this case adding, regularization of
type II in the dense and convolutional parts

### Results:
The regularization value employed is __extremely high__, comparing with the recommended _l2 value_ = 0.01; a value
of 0.025 has been employed on all layers.

| Kaggle - Private Score  | Kaggle - Public Score  |  
|---|---|
|  0.5037 |  0.5078  |
