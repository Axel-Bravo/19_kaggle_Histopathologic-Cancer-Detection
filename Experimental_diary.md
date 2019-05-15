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


## Model 05
__Less Regularization__: reduction in the presence of regularized layers, _specifically_ the convolutional one's,
as well as, the reduction of it's penalization; being this case the default value of __0.01__.

### Results:
The results have improved, though applying __batch normalization__ and __regularization__ (on the dense layers), 
__does not__ improve form the _model 03_, which had no _regularization_ neither _batch normalization_. Thus deducing
from it, that both __batch normalization__ and __regularization__ are useful but _no silver bullet_.

| Kaggle - Private Score  | Kaggle - Public Score  |  
|---|---|
|  0.6606 | 0.7486  |


## Model 06
__Bigger input images__: maybe the previous reduction of images ought to be something that is reducing the accuracy of 
the resulting model. The dimention will be increased from (32, 32) to (50, 50). We expect an imprvement of the results
on Kaggle.

### Results:
Augmenting the size of the input has __not resulted__ into better scores. 

| Kaggle - Private Score  | Kaggle - Public Score  |  
|---|---|
|  0.6441 | 0.6825  |


## Model 07
__More CNN filters__: based in the _Model 03_ architecture, we will double the number of __CNNs filters__ from
32 to 64

### Results:
Augmenting the size of the number of the CNN filters  has __not resulted__ into better scores. Maybe we need to,
employe a deeper stack!

| Kaggle - Private Score  | Kaggle - Public Score  |  
|---|---|
|  0.5950 | 0.6595  |


## Model 08
__Deeper CNN filters__: based in the _Model 03_ architecture, we will increase the depth of __CNNs filters__ 

### Results:
The augmentation of the depth in the NN has __not resulted__ in better scores.

| Kaggle - Private Score  | Kaggle - Public Score  |  
|---|---|
|  0.6435 | 0.7059  |


## Model 09
__No image augmentation__: maybe the image augmentation is not helping the network to better predict test images,
it is not necessary; hence more a burden than an improvement

### Results: 
 The  cancelation of many image augmentation procedures has caused a __performance increase__, being this one the best
 score for the moment. The techniques that have been removed are: 
   - Shear_range=10
   - Horizontal_flip=True
   - Vertical_flip=True

This may be due to the fact that, we are training the network to detect cases that are not in the _test set_, meaning, 
that the training dataset __is already biug enough__ for the prediction task required.  

| Kaggle - Private Score  | Kaggle - Public Score  |  
|---|---|
|  0.7413 | 0.7413  |


## Model 011, 012, 013
__SeparableConv2D__: Use Separable Convolution as type in order to improve the accuracy

### Results: 
| Model  | Kaggle - Private Score  | Kaggle - Public Score  |  
|---|---|---|
|  11 | 0.8036 | 0.8531  |
|  12 | 0.7833 | 0.8205  |
|  13 | 0.7842 | 0.8507  |


## Model 014
__Transfer Learning__: Use a pre-trained NN in order to profit from it's CNN part


### Results: 

| Model  | Kaggle - Private Score  | Kaggle - Public Score  |  
|---|---|---|
|  14 | 0.7415 | 0.8091  |


## Future Models:
 - Crop center of image, i.e., (32, 32), now it's reduction
 - add dropout in the CNN layers