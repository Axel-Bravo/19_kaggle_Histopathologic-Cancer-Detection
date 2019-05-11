# Experimental diary
Resume of intuitions, ideas nad source used to improve the algorithms performance


## Model 01
Basic model with not many CNN layers, neither many complex features

### Results:
| Kaggle - Private Score  | Kaggle - Public Score  |  
|---|---|
|  0.5683 |  0.5817  |


## Model 02
The assumption made in order to improve the network is to increase the __data augmentation__ procedure. This will be
done by a usage of more training data than the provided, that means using the _data augmentator_ to generate more 
images than the original number.
 
### Results: 
Results show worst results, thus a further understanting of the __augmentator__ working way is recommended.

| Kaggle - Private Score  | Kaggle - Public Score  |  
|---|---|
|  0.5450 |  0.5582  |