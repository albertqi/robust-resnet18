# Generalizing ResNet to Various Transformations
## Albert Qi
### Spring 2024

As machine learning has grown, image classification has become more and more complex. Models have increased in size, and datasets have gotten larger. However, there are many situations in which images themselves may be distorted. Maybe an object is perceived from a different perspective, or perhaps different lightings throughout the day cause shifts in saturation and hue. Models such as ResNet-18 are not very robust to these transformations. Thus, I tackle two fundamental questions: To which transformations, if any, is ResNet-18 robust, and can ResNet-18 be improved to generalize to various transformations?

To answer these questions, I first test ResNet-18 on the CIFAR-10 test set, resulting in an accuracy of 0.81. Afterward, I rerun the tests under a diverse set of transformations, showing that accuracy drops significantly. I then develop a more robust and generalizable model by retraining ResNet-18 on the CIFAR-10 dataset with random transformations applied at runtime. This results in an increase in accuracy for all transformations except the identity, indicating that training under a distorted dataset is highly beneficial. Finally, I test if robustness to a specific transformation is generalizable to others as well and show that, regardless of the transformation, robustness tends to generalize very well to photometric transformations.

## File Information

- `common.py` - This file contains common constants, including the data directory, batch size, and list of transformations.
- `visualize.py` - This file visualizes the list of transformations. It takes in a command line argument that specifies whether to visualize all of the transformations or visualize a single transformation.
- `pretrain.py` - This file develops a pretrained ResNet-18 model.
- `val.py` - This file evaluates a model under each transformation in the list of transformations. It takes in a command line argument that specifies the name of the model to evaluate.
- `train_robust.py` - This file develops a robust ResNet-18 model by training with random transformations applied at runtime.
- `train_specific.py` - This file develops a ResNet-18 model that is trained under a specific transformation for each transformation in the list of transformations.
