# AdaDeepStream
## Requirements
* Python 3.9 or higher
* Install libraries via:
```
pip install -r requirements.txt
```
Libraries specified assume there is a GPU.
torch, torchvision CUDA version may need to be adjusted.
torch torchvision and opencv-python may need to be installed directly with pip command instead of via requirements.txt.

## Datasets and Deep Neural Networks (DNNs)
* CIFAR-10 and Fashion-MNIST will be downloaded during first run
* DNN will be loaded and trained on first run.

## Parameter Setup and Execution
Execution starts from main.py by specifying start parameters. 
For example the following line will run the experiment for the VGG16 DNN, Fashion-MNIST dataset, trained classes of 0, 1, 2, 3, 5, 6, 8, 9 and applying classes 4 and 7 as concept evolution. A temporal-abrupt drift pattern is applied with DS-CBIR reduction method and DSAdapt adaptation method:
```
start(dnn_name='vgg16',
      dataset_name='mnistfashion',
      data_combination='01235689-47',
      drift_pattern='temporal-abrupt',
      reduction='dscbir',
      adaptation='dsadapt')
```
main.py also includes explanation and examples of how to run the comparison techniques.
