# AdaDeepStream
## Requirements and Installation
* Python 3.9 or higher
* Install libraries via:
```
pip install -r requirements.txt
```
Libraries specified assume there is a GPU.
torch, torchvision CUDA version may need to be adjusted.
torch torchvision and opencv-python may need to be installed directly with pip command instead of via requirements.txt.
Unzip external/ocl.zip and external/rsb.zip in the external directory.
OCL code is adjusted from https://doi.org/10.1016/j.neucom.2021.10.021:
```
Z. Mai, R. Li, J. Jeong, D. Quispe, H. Kim, S. Sanner, Online continual
learning in image classification: An empirical survey. Neurocomputing
469, 28–51 (2022). https://doi.org/10.1016/j.neucom.2021.10.021
```
RSB code is adjusted from https://doi.org/10.1109/CVPRW53098.2021.00404:
```
L. Korycki, B. Krawczyk, Class-Incremental Experience Replay for Continual Learning under Concept Drift, in 2021 IEEE/CVF Conference on
Computer Vision and Pattern Recognition Workshops (CVPRW) (IEEE,
Nashville, TN, USA, 2021), pp. 3644–3653. https://doi.org/10.1109/CVPRW53098.2021.00404
```

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
