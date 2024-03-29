# AdaDeepStream
This is the official code for the paper [**AdaDeepStream: Streaming Adaptation to Concept Evolution in Deep Neural Networks**]([https://www.sciencedirect.com/science/article/pii/S0167865522000186?casa_token=5AfAKScbQmoAAAAA:BldzZgyiLI_LgYHUWQi7jyJEZ4c1oM94X5p0JmifG3dnkDEuhPFTMS50ahdQTmy4HCarhJwLFw](https://link.springer.com/article/10.1007/s10489-023-04812-0))

![SysDescrip_tex](https://github.com/chambai/AdaDeepStream/assets/61065458/48e4ba4f-a4b4-4d0d-8deb-955408022109)

AdaDeepStream is a dynamic concept evolution detection and Deep Neural Network (DNN) adaptation system using minimal true-labelled samples. Activations from within the DNN are used to enable fast streaming machine learning techniques to detect concept evolution and assist in DNN adaptation.  It can be applied to existing DNNs without the need to re-train.  It brings together the two fields of 'Online Class Incremental' and 'Concept Evolution Detection and DNN Adaptation' for high dimensional data.  In this implementation, the high dimensional data is images and the DNN is a Convolutional Neural Network.

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
Execution starts from exec_experiment.py by specifying start parameters. 
For example the following line will run the experiment for the VGG16 DNN, Fashion-MNIST dataset, trained classes of 0, 1, 2, 3, 5, 6, 8, 9 and applying classes 4 and 7 as concept evolution. A temporal-abrupt drift pattern is applied with DS-CBIR reduction method and DSAdapt adaptation method:
```
exec.run(dnn_name='vgg16',
             dataset_name='mnistfashion',
             known_classes='0-1-2-3-5-6-8-9',
             unknown_classes='4-7',
             drift_pattern='categorical-reoccurring',
             deepstream_model_name='adadeepstream',
             reduction='dscbir',  # dsdivergelast, dscbir
             adaptation='dsadapt',  
             data_discrepancy='CE',  
             )
```
exec_experiment.py also includes explanation and examples of how to run the comparison techniques.

## Citation
If you use this paper/code in your research, please cite:
```
@article{chambers2023adadeepstream,
  title={AdaDeepStream: streaming adaptation to concept evolution in deep neural networks},
  author={Chambers, Lorraine and Gaber, Mohamed Medhat and Ghomeshi, Hossein},
  journal={Applied Intelligence},
  volume={53},
  number={22},
  pages={27323--27343},
  year={2023},
  publisher={Springer}
}
```
