# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 17:26:27 2019

@author: id127392
"""
from keras.datasets import cifar10, fashion_mnist, mnist
import util
import numpy as np
import cv2
from framework import pretrain_dnn
import os
import torchvision
from torchvision import datasets, transforms
import torch
from torch.utils.data import Dataset
from framework.cifar100coarse import CIFAR100Coarse

# gets training and test data and filters it depending on specified known and unknown classes
# also maps data to sub-classes
isDataLoaded = False
x_train = None
y_train = None
y_train_umapped = None
x_test = None
y_test = None
y_test_unmapped = None

#--------------------------------------------------------------------------
def getFilteredData(isMap=True, do_normalize=True):
    x_train, y_train, x_test, y_test = getAllData(do_normalize)
    # print(x_train.shape)
    # print(x_test.shape)

    classes = util.getParameter('DataClasses')
    # util.thisLogger.logInfo('Data classes to be used: %s'%(classes))
    x_train, y_train, x_test, y_test = filterData(x_train, y_train, x_test, y_test, classes)

    if isMap:
        # map data to different labels
        x_train, y_train, y_train_unmapped = mapClasses(x_train, y_train)
        x_test, y_test, y_test_unmapped = mapClasses(x_test, y_test)

    if len(y_train.shape) > 1:
        y_train = np.asarray([x[0] for x in y_train])
        y_test = np.asarray([x[0] for x in y_test])
    return x_train, y_train, x_test, y_test


def concatenate_data(self, x, y):
        if len(self.km.x) == 1:
            self.km.x = np.zeros((0,x.shape[1]))
            self.km.y = np.zeros((0))
        all_x = np.vstack((self.km.x,x))
        all_y = np.hstack((self.km.y, y))
        return all_x, all_y

#--------------------------------------------------------------------------
def filterData(x_train, y_train, x_test, y_test, classes):
    x_train, y_train = util.filterDataByClass(x_train, y_train, classes)
    x_test, y_test = util.filterDataByClass(x_test, y_test, classes)
        
    x_train = resize(x_train)
    x_test = resize(x_test)
    return x_train, y_train, x_test, y_test

#--------------------------------------------------------------------------
def getOutOfFilterData(isMap=True, do_normalize=True):
    x_train, y_train, x_test, y_test = getAllData(do_normalize)
    classes = util.getParameter('DataDiscrepancyClass')
    x_train, y_train, x_test, y_test = filterData(x_train, y_train, x_test, y_test, classes)

    if isMap:
        # map data to different labels
        x_train, y_train, y_train_unmapped = mapClasses(x_train, y_train)
        x_test, y_test, y_test_unmapped = mapClasses(x_test, y_test)

    if len(y_train.shape) > 1:
        y_train = np.asarray([x[0] for x in y_train])
        y_test = np.asarray([x[0] for x in y_test])

    return x_train, y_train, x_test, y_test

#--------------------------------------------------------------------------
def resetData():
    global isDataLoaded
    global x_train
    global y_train
    global y_train_unmapped
    global x_test
    global y_test
    global y_test_unmapped   
    
    isDataLoaded = False
    x_train = None
    y_train = None
    x_test = None
    y_test = None
    y_test_unmapped = None

#--------------------------------------------------------------------------
def getAllData(do_normalize=True):
    global isDataLoaded
    global x_train
    global y_train
    global y_train_unmapped
    global x_test
    global y_test
    global y_test_unmapped

    datasetName = util.getParameter('DatasetName')
    if datasetName == 'cifar10':
        # get the cifar 10 dataset from Keras
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    elif datasetName == 'cifar100':
        if util.getParameter('DataDiscrepancy') == 'CD':
            dataset_train = CIFAR100Coarse(root='input/data', train=True, transform=None, target_transform=None, download=True)
            inv_classes = {v: k for k, v in dataset_train.class_to_idx.items()}
            util.thisLogger.logInfo('Cifar100 trained fine class names: %s'%([inv_classes[t] for t in util.getParameter('DataClasses')]))
            util.thisLogger.logInfo('Cifar100 novel fine class names: %s' % ([inv_classes[t] for t in util.getParameter('DataDiscrepancyClass')]))
            x_train = dataset_train.data
            y_train = np.array(dataset_train.targets_fine)
            dataset_test = CIFAR100Coarse(root='input/data', train=False, download=True)
            x_test = dataset_test.data
            y_test = np.array(dataset_test.targets_fine)
        elif util.getParameter('DataDiscrepancy') == 'CE':
            # Coarse labels
            dataset_train = CIFAR100Coarse(root='input/data', train=True, transform=None, target_transform=None,
                                           download=True)
            inv_classes = {v: k for k, v in dataset_train.class_to_idx.items()}
            util.thisLogger.logInfo(
                'Cifar100 trained fine class names: %s' % ([inv_classes[t] for t in util.getParameter('DataClasses')]))
            util.thisLogger.logInfo('Cifar100 novel fine class names: %s' % (
            [inv_classes[t] for t in util.getParameter('DataDiscrepancyClass')]))
            x_train = dataset_train.data
            y_train = np.array(dataset_train.targets_coarse)
            dataset_test = CIFAR100Coarse(root='input/data', train=False, download=True)
            x_test = dataset_test.data
            y_test = np.array(dataset_test.targets_coarse)
        else:
            raise Exception('unhandled data discrepancy for cifar100 (currently only handles CD)')
    elif datasetName == 'mnistfashion':
        # get the MNIST-Fashion dataset from Keras
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        y_train = np.reshape(y_train, (y_train.shape[0],1))
        y_test = np.reshape(y_test, (y_test.shape[0],1))
    elif datasetName == 'mnist':
        # get the MNIST dataset from Keras
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        y_train = np.reshape(y_train, (y_train.shape[0],1))
        y_test = np.reshape(y_test, (y_test.shape[0],1))
    elif datasetName == 'cifar10distil':
        # get the distilled cifar 10 dataset
        x_train, y_train, x_test, y_test = get_distilled_dataset(datasetName)
    else:
        raise ValueError("Unhandled dataset name of %s. Make sure DnnModelPath parameter contains the dataset name"%(datasetName))

    if 'distil' not in datasetName:
        pass

    if do_normalize:
        # Normalise
        x_test = x_test.astype('float32')/255
        x_train = x_train.astype('float32')/255

    isDataLoaded = True

    y_train = y_train.astype('int16')
    y_test = y_test.astype('int16')

    return x_train, y_train, x_test, y_test

#-------------------------------------------------------------------------
def get_distilled_dataset(dataset_name):
    # gets distilled dataset with standard test images
    if 'cifar10' in dataset_name:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = np.load(r'c:\dsd\datasets\cifar10distil_xdata_ConvNet_ssize100_nozca_l_noaug_ckpt1000.npy')
        y_train = np.load(r'c:\dsd\datasets\cifar10distil_ydata_ConvNet_ssize100_nozca_l_noaug_ckpt1000.npy')
    else:
        raise Exception('unhandled distilled dataset of %s' % (dataset_name))

    return x_train, y_train, x_test, y_test
#-------------------------------------------------------------------------
def resize(x):
    out = []
    isReshaped, shape = getDataShape()
    
    if isReshaped:
        
        x_reshaped = []
        for i in range(len(x)):
            image = x[i]
            img = cv2.resize(image, (32,32), interpolation = cv2.INTER_NEAREST)
            x_reshaped.append(np.stack((img,)*3, axis=-1))
        x = np.asarray(x_reshaped)
        out = x

    else:
        out = x
    
    return np.asarray(out)

#-------------------------------------------------------------------------
def getDataShape():
    datasetName = util.getParameter('DatasetName')
    isReshaped = False
    if datasetName == 'cifar10' or datasetName == 'cifar100':
        out =  (32,32,3)
    elif datasetName == 'mnistfashion':
        out =  (32,32,3)
        isReshaped = True
    else:
        out = (32,32,3)
    return isReshaped, out

#-------------------------------------------------------------------------
def mapClasses(x, y):
    global isMapped
    y_unmapped = y
    y_out = y
    
    # maps data to higher level classes
    mapOriginalYValues = util.get_mapping_param('MapOriginalYValues')
    
    # map data if mapOriginalYValues contains data
    if len(mapOriginalYValues) != 0:
        # util.thisLogger.logInfo('Mapping classes: length of x data: %s. Length of y data: %s. Y data values: %s'%(len(x),len(y),np.unique(y)))
        mapNewYValues = np.asarray(util.get_mapping_param('MapNewYValues'))
        mapNewNames = util.get_mapping_param('MapNewNames')
    
        # check mapOriginalYValues and mapNewYValues are the same size
        if len(mapOriginalYValues) != len(mapNewYValues):
             raise ValueError("MapOriginalYValues array size (%s) does not match MapNewYValues array size (%s)"%(len(mapOriginalYValues), len(mapNewYValues)))

        # check distinct values of mapNewYValues match number of elements in mapNewNames
        distinctMapNewYValues = np.unique(mapNewYValues[mapNewYValues >= 0])
        if len(distinctMapNewYValues) != len(mapNewNames):
             raise ValueError("Distinct values of MapNewYValues (%s) does not match the number of elements in MapNewNames (%s)"%(len(distinctMapNewYValues), len(mapNewNames)))

        # if there's any -1 values in mapNewYValues, remove X and Y values for the corresponding class in mapOriginalYValues
        if -1 in mapNewYValues:
            # find out what elements in mapOriginalYValues the -1 corresponds to
            minusOneIndexes = np.where(mapNewYValues == -1)
            yValuesToRemove = mapOriginalYValues[minusOneIndexes]
            dataIndexesToRemove = np.in1d(y, yValuesToRemove).nonzero()[0]
            y = np.delete(y, dataIndexesToRemove, axis=0)
            y_unmapped = y
            x = np.delete(x, dataIndexesToRemove, axis=0)

        y_dict = {}
        for orig, new in zip(mapOriginalYValues, mapNewYValues):
            y_dict[orig] = new

        y_out = []
        y = np.reshape(y,(-1))
        for y_val in y:
            y_out.append(y_dict[y_val])
        y_out = np.array(y_out)

        isMapped = True
        
        # util.thisLogger.logInfo('Mapped classes: length of x data: %s. Length of y data: %s. Y data values: %s'%(len(x),len(y_out),np.unique(y_out)))
    return x, y_out, y_unmapped

#-------------------------------------------------------------------------
def getDataMapAsString():
    # returns the new set of y values
    mapNewYValues = np.asarray(util.get_mapping_param('MapNewYValues'))
    distinctMapNewYValues = np.unique(mapNewYValues[mapNewYValues >= 0])
    mapAsString = ''.join(map(str,distinctMapNewYValues))
    return mapAsString
    
#-------------------------------------------------------------------------
def getDataMap(unmappedClasses):
    # returns the new set of y values
    mapNewYValues = np.asarray(util.get_mapping_param('MapNewYValues'))
    distinctMapNewYValues = np.unique(mapNewYValues[mapNewYValues >= 0])
    mapAsString = ''.join(map(str,distinctMapNewYValues))
    return mapAsString

#--------------------------------------------------------------------------
def load_distilled_data(classes):
    num_images_per_class = 50    # 1, 10, 50
    dir = os.getcwd()
    data_dir = os.path.join(dir, 'traindnn', 'data','dist_traj',util.getParameter('DatasetName'))
    x_file = os.path.join(data_dir, str(num_images_per_class), 'images_best.pt')
    y_file = os.path.join(data_dir, str(num_images_per_class), 'labels_best.pt')
    x_data = torch.load(x_file)
    x_data = np.reshape(x_data, (x_data.shape[0], 32,32,3))
    y_data = torch.load(y_file)

    # filter the data to classes that are in self.all_classes
    idxs = []
    for c in classes:
        for i, y in enumerate(y_data):
            if c == y:
                idxs.append(i)

    x_data = x_data[idxs]
    y_data = y_data[idxs]

    x_data.cpu().detach().numpy()
    y_data.cpu().detach().numpy()

    return x_data, y_data


#--------------------------------------------------------------------------
# not currently used
def get_all_data_and_store(is_training_data=False):
    dataset_name = util.getParameter('DatasetName')
    act_red = util.getParameter('LayerActivationReduction')[-1]
    data_folder_name = r'R:\data_%s_%s'%(act_red, util.getParameter('DeepLearningFramework'))
    data_file_name = os.path.join(data_folder_name, util.getParameter('DatasetName'))
    if is_training_data:
        data_file_name += '_train'
    else:
        data_file_name += '_test'

    if os.path.exists(data_file_name + '.npz'):
        data = np.load(data_file_name + '.npz')
    else:
        if dataset_name == 'cifar10':
            trainset = torchvision.datasets.CIFAR10(root='./data', train=is_training_data,
                                                    download=True)
        elif dataset_name == 'cifar100':
            trainset = CIFAR100Coarse(root='./data', train=is_training_data,
                                                    download=True)
        elif dataset_name == 'mnistfashion':
            trainset = torchvision.datasets.FashionMNIST(root='./data', train=is_training_data,
                                                    download=True)
        else:
            raise Exception('unhandled dataset name of %s'%(dataset_name))

        np.savez(data_file_name, x=trainset.data, y=trainset.targets)
        data = np.load(data_file_name + '.npz')

    x = data['x']
    y = data['y']
    x = trainset.data/255

    return x, y

def has_sub_classes():
    return len(util.get_mapping_param('MapNewYValues')) > 0

def get_super_classes():
    # get data classes
    sub_data_classes = util.getParameter('DataClasses')
    # get super classes
    superclasses = np.array(util.get_mapping_param('MapNewYValues'))
    superclasses = np.sort(np.unique(superclasses[sub_data_classes]))
    return superclasses

class experimental_dataset(Dataset):

    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data[0].shape[0])

    def __getitem__(self, idx):
        item = self.data[0][idx]
        x_item = self.transform(item)
        y_item = self.data[1][idx]
        return x_item, y_item

    transform = transforms.Compose([
        transforms.RandomRotation(90)
    ])







    
    



    