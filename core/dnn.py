import os
import util
import numpy as np
from core import dataset
from core.pytorch_wrapper import DNNPytorch

def loadDnnModel(trainDnn, datasetName, dataClasses, x_train, y_train, x_test, y_test, dataMapAsString):
    dnnModelPath = getFullDnnModelPathAndName()
    util.thisLogger.logInfo("Loading DNN model %s..."%(dnnModelPath))

    stats = ''
    model = get_dnn_model_object()
    if os.path.isfile(dnnModelPath):
        model.load(dnnModelPath)
        if trainDnn:
            util.thisLogger.logInfo("%s DNN model exists but TrainDnn=True. Re-training model..." % (dnnModelPath))
            stats = model.train(datasetName, dataClasses, x_train, y_train, x_test, y_test, dataMapAsString)
    else:
        util.thisLogger.logInfo("%s DNN model could not be found, training a new one" % (dnnModelPath))
        stats = model.train(datasetName, dataClasses, x_train, y_train, x_test, y_test, dataMapAsString)

    model.print_summary()
    util.thisLogger.logInfo("%s DNN model trained %s"%(dnnModelPath,stats))

    return model

def getFullDnnModelPathAndName():
    return os.path.join(getDnnModelPath(), getDnnModelName())

def getDnnModelPath():
    return "input/models"

def getDnnModelName():
    dnn = util.getParameter('Dnn')
    datasetName = util.getParameter('DatasetName')
    classes = np.unique(util.getParameter('DataClasses'))
    num_classes = len(classes)
    classesName = ''.join(map(str, classes))
    dataMapString = dataset.getDataMapAsString()
    am = util.getParameter('AnalysisModule')
    model_name = '%s_%s_%s.plt' % (dnn, datasetName, classesName)
    return model_name

def get_dnn_model_object():
    model = DNNPytorch()
    return model

def loadDnn(datasetName, dataClasses, x_train, y_train, x_test, y_test, dataMapAsString, dataDir, dataName):
    util.thisLogger.logInfo('Loading DNN Model...')
    dnnModelPath = getFullDnnModelPathAndName()
    dnnModel = loadDnnModel(False, datasetName, dataClasses, x_train, y_train, x_test, y_test, dataMapAsString)
    dnnCreationTime = util.getFileCreationTime(dnnModelPath)
    util.thisLogger.logInfo('DNN creation time=%s' % (dnnModelPath))
    return dnnModel
        