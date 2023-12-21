import os
import util
import numpy as np
from framework.dnn_models import DNNPytorch
from external.ocl.general_main_dsd import OclMgr

# helper module to create and train DNNs for use in DeepStream
def loadDnnModel(trainDnn, datasetName, dataClasses, x_train, y_train, x_test, y_test):
    dnnModelPath = getFullDnnModelPathAndName()
    util.thisLogger.logInfo("Loading DNN model %s..."%(dnnModelPath))

    stats = ''
    model = get_dnn_model_object()
    if os.path.isfile(dnnModelPath):
        model.load(dnnModelPath)
        if trainDnn:
            util.thisLogger.logInfo("%s DNN model exists but TrainDnn=True. Re-training model..." % (dnnModelPath))
            stats = model.train(datasetName, dataClasses, x_train, y_train, x_test, y_test)
    else:
        util.thisLogger.logInfo("%s DNN model could not be found, training a new one" % (dnnModelPath))
        stats = model.train(datasetName, dataClasses, x_train, y_train, x_test, y_test)

    model.print_summary()
    util.thisLogger.logInfo("%s DNN model trained %s"%(dnnModelPath,stats))

    if 'ocl' in util.getParameter('AnalysisModule'):
        model = load_ocl_dnn(model, x_train, y_train)

    return model

def load_ocl_dnn(dnnModel, x_train, y_train):
    pt_dnn = DNNPytorch()
    lr, wd = pt_dnn.get_optimizer_params(util.getParameter('Dnn'), is_ocl=False)

    ocl_name = util.getParameter('AnalysisModule').split('ocl')[1].upper()
    ocl_mgr = OclMgr(dnnModel.model, ocl_name, util.getParameter('DatasetName'), util.getParameter('Dnn'),
                     util.getParameter('DataClasses'), 0.6, lr, wd)
    ocl_mgr.fit(x_train, y_train)

    return ocl_mgr

def getFullDnnModelPathAndName():
    return os.path.join(getDnnModelPath(), getDnnModelName())

def getDnnModelPath():
    return "input/models"

def getDnnModelName():
    return util.getExperimentName() + '.plt'

def get_dnn_model_object():
    model = DNNPytorch()
    return model

def loadDnn(datasetName, dataClasses, x_train, y_train, x_test, y_test):
    # util.thisLogger.logInfo('Loading DNN Model...')
    dnnModel = loadDnnModel(False, datasetName, dataClasses, x_train, y_train, x_test, y_test)
    return dnnModel
        