# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 12:05:39 2019

@author: id127392
"""
import util
import numpy as np

# helper module for modules_detect_adapt scripts
moduleName = None
module = None
results = None
instanceProcessingStartTime = None
unadapted_window_processing_time = np.array([])
adapted_window_processing_time = np.array([])

#----------------------------------------------------------------------------
def loadModule(modName, dsData=None):
    global moduleName, module, results
    results = None
    moduleName = modName
    #module = __import__("das.analyse_" + moduleName, fromlist=[''])
    module = __import__(moduleName, fromlist=[''])
    if hasattr(module, 'setData'):
        module.setData(dsData)

#----------------------------------------------------------------------------
def setup(dnn, flatActivations, y_train, in_x_data_train):
    global module, moduleName
    util.thisLogger.logInfo("setup module %s..."%(moduleName))
    module.setup(dnn, flatActivations, y_train, in_x_data_train)

#----------------------------------------------------------------------------
def setupCompare(dnn, x_train, y_train):
    global module, moduleName
    util.thisLogger.logInfo("setup module %s..."%(moduleName))
    module.setup(dnn, x_train, y_train)


#----------------------------------------------------------------------------
def getAnalysisParams():
    # return a list of parameters - optional
    global module
    params = module.getAnalysisParams()
    return params

# ----------------------------------------------------------------------------
def getPredictions(dnnModel, instances, classes=[]):
    predictions = dnnModel.predict(instances)

    # The DNN is trained to output 0 or 1 only.
    if not util.has_sub_classes():
        # get the original classes it was trained on and transform the outputs
        if len(classes) == 0:
            classes = util.getParameter('DataClasses')
        if 'ocl' not in util.getParameter('AnalysisModule'):
            predictions = util.transformZeroIndexDataIntoClasses(predictions, classes)

    return predictions






