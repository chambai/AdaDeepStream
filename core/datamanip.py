# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 11:24:26 2019

@author: id127392
"""
import numpy as np
import util
from sklearn import preprocessing

#----------------------------------------------------------------------------
def normalizeFlatValues(flatActivations, isTrainingData, maxValue=None):
    # divides each element by the max value of the training data
    if isTrainingData:
        # find max value from the training data, then normalise
        maxValue = np.amax(flatActivations)
        util.thisLogger.logDebug('Max Value: %s'%(maxValue))
    
    dataNormalization = 'norm'
    if dataNormalization == 'norm':
        flatActivations = flatActivations/maxValue
        util.thisLogger.logInfo("Normalization (%s) applied"%(dataNormalization))
    elif dataNormalization == 'std':
        flatActivations = preprocessing.scale(flatActivations)
        util.thisLogger.logInfo("Standardization (%s) applied"%(dataNormalization))
    elif dataNormalization == 'none':
        util.thisLogger.logInfo("No data normalisation/standardization")
    else:
        util.thisLogger.logInfo("unhandled data normalization of %s"%(dataNormalization))

    return flatActivations, maxValue
