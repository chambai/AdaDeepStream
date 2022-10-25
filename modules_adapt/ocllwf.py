from external.ocl.general_main_dsd import OclMgr
from modules_adapt.base_ocl import AnalysisOcl
import util

analysis = None

def setData(inDsData):
    pass

def setupAnalysis(indnn, act_train_batch, y_train_batch, inData=[]):
    global analysis
    analysis = AnalysisOcl()
    analysis.sequential_adapt_override = True
    analysis.setup_activation_classifiers(inData, act_train_batch, y_train_batch)
    analysis.setup_drift_detectors()
    analysis.adaptive_dnn = indnn


def logTrainAccuracy(act_train, y_train):
    global analysis
    analysis.logTrainAccuracy(act_train, y_train)

def logTestAccuracy(act_unseen, y_unseen):
    global analysis
    analysis.logTestAccuracy(act_unseen, y_unseen)

def processUnseenStreamBatch(xdata_unseen_batch, act_unseen_batch, dnnPredict_batch, true_values, true_discrep):
    global analysis
    return analysis.processUnseenStreamBatch(xdata_unseen_batch, act_unseen_batch, dnnPredict_batch, true_values, true_discrep)

def processStreamInstances():
    # only used if instances need to be processed on a separate thread
    pass

def stopProcessing():
    global analysis
    analysis.stopProcessing()

def getAnalysisParams():
    return analysis.getAnalysisParams()
