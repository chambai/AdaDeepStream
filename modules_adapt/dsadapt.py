from external.ocl.general_main_dsd import OclMgr
from modules_adapt.base import Analysis

analysis = None

def setData(inDsData):
    pass

# The same as hoeffddmdnn (Hoeffding tree with ddm drift detect but detecting changes only in a window and not toggling the change into other windows drift=drift_wcn_true
def setupAnalysis(indnn, act_train_batch, y_train_batch, inData=[]):
    # hoeffding adaptive tree drift detection with dnn adaption using 2 prev dirft deteion window data and 5 surronding windows and 3 epochs and class buffer
    # for all data where drift is not detected.
    global analysis
    analysis = Analysis()
    analysis.num_window_buffers = 2
    analysis.num_previous_buffers = 5
    analysis.augment = False
    analysis.use_clustering = True
    analysis.use_anomaly_detection = False
    analysis.use_class_buffer = True
    analysis.setup_activation_classifiers(inData, act_train_batch, y_train_batch)
    analysis.setup_drift_detectors()
    analysis.setup_adaptive_dnn(['always_update', 'lin'], 'none')



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
    pass

def stopProcessing():
    global analysis
    analysis.stopProcessing()

def getAnalysisParams():
    return analysis.getAnalysisParams()
