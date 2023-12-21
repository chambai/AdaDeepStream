from external.ocl.general_main_dsd import OclMgr
from modules_detect_adapt.ads_base import Analysis

analysis = None
adapt_times = []

def setData(inDsData):
    pass

# The same as hoeffddmdnn (Hoeffding tree with ddm drift detect but detecting changes only in a window and not toggling the change into other windows drift=drift_wcn_true
def setup(indnn, act_train_batch, y_train_batch, inData=[]):
    # hoeffding adaptive tree drift detection with dnn adaption using 2 prev dirft deteion window data and 5 surronding windows and 3 epochs and class buffer
    # for all data where drift is not detected.
    global analysis, adapt_times
    analysis = Analysis()
    analysis.num_window_buffers = 2
    analysis.num_previous_buffers = 5
    analysis.use_clustering = True
    analysis.use_class_buffer = True
    analysis.setup_activation_classifiers(inData, act_train_batch, y_train_batch)
    analysis.setup_drift_detectors()
    analysis.setup_adaptive_dnn()



def logTrainAccuracy(act_train, y_train):
    global analysis
    analysis.logTrainAccuracy(act_train, y_train)

def logTestAccuracy(act_unseen, y_unseen):
    global analysis
    analysis.logTestAccuracy(act_unseen, y_unseen)

def detect(xdata_unseen_batch, act_unseen_batch, dnnPredict_batch):
    global analysis
    return analysis.processUnseenStreamBatch(xdata_unseen_batch, act_unseen_batch, dnnPredict_batch)

def adapt(xdata_unseen_batch, act_unseen_batch, dnnPredict_batch, drift_result, sc_predicts, true_classes):
    # global dnn_block, adapt_ens, adaptive_dnn, is_adapted
    global analysis

    win_drift_val = 0
    if 'D' in drift_result:
        win_drift_val = 1

    # adapt
    adaptive_dnn, majority_result, y_classifier_batch, adapt_discrepancy, adapt_class = \
        analysis.adaptation(win_drift_val, act_unseen_batch, xdata_unseen_batch, dnnPredict_batch, sc_predicts,
                             drift_result, true_classes)
    # is_adapted = analysis.is_adapted

    return majority_result, y_classifier_batch, adapt_discrepancy, adapt_class

def processStreamInstances():
    pass

def stopProcessing():
    global analysis, adapt_times
    adapt_times = analysis.adapt_times
    analysis.stopProcessing()

def getAnalysisParams():
    return analysis.getAnalysisParams()
