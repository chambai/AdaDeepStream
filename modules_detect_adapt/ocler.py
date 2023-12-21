import util
from external.ocl.general_main_dsd import OclMgr
from modules_detect_adapt.ocl_base import AnalysisOcl

analysis = None
adapt_times = []

def setData(inDsData):
    pass

def setup(indnn, act_train_batch, y_train_batch, inData=[]):
    global analysis
    analysis = AnalysisOcl()
    analysis.sequential_adapt_override = True
    analysis.setup_activation_classifiers(inData, act_train_batch, y_train_batch)
    analysis.setup_drift_detectors()
    analysis.adaptive_dnn = indnn


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

    return majority_result, y_classifier_batch, adapt_discrepancy, adapt_class

def stop():
    global analysis, adapt_times
    adapt_times = analysis.adapt_times
    analysis.stopProcessing()

def getAnalysisParams():
    return analysis.getAnalysisParams()
