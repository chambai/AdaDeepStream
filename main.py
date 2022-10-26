import util
from core import dataset, train, analyse, datastream, pretrain_dnn
import numpy as np
from core.dnn_models import DNNPytorch
from collections import defaultdict

def start(dnn_name='vgg16', dataset_name='mnistfashion', data_combination='23456789-01', drift_pattern='categorical-abrupt', reduction='dscbir', adaptation='dsadapt'):
    if __name__ == '__main__':
        trainedClasses = data_combination.split('-')[0]
        unknownClasses = data_combination.split('-')[1]

        # load setup file
        util.params = None
        util.usedParams = []
        util.setupFile = 'input/params/%s_%s_%s_%s.txt'%(dnn_name,dataset_name,trainedClasses,unknownClasses)
        util.setParameter('Dnn', dnn_name)
        util.setParameter('DatasetName', dataset_name)
        util.setParameter('DriftType', drift_pattern.split('-')[0])
        util.setParameter('DriftPattern', drift_pattern.split('-')[1])
        util.setParameter('DataClasses', '[%s]'%(','.join(list(trainedClasses))))
        if adaptation == 'rsb':
            util.setParameter('AnalysisModule', 'modules_compare.%s' % (adaptation))
            util.setParameter('LayerActivationReduction', 'none')
        else:
            util.setParameter('AnalysisModule', 'modules_analysis.%s' % (adaptation))
            if reduction == 'jsdl':
                # JSDL requires that the data is flattened and padded prior to jsdl calculations
                reduction = 'flatten,pad,jsdl'
            util.setParameter('LayerActivationReduction', reduction)

        util.thisLogger = util.Logger()

        # Get the training class data
        x_train, y_train, x_test, y_test = dataset.getFilteredData()

        # Load the DNN
        model = pretrain_dnn.loadDnn(dataset_name, list(map(int, list(trainedClasses))), x_train, y_train, x_test, y_test)
        # model_filename = 'input/models/%s_%s_%s.plt' % (dnn_name, dataset_name, trainedClasses)
        # model = DNNPytorch()
        # if not model.load(model_filename, raise_load_exception=False):
        #     stats = model.train(dataset_name, util.getParameter('DataClasses'), x_train, y_train, x_test, y_test)
        #     print(stats)

        if adaptation == 'rsb':
            analyse.loadModule('modules_compare.' + adaptation)
            analyse.setupCompare(model, x_train, y_train)
            # pixel data is used and this is already normalized
            # when unseen data is extracted this will be normalized, so we can pass 1 in as the maxValue
            unseenData = datastream.getData()
            processDataStream(unseenData, model, 1)
        else:
            # get activations
            activations, xData, yData = train.getActivations(x_train, -1, model, y_train)

            # normalize
            maxValue = np.amax(activations)
            activations = activations / maxValue
            analyse.loadModule('modules_adapt.' + adaptation)
            analyse.setup(model, activations, yData, xData)
            unseenData = datastream.getData()
            processDataStream(unseenData, model, maxValue)

# ------------------------------------------------------------------------------------------
def processDataStream(all_unseen_data, model, maxValue):
    unseenDataDict = defaultdict(list)
    for item in all_unseen_data:
        unseenDataDict[item.adaptState].append(item)

    unseenInstancesList = []
    for adapt_state, unseenData in unseenDataDict.items():
        unseenInstancesObjs = analyse.startDataInputStream(model, maxValue,
                                                           unseenData)
        unseenInstancesList.append(unseenInstancesObjs)
        if unseenInstancesObjs[0].adaptClass != '-':
            # calculate accuracy of each section
            correct = [x.correctResult for x in unseenInstancesObjs if x.correctResult == int(x.adaptClass)]
            acc = len(correct) / len(unseenInstancesObjs)
            util.thisLogger.logInfo('%sAcc=%s' % (adapt_state, acc))

    unseenInstancesObjList = [item for sublist in unseenInstancesList for item in sublist]
    util.thisLogger.logInfo('TotalNumberOfUnseenInstances=%s' % (len(unseenInstancesObjList)))
    drift_instances = [u.id for u in unseenInstancesObjList if u.driftDetected]
    util.thisLogger.logInfo('DriftDetectionInstances=%s' % (drift_instances))

    analyse.stopProcessing()


# valid parameter values
# dnn_name: vgg16
# dataset_name: mnistfashion, cifar10
# data_combination: 23456789-01,  01235689-47 # choose any combination of class numbers 0 to 9 and create txt file in input/params folder
# drift_pattern: categorical-abrupt, temporal-abrupt, categorical-gradual, temporal-gradual, categorical-incremental, categorical-reoccurring, temporal-reoccurring
# reduction: dscbir, jsdl
# adaptation: dsadapt, ocllwf, ocler, oclicarl, oclmirrv, noadapt, rsb

# Examples - run one at a time
# mobilenet fashion class data example (our dsadapt method)

# AdaDeepStream method (Ours)
start(dnn_name='vgg16',
      dataset_name='mnistfashion',
      data_combination='01235689-47',
      drift_pattern='temporal-abrupt',
      reduction='dscbir', # dscbir, jsdl
      adaptation='dsadapt') # dsadapt, noadapt

# OCL adaptation methods
# start(dnn_name='vgg16',
#       dataset_name='mnistfashion',
#       data_combination='01235689-47',
#       drift_pattern='temporal-abrupt',
#       reduction='dscbir',  # dscbir, jsdl
#       adaptation='oclicarl') # ocllwf, ocler, oclicarl, oclmirrv

# RSB comparison method
# start(dnn_name='vgg16',
#       dataset_name='mnistfashion',
#       data_combination='01235689-47',
#       drift_pattern='temporal-abrupt',
#       reduction='',
#       adaptation='rsb')




