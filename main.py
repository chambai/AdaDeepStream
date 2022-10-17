import util
from core import dataset, train, analyse, datastream
import numpy as np
from core.pytorch_wrapper import DNNPytorch

from collections import defaultdict

def start(dnnName, datasetName, dataCombination, drift_pattern='categorical-abrupt', reduction='dscbir', adaptation='dsadapt'):
    if __name__ == '__main__':
        trainedClasses = dataCombination.split('-')[0]
        unknownClasses = dataCombination.split('-')[1]

        # load setup file
        util.params = None
        util.usedParams = []
        util.setupFile = 'input/params/%s_%s_%s_%s.txt'%(dnnName,datasetName,trainedClasses,unknownClasses)
        util.setParameter('Dnn', dnnName)
        util.setParameter('DatasetName', datasetName)
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
        model_filename = 'input/models/%s_%s_%s.plt' % (dnnName, datasetName, trainedClasses)
        model = DNNPytorch()
        if not model.load(model_filename, raise_load_exception=False):
            stats = model.train(datasetName, util.getParameter('DataClasses'), x_train, y_train, x_test, y_test)
            print(stats)

        if adaptation == 'rsb':
            analyse.loadModule('modules_compare.' + adaptation)
            analyse.setupCompare(model, x_train, y_train)
            # pixel data is used and this is already normalized
            # when unseen data is extracted this will be normalized, so we can pass 1 in as the maxValue
            unseenData = datastream.getData()
            unseenInstancesObjList = analyse.startDataInputStream(model, False, 1, 1, unseenData)
        else:
            # get activations
            activations, xData, yData = train.getActivations(x_train, -1, model, y_train)

            # normalize
            maxValue = np.amax(activations)
            activations = activations / maxValue
            util.saveReducedData('output/trainingactivations', activations, y_train)
            analyse.loadModule('modules_adapt.' + adaptation)
            analyse.flatSetupDnn(model, activations, yData, xData) # why is y passed twice?
            unseenData = datastream.getData()
            processDataStream(unseenData, model, maxValue)

# ------------------------------------------------------------------------------------------
def processDataStream(all_unseen_data, model, maxValue):
    unseenDataDict = defaultdict(list)
    for item in all_unseen_data:
        unseenDataDict[item.adaptState].append(item)

    unseenInstancesList = []
    for adapt_state, unseenData in unseenDataDict.items():
        unseenInstancesObjs = analyse.startDataInputStream(model, False,
                                                           maxValue, maxValue,
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

    # calculate total accuracy of ND instances
    ndUnseenPredicts = [x.correctResult for x in unseenInstancesObjList if
                        x.discrepancyName == 'ND' and x.correctResult == x.predictedResult]
    ndUnseenCorrectPredicts = [x.correctResult for x in unseenInstancesObjList if x.discrepancyName == 'ND']
    acc = len(ndUnseenPredicts) / len(ndUnseenCorrectPredicts)
    util.thisLogger.logInfo('UnseenNDInstancesAcc=%s' % (acc))

    analyse.stopProcessing()

# Examples - run one at a time
# mobilenet cifar10 class data example (our method)
start(dnnName='vgg16',
      datasetName='mnistfashion',
      dataCombination='01235689-47',
      drift_pattern='categorical-abrupt',
      reduction='dscbir',
      adaptation='dsadapt')




