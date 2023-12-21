import numpy as np
import random
import util
from framework import dataset

# gets the data stream from the test data
unseenDataList = None
instanceProcessingStartTime = None
numberNdUnseenInstances = 0
numberDUnseenInstances = 0

# ----------------------------------------------------------------------------
def getData(isApplyDrift=True):
    # Gets the unseen data from the dataset
    global numberNdUnseenInstances, numberDUnseenInstances
    ndUnseenDataList = []
    dUnseenDataList = []
    numUnseenInstances = util.getParameter('NumUnseenInstances')
    dataDiscrepancyClass = util.getParameter('DataDiscrepancyClass')
    f_x_train, f_y_train, f_x_test, f_y_test = dataset.getFilteredData()
    f_x_train_no_norm, f_y_train_no_norm, f_x_test_no_norm, f_y_test_no_norm = dataset.getFilteredData(
        do_normalize=False)
    if util.has_sub_classes():
        _, u_f_y_train, _, u_f_y_test = dataset.getFilteredData(
            isMap=False)  # if data is sub-classed, get unmapped classes
        _, u_f_y_train_no_norm, _, u_f_y_test_no_norm = dataset.getFilteredData(isMap=False, do_normalize=False)

    if numUnseenInstances == -1:
        numberNdUnseenInstances = len(f_x_test)
    oof_x_train, oof_y_train, oof_x_test, oof_y_test = dataset.getOutOfFilterData()
    oof_x_train_no_norm, oof_y_train_no_norm, oof_x_test_no_norm, oof_y_test_no_norm = dataset.getOutOfFilterData(
        do_normalize=False)
    if util.has_sub_classes():
        _, u_oof_y_train, _, u_oof_y_test = dataset.getOutOfFilterData(isMap=False)
        _, u_oof_y_train_no_norm, _, u_oof_y_test_no_norm = dataset.getOutOfFilterData(isMap=False, do_normalize=False)

    if numUnseenInstances == -1:
        numberDUnseenInstances = len(oof_x_test)
    dataDiscrepancy, numDiscrepancy, numNonDiscrepancy, unseenDataSource = getInstanceParameters()

    isRandom = False
    numUnseenInstances = util.getParameter('NumUnseenInstances')

    # get non-discrepancy data
    if numUnseenInstances != -1:
        if isRandom:
            rndIdexes = np.random.randint(f_x_test.shape[0], size=numNonDiscrepancy)
            f_x_test = f_x_test[rndIdexes, :]
            f_x_test_no_norm = f_x_test_no_norm[rndIdexes, :]
            f_y_test = f_y_test[rndIdexes, :]
            if util.has_sub_classes():
                u_f_y_test = u_f_y_test[rndIdexes, :]
        else:
            f_x_test = f_x_test[:numUnseenInstances]
            f_x_test_no_norm = f_x_test_no_norm[:numUnseenInstances]
            f_y_test = f_y_test[:numUnseenInstances]
            if util.has_sub_classes():
                u_f_y_test = u_f_y_test[:numUnseenInstances]

    # get discrepancy data
    if (dataDiscrepancy == 'none') or (dataDiscrepancy == 'AA'):
        numDiscrepancy = 0
    else:
        if numUnseenInstances != -1:
            if isRandom:
                rndIdexes = np.random.randint(oof_x_test.shape[0], size=numDiscrepancy)
                oof_x_test = oof_x_test[rndIdexes, :]
                oof_x_test_no_norm = oof_x_test_no_norm[rndIdexes, :]
                oof_y_test = oof_y_test[rndIdexes, :]
                if util.has_sub_classes():
                    u_oof_y_test = u_oof_y_test[rndIdexes, :]
            else:
                oof_x_test = oof_x_test[:numUnseenInstances]
                oof_x_test_no_norm = oof_x_test_no_norm[:numUnseenInstances]
                oof_y_test = oof_y_test[:numUnseenInstances]
                if util.has_sub_classes():
                    u_oof_y_test = u_oof_y_test[:numUnseenInstances]

    iCount = 1
    for index in range(numNonDiscrepancy):
        nonDiscrepancyInstance = f_x_test[np.array([index])]
        nonDiscrepancyInstance_no_norm = f_x_test_no_norm[np.array([index])]
        result = f_y_test[index]
        i = UnseenData(instance=nonDiscrepancyInstance, instance_no_norm=nonDiscrepancyInstance_no_norm,
                       correctResult=result, discrepancyName='ND')
        i.id = str(iCount)
        if util.has_sub_classes():
            i.hasSubClasses = True
            i.subCorrectResult = u_f_y_test[index]
        ndUnseenDataList.append(i)
        iCount += 1
    # collect discrepancy data and insert randomly
    for index in range(numDiscrepancy):
        discrepancyInstance = oof_x_test[np.array([index])]
        discrepancyInstance_no_norm = oof_x_test_no_norm[np.array([index])]
        result = oof_y_test[index]
        i = UnseenData(instance=discrepancyInstance, instance_no_norm=discrepancyInstance_no_norm, correctResult=result,
                       discrepancyName=util.getParameter('DataDiscrepancy'))
        i.id = str(iCount)
        if util.has_sub_classes():
            i.hasSubClasses = True
            i.subCorrectResult = u_oof_y_test[index]
        else:
            i.hasSubClasses = False
            i.subCorrectResult = i.correctResult
        if int(i.subCorrectResult) in dataDiscrepancyClass:
            dUnseenDataList.append(i)
        iCount += 1

    if isApplyDrift:
        unseenDataList = applyDrift(ndUnseenDataList, dUnseenDataList)
    else:
        # if this method is being called from the getDataFromAllInstFile function, we do not want to apply drift
        ndUnseenDataList.extend(dUnseenDataList)
        unseenDataList = ndUnseenDataList

    # check all discrepancy classes exist
    check_discrepancy_classes(unseenDataList)

    return unseenDataList


# ----------------------------------------------------------------------------
def check_discrepancy_classes(unseenDataList):
    # todo: edit for mapped data
    if util.getParameter('DriftPattern')[0] != 'none':
        discrepancies = util.getParameter('DataDiscrepancyClass')
        if util.has_sub_classes():
            discrepancies, _ = util.map_classes_y(discrepancies)
        true_values = [u.correctResult for u in unseenDataList]
        isin = np.isin(discrepancies, true_values)
        if np.any(isin == False):
            raise Exception(
                'Not all classes were found in the data stream. Expected %s, got %s' % (discrepancies, isin))

# ----------------------------------------------------------------------------
def EqualizeNumInstances(ndUnseenDataList, dUnseenDataList):
    # make the number of ND and D instances equal so drift can be applied without re-using instances
    numUnseenInstances = util.getParameter('NumUnseenInstances')
    driftType = util.getParameter('DriftType')[0]
    driftPattern = util.getParameter('DriftPattern')[0]

    if numUnseenInstances != -1:
        ndUnseenDataList = ndUnseenDataList[:numUnseenInstances // 2]
        dUnseenDataList = dUnseenDataList[:numUnseenInstances // 2]

    numberNdUnseenInstances = len(ndUnseenDataList)
    numberDUnseenInstances = len(dUnseenDataList)

    if numberDUnseenInstances == 0:
        numInst = numberNdUnseenInstances
    else:
        numInst = min(numberNdUnseenInstances, numberDUnseenInstances)


    if driftType == 'categorical':
        # drift type is temporal, drift is applied in class blocks.  Need to make sure the stream will contain all classes
        classes = np.unique(np.array([d.correctResult for d in dUnseenDataList]))
        num_inst_per_class = int(numInst / len(classes))
        separated_classes = []
        for c in classes:
            separated_classes.append([d for d in dUnseenDataList if int(d.correctResult) == int(c)])
        separated_classes = np.array(separated_classes)

        reduced_separated_classes = []
        [reduced_separated_classes.append(s[:num_inst_per_class]) for s in separated_classes]

        dUnseenRemainingDataList = []
        [dUnseenRemainingDataList.extend(s[num_inst_per_class:]) for s in separated_classes]
        np.random.shuffle(dUnseenRemainingDataList)

        dUnseenDataList = None
        for i, r in enumerate(reduced_separated_classes):
            if i == 0:
                dUnseenDataList = reduced_separated_classes[0]
            else:
                dUnseenDataList = np.concatenate([dUnseenDataList, r])
    else:
        # drift type is temporal, so each class occurs cyclically so we can just slice of the instances
        # take the samples in order from the begining
        np.random.shuffle(dUnseenDataList)
        dUnseenRemainingDataList = dUnseenDataList[numInst:]
        dUnseenDataList = dUnseenDataList[:numInst]

    ndUnseenDataList = ndUnseenDataList[:numInst]

    if len(dUnseenRemainingDataList) == 0:
        if driftPattern == 'reoccurring':
            # take some ND instances and some D instances as the after-drift data
            dUnseenRemainingDataList = dUnseenDataList[(len(dUnseenDataList) // 3) * 2:]
            dUnseenDataList = dUnseenDataList[:(len(dUnseenDataList) // 3) * 2]
            ndUnseenDataList = dUnseenDataList[:(len(ndUnseenDataList) // 3) * 2]
        else:
            # take only D instances as after-drift instances
            dUnseenRemainingDataList = dUnseenDataList[(len(dUnseenDataList) // 3) * 2:]
            dUnseenDataList = dUnseenDataList[:(len(dUnseenDataList) // 3) * 2]
            # equalize lists
            ndUnseenDataList = ndUnseenDataList[:(len(ndUnseenDataList) // 3) * 2]

    return ndUnseenDataList, dUnseenDataList, dUnseenRemainingDataList


# ----------------------------------------------------------------------------
def check_data(data, check_data_obj, description):
    missing = []
    for u in data:
        if u not in check_data_obj:
            missing.append(u)

    if len(missing) > 0:
        raise Exception("class %s not in '%s' section of datastream" % (missing, description))


# ----------------------------------------------------------------------------
def applyDrift(ndUnseenDataList, dUnseenDataList):
    # abrupt, gradual, incremental, recurring, outlier
    driftType = util.getParameter('DriftType')
    unseenDataList = []
    driftPattern = util.getParameter('DriftPattern')
    start = 500 # after how many ND's the discrepancy starts - used in gradual, incremental and reoccurring
    end = 0  # how many discrepancies it ends on - used in gradual and incremental
    discrepFreq = 100 # 'DriftDFrequency' how many D instances are added at each re-occurence - used in gradual and incremental
    ndFreq = 100  # DriftNdFrequency how many ND instances are inbetween D ocurrences - used in gradual
    numInst = 100  # DriftNumDInstances 20 ND instances, 20 D instances - used in reoccurring
    do_equalize = True

    dUnseenRemainingDataList = []  # isntances not used in the equalization of classes
    remainingDataList = []  # instances not used in the equalization of classes

    if driftType == 'categorical':
        # order discrepancies by data class
        dUnseenDataList, remainingDataList = applyCategoricalDrift(dUnseenDataList)

    if driftPattern == 'none':
        # nd data only
        unseenDataList.extend(ndUnseenDataList)

    elif driftPattern == 'allinst':
        # all instances, ND data, then D data
        ndUnseenDataList.extend(dUnseenDataList)
        unseenDataList = ndUnseenDataList

    elif driftPattern == 'discrepancy':
        # discrepancy data only
        unseenDataList.extend(dUnseenDataList)

    elif driftPattern == 'outlier':
        # put ND and D dta together and randomly shuffle
        random.shuffle(ndUnseenDataList)
        random.shuffle(dUnseenDataList)
        if do_equalize:
            ndUnseenDataList, dUnseenDataList, dUnseenRemainingDataList = EqualizeNumInstances(ndUnseenDataList,
                                                                                               dUnseenDataList)
        unseenDataList.extend(ndUnseenDataList)
        unseenDataList.extend(dUnseenDataList)
        random.shuffle(unseenDataList)

    elif driftPattern == 'abrupt':
        if do_equalize:
            ndUnseenDataList, dUnseenDataList, dUnseenRemainingDataList = EqualizeNumInstances(ndUnseenDataList,
                                                                                               dUnseenDataList)
        # put all the ND instances first, then the D instances
        if driftType == 'categorical':
            # put all the ND instances first, then the D instances, mixing all instances abruptly after each 100 instances of each category
            classes = np.unique([u.correctResult for u in dUnseenDataList])
            splitDataList = []
            splitUnusedDataList = []
            for n, c in enumerate(classes):
                data = list(filter(lambda x: x.correctResult == c, dUnseenDataList))
                splitDataList.append(data[:50])
                splitUnusedDataList.append(data[51:])
            maxLength = max([len(s) for s in splitUnusedDataList])
            splitUnusedDataList = [s + [None] * (maxLength - len(s)) for s in splitUnusedDataList]
            dUnseenDataList = []
            for s in splitDataList:
                dUnseenDataList.extend(s)
            mixed = np.asarray([val for tup in zip(*splitUnusedDataList) for val in tup])
            mixed = mixed[mixed != np.array(None)]
            dUnseenDataList.extend(mixed)

        unseenDataList.extend(ndUnseenDataList)
        unseenDataList.extend(dUnseenDataList)

    elif driftPattern == 'gradual':
        # insert the instances so they change gradually, progressively adding more different types of discrepancy classes
        # gradually insert the discrepancies
        if do_equalize:
            ndUnseenDataList, dUnseenDataList, dUnseenRemainingDataList = EqualizeNumInstances(ndUnseenDataList,
                                                                                               dUnseenDataList)
        dIndex = 0
        dCount = 0
        iCount = 0
        for s in range(1, start):
            # add ND instance
            ndIndex = random.randint(0, len(ndUnseenDataList) - 1)
            nonDiscrepancyInstance = ndUnseenDataList[ndIndex]
            unseenDataList.append(nonDiscrepancyInstance)
            iCount += 1

        stop = False
        while stop == False and iCount < len(ndUnseenDataList) + len(dUnseenDataList):
            if iCount < len(ndUnseenDataList) + len(dUnseenDataList) - end:
                for nd in range(ndFreq):
                    if ndIndex < len(ndUnseenDataList):
                        # add ND instance
                        ndIndex = random.randint(0, len(ndUnseenDataList) - 1)
                        nonDiscrepancyInstance = ndUnseenDataList[ndIndex]
                        unseenDataList.append(nonDiscrepancyInstance)
                    else:
                        stop = True
                    iCount += 1
                for d in range(dCount):
                    for d in range(discrepFreq):
                        if dIndex < len(dUnseenDataList):
                            # add discrepancy
                            discrepancyInstance = dUnseenDataList[dIndex]
                            unseenDataList.insert(iCount, discrepancyInstance)
                            dIndex += 1
                        else:
                            stop = True
                        iCount += 1
                dCount += 1

    elif driftPattern == 'incremental':
        if do_equalize:
            ndUnseenDataList, dUnseenDataList, dUnseenRemainingDataList = EqualizeNumInstances(ndUnseenDataList,
                                                                                               dUnseenDataList)
        if driftType == 'temporal':
            raise Error('The drift type of %s is not valid for temporal drift' % (driftType))
        else:
            ndIndex = 0
            dIndex = 0
            iCount = 0
            for s in range(1, start):
                # add ND instance
                if ndIndex < len(ndUnseenDataList):
                    nonDiscrepancyInstance = ndUnseenDataList[ndIndex]
                    unseenDataList.append(nonDiscrepancyInstance)
                ndIndex += 1
                iCount += 1

            stop = False
            while stop == False and iCount < len(ndUnseenDataList) + len(dUnseenDataList):
                if iCount <= len(ndUnseenDataList) + len(dUnseenDataList) - end:
                    for d in range(discrepFreq):
                        if dIndex < len(dUnseenDataList):
                            # add discrepancy
                            discrepancyInstance = dUnseenDataList[dIndex]
                            unseenDataList.insert(iCount, discrepancyInstance)
                            dIndex += 1
                        else:
                            stop = True
                        iCount += 1
                else:
                    if ndIndex < len(ndUnseenDataList):
                        if dIndex < len(dUnseenDataList):
                            # add discrepancies at the end
                            discrepancyInstance = dUnseenDataList[dIndex]
                            unseenDataList.append(discrepancyInstance)
                            dIndex += 1
                    else:
                        stop = True
                    iCount += 1

    elif driftPattern == 'reoccurring':
        if do_equalize:
            dUnseenRemainingDataList = []
        # have both concepts for the same length of time
        ndIndex = 0
        dIndex = 0
        iCount = 0

        for s in range(start):
            if ndIndex < len(ndUnseenDataList):
                # add ND instance
                nonDiscrepancyInstance = ndUnseenDataList[ndIndex]
                nonDiscrepancyInstance.id = str(iCount)
                unseenDataList.append(nonDiscrepancyInstance)
                ndIndex += 1
            iCount += 1

        stop = False
        # while stop == False and iCount < (numDiscrepancy + numNonDiscrepancy):
        while stop == False and iCount < len(ndUnseenDataList) + len(dUnseenDataList):
            for nd in range(numInst):
                if ndIndex < len(ndUnseenDataList):
                    # add ND instance
                    nonDiscrepancyInstance = ndUnseenDataList[ndIndex]
                    nonDiscrepancyInstance.id = str(iCount)
                    unseenDataList.append(nonDiscrepancyInstance)
                    ndIndex += 1
                else:
                    stop = True
                iCount += 1
            for d in range(numInst):
                if dIndex < len(dUnseenDataList):
                    # add discrepancy
                    discrepancyInstance = dUnseenDataList[dIndex]
                    discrepancyInstance.id = str(iCount)
                    unseenDataList.insert(iCount, discrepancyInstance)
                    dIndex += 1
                else:
                    stop = True
                iCount += 1


    else:
        raise Exception('unrecognised drift type of %s' % (driftPattern))

    # number the list
    newUnseenDataList = []
    i = 0
    for u in unseenDataList:
        d = UnseenData(id=str(i), instance=u.instance, instance_no_norm=u.instance_no_normalization,
                       reducedInstance=u.reducedInstance, correctResult=u.correctResult,
                       discrepancyName=u.discrepancyName,
                       predictedResult=u.predictedResult, analysisPredict=u.analysisPredict)
        d.subCorrectResult = u.subCorrectResult
        d.hasSubClasses = u.hasSubClasses
        newUnseenDataList.append(d)
        i += 1

    # print out the drift pattern for verification
    driftTypeDisplay = ''
    for u in newUnseenDataList:
        if u.discrepancyName == 'ND':
            driftTypeDisplay += 'N'
        else:
            driftTypeDisplay += str(u.correctResult)
    # util.thisLogger.logInfo('DriftDisplay=%s' % (driftTypeDisplay))

    # set the adaption state
    for u in newUnseenDataList:
        discrepancy_detected = False
        if u.discrepancyName == 'ND' and discrepancy_detected == False:
            u.adaptState = 'before'
        else:
            discrepancy_detected = True
            driftTypeDisplay += str(u.correctResult)
            ds_model = util.getParameter('DeepStreamModelName')
            if ds_model == 'adadeepstream':
                u.adaptState = 'during'
            elif ds_model == 'deepstreamensemble':
                u.adaptState = 'after'
            else:
                raise Exception('unhandled deep stream model name of %s'%ds_model)

    for n in dUnseenRemainingDataList:
        n.adaptState = 'after'
    for n in remainingDataList:
        n.adaptState = 'after'

    newUnseenDataList.extend(dUnseenRemainingDataList)
    newUnseenDataList.extend(remainingDataList)

    # if it's reoccurring, assign he last third of the drift as after the drift
    num_drift = len([u for u in newUnseenDataList if u.adaptState == 'during'])
    num_before = len([u for u in newUnseenDataList if u.adaptState == 'before'])
    drift_max = int((num_drift // 3) * 2)
    for i, u in enumerate(newUnseenDataList):
        if i > num_before + drift_max:
            u.adaptState = 'after'

    return newUnseenDataList


# ----------------------------------------------------------------------------
def applyCategoricalDrift(dataList):
    orderedDataList = []
    remainingDataList = []
    classes = np.unique([x.correctResult for x in dataList])
    chunkSize = int((len(dataList) / len(classes) / 2))
    # split the data list into groups based on their class
    splitDataList = []

    for n, c in enumerate(classes):
        class_chunk_size = chunkSize / len(classes)
        data = list(filter(lambda x: x.correctResult == c, dataList))
        splitDataList.append(data[:chunkSize])  # store data that will be used in the drift
        remainingDataList.extend(data[chunkSize:])  # store data that will be used after the drift

    for n, c in enumerate(classes):
        class_chunk_size = int(chunkSize / len(classes))
        if n == 0:
            orderedDataList.extend(splitDataList[n][:class_chunk_size])  # first 62 instances are the first step
        else:
            data_source = []
            for i in range(n + 1):
                data_source.append(splitDataList[i][n * class_chunk_size:n * class_chunk_size + class_chunk_size])
            interleavedLists = [val for tup in zip(*data_source) for val in tup]  # interleave the drift data so far
            orderedDataList.extend(interleavedLists)

            displayData = ''
            for o in orderedDataList:
                displayData += str(o.correctResult)
            pass

    displayData = ''
    for o in orderedDataList:
        displayData += str(o.correctResult)

    np.random.shuffle(remainingDataList)
    for n in remainingDataList:
        n.adaptState = 'after'

    return orderedDataList, remainingDataList

# ----------------------------------------------------------------------------
class UnseenData:
    def __init__(self, id='', instance=None, instance_no_norm=None, reducedInstance=np.asarray([0]), correctResult=None,
                 discrepancyName=None, predictedResult=0, analysisPredict='', adaptDiscrepancy='', adaptClass='',
                 adaptState=''):
        self.id = id
        self.instance = instance
        self.instance_no_normalization = instance_no_norm
        self.reducedInstance = reducedInstance
        self.correctResult = correctResult
        self.discrepancyName = discrepancyName
        self.discrepancyResult = ''
        self.predictedResult = predictedResult
        self.subCorrectResult = np.asarray([0])
        self.analysisPredict = analysisPredict
        self.adaptDiscrepancy = adaptDiscrepancy
        self.adaptClass = adaptClass
        self.hasSubClasses = False
        self.adaptState = ''
        self.driftDetected = False

# ----------------------------------------------------------------------------
def getInstanceParameters():
    global numberNdUnseenInstances, numberDUnseenInstances
    numUnseenInstances = util.getParameter('NumUnseenInstances')

    # util.thisLogger.logInfo('NumUnseenInstances=%s' % (numUnseenInstances))

    if numUnseenInstances == -1:
        totalDiscrepancy = numberDUnseenInstances
        totalNonDiscrepancy = numberNdUnseenInstances
    else:
        dataDiscrepancyFrequency = '1in2'
        splitData = dataDiscrepancyFrequency.split('in')
        numDiscrepancy = int(splitData[0].strip())  # first number is number of discrepancies
        numNonDiscrepancy = int(splitData[1].strip())  # second number is number of  non-discrepancies
        ratio = numUnseenInstances / numNonDiscrepancy
        totalDiscrepancy = int(numDiscrepancy * ratio)
        totalNonDiscrepancy = int((numNonDiscrepancy - numDiscrepancy) * ratio)

    dataDiscrepancy = util.getParameter('DataDiscrepancy')

    return dataDiscrepancy, totalDiscrepancy, totalNonDiscrepancy, 'test'


