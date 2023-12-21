import util
import os

def setup(dnn_name, dataset_name, known_classes, unknown_classes, drift_pattern, reduction, adaptation, deepstream_model_name, data_discrepancy, is_simulate_drift):
    trainedClasses = [int(i) for i in known_classes.split('-')]
    unknownClasses = [int(i) for i in unknown_classes.split('-')]

    # load setup file
    util.params = None
    util.usedParams = []
    util.setupFile = 'input/params/default.txt'
    util.setParameter('Dnn', dnn_name)
    util.setParameter('DatasetName', dataset_name)
    util.setParameter('DriftType', drift_pattern.split('-')[0])
    util.setParameter('DriftPattern', drift_pattern.split('-')[1])
    util.setParameter('DataClasses', '%s' % trainedClasses)
    util.setParameter('DataDiscrepancyClass', '%s' % unknownClasses)
    if deepstream_model_name == '':     # if no deep stream model is specified, it is a comparison method
        util.setParameter('AnalysisModule', 'modules_compare.%s' % (adaptation))
        util.setParameter('LayerActivationReduction', 'none')
    else:
        util.setParameter('AnalysisModule', 'modules_detect_adapt.%s' % (adaptation))
        if reduction == 'jsdl':
            # JSDL requires that the data is flattened and padded prior to jsdl calculations
            reduction = 'flatten,pad,jsdl'
        util.setParameter('LayerActivationReduction', reduction)

    if dnn_name == 'vgg16':
        pad_length = 65536
    elif dnn_name == 'mobilenet':
        pad_length = 18496
    else:
        raise Exception('unhandled Dnn of %s'%dnn_name)

    util.setParameter('PadLength', pad_length)
    util.setParameter('NumUnseenInstances', -1)
    util.setParameter('DataDiscrepancy', data_discrepancy)
    util.setParameter('StreamBatchLength', 100)
    util.setParameter('Tune', False)
    util.setParameter('SimulateDriftDetection', is_simulate_drift)
    util.setParameter('AdaptationTimeDelay', 2)
    util.setParameter('DnnTrainEpochs', 20)
    if 'comparison' in deepstream_model_name:
        deepstream_model_name = deepstream_model_name.split('.')[0]
    util.setParameter('DeepStreamModelName', deepstream_model_name)

    data_dir = 'output/intermediate'
    run_data_file = os.path.join(data_dir, 'activations_%s.npz' % (util.getExperimentName()))

    util.thisLogger = util.Logger()
    return run_data_file, dataset_name, trainedClasses, data_dir