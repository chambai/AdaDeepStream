from experiment.deepstream_models import DeepStreamOs, AdaDeepStream, DeepStreamEnsemble, DeepStreamCompare
from framework import dataset, pretrain_dnn
import util
import numpy as np
from experiment import parameters
from experiment.simulate import Simulator
from experiment.eval import EvalMgr
from skmultiflow.data import DataStream
import warnings
warnings.filterwarnings('ignore')

class ExecMgr():
    def __init__(self):
        self.eval = EvalMgr()  # only required to evaluate accuracy (requires pre-loaded stream and true values)
        self.simulator = Simulator()    # simulates the drift detection and provides drift detection window true values


    def run(self, dnn_name, dataset_name, known_classes, unknown_classes, drift_pattern, deepstream_model_name, reduction, adaptation, data_discrepancy, is_simulate_drift=False):
        # extract params and setup framework
        run_data_file, dataset_name, trainedClasses, data_dir = parameters.setup(dnn_name, dataset_name, known_classes,
               unknown_classes, drift_pattern, reduction, adaptation, deepstream_model_name, data_discrepancy, is_simulate_drift)
        # get data
        x_train, y_train, x_test, y_test = dataset.getFilteredData()
        # get dnn
        dnn = pretrain_dnn.loadDnn(dataset_name, list(map(int, list(trainedClasses))),x_train, y_train, x_test, y_test)

        # get stream, set up evaluation and simulation
        stream_data = self.eval.get_stream()
        batch_length = util.getParameter('StreamBatchLength')
        self.simulator.setup(self.eval.all_unseen_data)

        # set deep stream model
        if deepstream_model_name == 'deepstreamos':
            deepstream_model = DeepStreamOs(dnn, reduction, self.eval, self.simulator)
        elif deepstream_model_name == 'adadeepstream':
            deepstream_model = AdaDeepStream(dnn, 0, reduction, adaptation, self.eval, self.simulator) # Todo: specify learning rate from user input
        elif deepstream_model_name == 'deepstreamensemble':
            deepstream_model = DeepStreamEnsemble(dnn, 0, reduction, adaptation, self.eval, self.simulator) # Todo: specify learning rate from user input
        elif 'comparison' in deepstream_model_name:
            deepstream_model = DeepStreamCompare(dnn, adaptation, self.eval, self.simulator)
        else:
            raise Exception('Unrecognised deepstream_model of %s'%deepstream_model_name)

        # setup peripheral info
        deepstream_model.data_dir = data_dir                # dir to store activations
        deepstream_model.run_data_file = run_data_file      # activations storage file

        # offline deepstream model training
        deepstream_model.fit(x_train, y_train)

        # online inference
        stream = DataStream(stream_data, np.zeros(len(stream_data)))
        while stream.has_more_samples():
            X_batch, _ = stream.next_sample(batch_length)   # get the next sample from the stream
            X = self.eval.get_dimensioned_data(X_batch)     # DataStream requires flattened data, reshape it
            deepstream_model.partial_fit(X)                 # partial fit

        deepstream_model.stop()


if __name__ == '__main__':
    # Example
    # mobilenet fashion class data example (our dseadapt method)
    # DeepStreamEnsemble method (Ours)
    exec = ExecMgr()

    # AdaDeepStream
    exec.run(dnn_name='vgg16',
             dataset_name='mnistfashion',
             known_classes='0-1-2-3-5-6-8-9',
             unknown_classes='4-7',
             drift_pattern='categorical-reoccurring',
             deepstream_model_name='adadeepstream',  # deepstreamos, adadeepstream, deepstreamensemble
             reduction='dscbir',  # jdsiverge, dsdivergelast, dscbir, blockcbir
             adaptation='dsadapt',  # dsadapt, dseadapt
             data_discrepancy='CE',  # CE, Outlier
             )

    # # OCL adaptation methods - AdaDeepStream
    # exec.run(dnn_name='vgg16',
    #       dataset_name='mnistfashion',
    #       known_classes='0-1-2-3-5-6-8-9',
    #       unknown_classes='4-7',
    #       drift_pattern='temporal-abrupt',
    #       deepstream_model_name='adadeepstream',
    #       reduction='dscbir',   # dscbir, jsdl
    #       adaptation='ocler',   # ocllwf, ocler, oclicarl, oclmirrv
    #       data_discrepancy='CE',
    #         )
    #
    # # RSB comparison method - AdaDeepStream
    # exec.run(dnn_name='vgg16',
    #       dataset_name='mnistfashion',
    #       known_classes='0-1-2-3-5-6-8-9',
    #       unknown_classes='4-7',
    #       drift_pattern='temporal-abrupt',
    #       deepstream_model_name='adadeepstream.comparison',
    #       reduction='',
    #       adaptation='rsb',
    #       data_discrepancy='CE')