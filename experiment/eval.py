import util
from collections import defaultdict
from framework import datastream
import numpy as np
from framework import analyse

class EvalMgr():
    # purely for evaluation purposes
    # data stream created from test data and is split into states - before during and after
    # just the x values of the stream is provided to the caller
    # caller uses the stream in the deep stream models
    # caller adds the results of each stream window via add_stream_results method
    # caller invokes the evaluate method at the end of the stream
    # accuracy of each adapt state section of the stream is calculated when evaluate is called

    def __init__(self):
        self.all_unseen_data = None
        self.inst_index = 0
        self.drift_det_window_times = np.array([])
        self.adapt_times = np.array([])


    def get_stream(self):
        # gets the stream set up with evaluation objects and stores in here
        # returns just the X values
        self.all_unseen_data = datastream.getData()
        self.inst_index = 0
        stream_data = np.array([x.instance.flatten() for x in self.all_unseen_data], dtype=float)
        return stream_data

    def get_dimensioned_data(self, X):
        # shape the data to the original dimensions
        dataShape = self.all_unseen_data[0].instance.shape  # (1, 32, 32, 3)
        X = np.reshape(X, (len(X), dataShape[1], dataShape[2], dataShape[3]))
        return X

    def get_true_values(self, batch_number):
        batch_length = util.getParameter('StreamBatchLength')
        vals = [x.correctResult for x in self.all_unseen_data[(batch_number-1)*100:batch_length-1]]
        return vals

    def add_stream_results(self, acts, dnn_predicts, batch_result):
        if self.all_unseen_data is not None:
            # Stores results from the stream ready for future evaluation
            if len(batch_result) == 2:
                batchDiscrepResult = batch_result[0]
                analysis_predicts = batch_result[1]
                adapt_discrepancy = np.empty(len(batchDiscrepResult), dtype=str)
                adapt_discrepancy[:] = "-"
                adapt_class = np.empty(len(batchDiscrepResult), dtype=str)
                adapt_class[:] = "-"
            elif len(batch_result) == 4:
                batchDiscrepResult = batch_result[0]
                analysis_predicts = batch_result[1]
                adapt_discrepancy = batch_result[2]
                adapt_class = batch_result[3]
            else:
                batchDiscrepResult = batch_result
                analysis_predicts = np.empty(len(batchDiscrepResult), dtype=str)
                analysis_predicts[:] = "-"
                adapt_discrepancy = np.empty(len(batchDiscrepResult), dtype=str)
                adapt_discrepancy[:] = "-"
                adapt_class = np.empty(len(batchDiscrepResult), dtype=str)
                adapt_class[:] = "-"


            for i, (act, dnnPredict, res, analysisPredict, adaptDiscrepancy, adaptClass) in enumerate(zip(acts, dnn_predicts, batchDiscrepResult, analysis_predicts, adapt_discrepancy, adapt_class)):
                self.all_unseen_data[self.inst_index].reducedInstance = np.array(act) #act.reshape(1,act.shape[0])
                self.all_unseen_data[self.inst_index].predictedResult = dnnPredict
                self.all_unseen_data[self.inst_index].discrepancyResult = res
                self.all_unseen_data[self.inst_index].analysisPredict = analysisPredict
                self.all_unseen_data[self.inst_index].adaptDiscrepancy = adaptDiscrepancy
                self.all_unseen_data[self.inst_index].adaptClass = adaptClass
                if (hasattr(analyse.module, 'analysis') and analyse.module.analysis.is_adapted) or  (hasattr(analyse.module, 'is_adapted') and analyse.module.is_adapted):
                    self.all_unseen_data[self.inst_index].driftDetected = True
                self.inst_index += 1
        else:
            util.thisLogger.logInfo('Stream evaluation objects have not been set up. No stream evaluation will occur.')


    def evaluate_stream(self):
        # evaluate the stream results stored in here
        # setup a dictionary, splitting the data into adaptation states
        unseenDataDict = defaultdict(list)
        for item in self.all_unseen_data:
            unseenDataDict[item.adaptState].append(item)

        # drift_instances = [u.id for u in self.all_unseen_data if u.driftDetected]
        # util.thisLogger.logInfo('DriftDetectionInstances=%s' % (drift_instances))

        total_correct = 0
        total_unseen = 0
        for i, (adapt_state, unseenData) in enumerate(unseenDataDict.items()):
            # only show the after accuracy
            if self.all_unseen_data[0].adaptClass != '-':
                # calculate accuracy of each section
                correct = [x.correctResult for x in unseenData if int(x.correctResult) == int(x.adaptClass)]
                acc = round(len(correct) / len(unseenData),6)
                util.thisLogger.logInfo('%s accuracy=%s (%s / %s)' % (adapt_state, acc, len(correct), len(unseenData)))
                total_correct += len(correct)
                total_unseen += len(unseenData)

        correct = [x.correctResult for x in self.all_unseen_data if int(x.correctResult) == int(x.adaptClass)]
        acc = round(len(correct) / len(self.all_unseen_data),6)
        util.thisLogger.logInfo('total accuracy=%s (%s / %s)' % (acc, len(correct), len(self.all_unseen_data)))



    def evaluate_timings(self):
        # calculate timings
        window_size = util.getParameter('StreamBatchLength')
        # calculate average per instance
        unadapted_instance_processing_time = self.drift_det_window_times / window_size
        avg_unadapted_instance_time = round(np.average(unadapted_instance_processing_time), 3)
        sd_unadapted_instance_time = round(np.std(unadapted_instance_processing_time), 3)
        util.thisLogger.logInfo('Average inference time (and std dev) per instance (ms)=%s (%s)' % (
        avg_unadapted_instance_time, sd_unadapted_instance_time))

        if len(self.adapt_times) > 0:
            avg_adapted_window_time = round(np.average(self.adapt_times), 3)
            sd_adapted_window_time = round(np.std(self.adapt_times), 3)
            util.thisLogger.logInfo(
                'Average adaptation time (and std dev) (s)=%s (%s)' % (avg_adapted_window_time, sd_adapted_window_time))


