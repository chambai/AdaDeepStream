import util
from modules_detect_adapt import ads_clusterer
from skmultiflow.drift_detection.ddm import DDM
from skmultiflow.trees import HoeffdingAdaptiveTreeClassifier
import threading
import copy
from framework import analyse
import time
import datetime
from framework import pretrain_dnn as dnnModel
from framework.dnn_models import DNNPytorchAdaptive
from framework import dataset
import numpy as np


class Analysis:
    def __init__(self):
        self.classifiers = None
        self.streamClassifierNames = []
        self.threads = []
        self.drift_detector = None
        self.adaptive_dnn = None
        self.run_update_thread = False
        self.update_thread_started = False
        self.adapt = False
        self.data_lock = threading.Lock()
        self.clus_lock = threading.Lock()
        self.buffer_windows_count = 0
        self.num_window_buffers = 0
        self.x_act_buffer = np.array([0])
        self.x_data_buffer = np.array([0])
        self.y_data_buffer = np.array([0])
        self.num_previous_buffers = 0
        self.buffer_prev_windows_count = 0
        self.x_act_prev_buffer = np.array([0])
        self.x_data_prev_buffer = np.array([0])
        self.y_data_prev_buffer = np.array([0])
        self.adaption_thread_error = False
        self.adaption_error_message = ''
        self.is_adapted = False
        self.is_adapting = False
        self.window_count = 0
        self.instance_count = 0
        self.drift_trigger_count = 0
        self.use_clustering = False
        self.clusterer = ads_clusterer.SamKnn()
        self.x_train = None
        self.y_train = None
        self.x_train_act = None
        self.y_train_act = None
        self.use_class_buffer = False
        self.sequential_adapt_override = False
        self.adapt_times = []

    def setup_activation_classifiers(self, x_train_batch, act_train_batch, y_train_batch):
        self.x_train_act = act_train_batch
        self.y_train_act = y_train_batch

        if self.use_clustering:
            self.clus_lock.acquire()
            self.clusterer.partial_fit(x_train_batch, act_train_batch, y_train_batch)
            self.clus_lock.release()

        self.classifier = HoeffdingAdaptiveTreeClassifier()
        self.classifier.fit(act_train_batch, y_train_batch)

        class_name = type(self.classifier).__name__
        self.streamClassifierName = class_name

        util.thisLogger.logInfo('streamClassifierNames=%s' % (self.streamClassifierName))

        util.thisLogger.logInfo('%s activation classifier trained'%self.streamClassifierName)
        accuracies = self.logTrainAccuracy(act_train_batch, y_train_batch)

        self.start_model_update_thread()
        return accuracies

    def logTrainAccuracy(self, act_train, y_train):
        accuracies = self.getAccuracy(act_train, y_train)
        [util.thisLogger.logInfo('StreamClassifierTrainAccuracy%s=%s' % (i, a)) for i, a in
         enumerate(accuracies)]
        return accuracies

    def logTestAccuracy(self, act_unseen, y_unseen):
        accuracies = self.getAccuracy(act_unseen, y_unseen)
        [util.thisLogger.logInfo('StreamClassifierTestAccuracy%s=%s' % (i, a)) for i, a in
         enumerate(accuracies)]
        return accuracies

    def getAccuracy(self, x, y):
        accuracies = []
        y_predict = self.classifier.predict(x)
        bool_array = y == y_predict
        num_correct = np.count_nonzero(bool_array)
        accuracies.append(num_correct / len(y))
        return np.array(accuracies)

    def setup_drift_detectors(self):
        self.drift_detector = DDM()

    def get_training_data(self):
        # get original training data
        if util.has_sub_classes():
            self.x_train, self.y_train, _, _ = dataset.getFilteredData(isMap=True, do_normalize=True)
        else:
            self.x_train, self.y_train, _, _ = dataset.getFilteredData(isMap=False, do_normalize=True)
        return self.x_train, self.y_train

    def get_discrepancy_data(self):
        # get original training data
        if util.has_sub_classes():
            x_train_orig, y_train_orig, _, _ = dataset.getOutOfFilteredData(isMap=True)
        else:
            x_train_orig, y_train_orig, _, _ = dataset.getOutOfFilteredData(isMap=False)
        return x_train_orig, y_train_orig

    def setup_adaptive_dnn(self):
        self.adaptive_dnn = DNNPytorchAdaptive(adaption_strategies=['always_update', 'lin'], buffer_type='none',
                                               freeze_layer_type='classification', epochs=3)
        self.adaptive_dnn.load((dnnModel.getFullDnnModelPathAndName()))
        x_train_orig, y_train_orig = self.get_training_data()
        self.adaptive_dnn.fit(x_train_orig, y_train_orig)  # applies extra training if required


    def start_model_update_thread(self):
        if not self.sequential_adapt_override:
            self.run_update_thread = True
            update_thread = threading.Thread(target=self.updateClassifiers, args=())
            update_thread.start()
            util.thisLogger.logInfo('continuing...')


    def updateClassifiers(self):
        # update models on incorrect instances only
        while self.run_update_thread:
            update_thread_started = True
            while self.adapt:
                self.adapt_dnn()
                time.sleep(0.01)

            time.sleep(1)
        util.thisLogger.logInfo('adaption thread ended')


    def processUnseenStreamBatch(self, xdata_unseen_batch, act_unseen_batch, dnnPredict_batch):
        self.window_count += 1
        self.instance_count = self.window_count*len(xdata_unseen_batch)

        dnnPredict_batch = analyse.getPredictions(self.adaptive_dnn, xdata_unseen_batch, self.adaptive_dnn.all_classes)

        # streaming classifier
        self.classifier.partial_fit(act_unseen_batch, dnnPredict_batch)
        classifier_predictions = self.classifier.predict(act_unseen_batch)

        dnn_predict_str = ''
        stream_classifier_predict_str = ''
        drift_input_str = ''
        drift_output_str = ''
        for a, d, c in zip(act_unseen_batch, dnnPredict_batch, classifier_predictions):
            dnn_predict_str += str(int(d))
            stream_classifier_predict_str += str(int(c))
            drift_input, drift_result = self.get_drift(d, c, self.drift_detector)
            drift_input_str += drift_input
            drift_output_str += drift_result

        classifier_discrepancies = self.convertDriftToDiscrepancies(list(drift_output_str))

        return dnnPredict_batch, classifier_predictions, classifier_discrepancies

    def adaptation(self, has_drift, act_unseen_batch, xdata_unseen_batch, dnnPredict_batch, classifier_predictions, drift_result, true_values):
        self.is_adapted = False

        if has_drift:
            self.drift_trigger_count += 1

            self.data_lock.acquire()
            self.concatenate_buffer_data(act_unseen_batch, xdata_unseen_batch, true_values)

            if self.use_clustering:
                self.clus_lock.acquire()
                self.clusterer.partial_fit(xdata_unseen_batch, act_unseen_batch, true_values)
                self.clus_lock.release()

            self.data_lock.release()

            # adapt lock will be taken if doing adaption sequentially
            sequential_adapt = False
            if sequential_adapt or (not sequential_adapt and self.drift_trigger_count == 1):
                self.adapt_dnn()
                # predict unseen instances via the DNN
                dnnPredict_batch = analyse.getPredictions(self.adaptive_dnn, xdata_unseen_batch, self.adaptive_dnn.all_classes)
                classifier_predictions = self.classifier.predict(act_unseen_batch)
                self.is_adapted = True

                if not sequential_adapt:
                    self.adapt = False # todo: temp
            else:
                self.adapt = True
        else:
            if self.num_window_buffers > 0:
                if self.use_clustering:
                    self.clus_lock.acquire()
                    cp = self.clusterer.predict_act(act_unseen_batch)
                    self.clus_lock.release()
                    self.concatenate_buffer_data(act_unseen_batch, xdata_unseen_batch, cp)
                else:
                    self.concatenate_buffer_data(act_unseen_batch, xdata_unseen_batch, dnnPredict_batch)

        if self.adaption_thread_error:
            raise Exception(self.adaption_error_message)

        adapt_discrepancy = []
        adapt_class_dnn = []
        adapt_class_clust = []
        discrepancyType = util.getParameter('DataDiscrepancy')

        # if not self.is_adapting:
        dnnPredict_batch = self.adaptive_dnn.predict(xdata_unseen_batch)
        for r, d, c in zip(drift_result, dnnPredict_batch, classifier_predictions):
            if d in util.getParameter('DataClasses'):
                # original class
                adapt_class_dnn.append(d)   # if dnn predicted result is a known class, use value from streaming classifier # todo should i be doing this???
            else:
                adapt_class_dnn.append(d)   # if dnn predicted result is an unknown class, use value from DNN
            if r == 'D':
                adapt_discrepancy.append(discrepancyType)
            if r == 'N':
                adapt_discrepancy.append('ND')

        if self.use_clustering:
            self.clus_lock.acquire()
            cp = []
            cp = self.clusterer.predict_act(act_unseen_batch)
            self.clus_lock.release()

            for i, (r, d) in enumerate(zip(drift_result, dnnPredict_batch)):
                if self.use_clustering:
                    adapt_class_clust.append(cp[i])
                if r == 'D':
                    adapt_discrepancy.append(discrepancyType)
                if r == 'N':
                    adapt_discrepancy.append('ND')

            # calc accuracy
            correct = [cl for i, cl in enumerate(adapt_class_clust) if cl == adapt_class_dnn[i]]
            acc = round(len(correct) / len(adapt_class_clust), 6)

            if acc > 0.8:
                util.thisLogger.logInfo('Using DNN predictions')
                adapt_class = adapt_class_dnn
            else:
                util.thisLogger.logInfo('Using clusterer predictions')
                adapt_class = adapt_class_clust
        else:
            adapt_class = adapt_class_dnn

        return self.adaptive_dnn, drift_result, classifier_predictions, adapt_discrepancy, adapt_class


    def adapt_dnn(self):
        try:
            # concatenate current data with any stored data
            self.data_lock.acquire()
            activations, xdata, ydata = self.get_buffer_data()
            prev_act, prev_xdata, prev_ydata = self.get_previous_buffer_data()
            if len(activations) > 1 and self.num_previous_buffers > 1:
                self.concatenate_previous_buffer_data(activations, xdata, ydata)    # store this data that is used for the adaption for future use
                # concatenate previous adaption data with current buffer data
                activations, xdata, ydata = self.concatenate(prev_act, prev_xdata, prev_ydata, activations, xdata, ydata)

            if activations.shape[0] > 1:
                self.is_adapting = True
                util.thisLogger.logInfo('adapting...')
                self.clear_buffer_data()

                self.data_lock.release()

                # re-train on window data
                start = datetime.datetime.now()
                # for classifier in self.classifiers:
                copy_method = getattr(self.classifier, "get_copy", None)
                if callable(copy_method):
                    classifier_copy = self.classifier.get_copy()
                else:
                    classifier_copy = copy.deepcopy(self.classifier)
                adaptive_dnn_copy = self.adaptive_dnn.get_copy()

                if self.use_clustering:
                    self.clus_lock.acquire()
                    ydata = self.clusterer.predict_act(activations)
                    if self.use_class_buffer:
                        xdata, activations, ydata = self.clusterer.get_buffer_data(ydata, 100)
                    self.clus_lock.release()


                threads = []
                # update adaptive DNN copy
                adaptive_dnn_copy.partial_fit(xdata, ydata)

                # update streaming classifier copy
                classifier_copy.partial_fit(np.asarray(activations), np.asarray(ydata))

                for i, t in enumerate(threads):
                    t.start()
                    time.sleep(1)

                for t in threads:
                    t.join()
                    time.sleep(1)

                self.adaptive_dnn = adaptive_dnn_copy
                self.classifier = classifier_copy
                self.is_adapting = False

                stop = datetime.datetime.now()
                time_diff = stop - start
                self.adapt_times.append(time_diff.total_seconds())
            else:
                self.data_lock.release()
        except Exception as e:
            util.thisLogger.logInfo('Error in adaption thread: %s'%(e))
            self.adaption_thread_error = True
            self.adaption_error_message = str(e)
            raise(e)

    def convertDriftToDiscrepancies(self, batchDrift):
        discrepancyStr = ''.join(['N'] * len(batchDrift))
        batch_list = [*batchDrift]
        if 'W' in batchDrift:
            discreps = []
            for d in batch_list:
                if d == 'W':
                    discreps.append('N')
                else:
                    discreps.append('D')
            discrepancyStr = ''.join([str(t) for t in discreps])

        return list(discrepancyStr)

    def concatenate_buffer_data(self, x_act, x_data, y):
        if len(self.x_act_buffer) == util.getParameter('StreamBatchLength') * self.num_window_buffers:
            # window buffer reached - remove first window
            self.x_act_buffer = self.x_act_buffer[len(x_act):]
            self.x_data_buffer = self.x_data_buffer[len(x_act):]
            self.y_data_buffer = self.y_data_buffer[len(x_act):]
        if len(self.x_act_buffer) == 1:
            self.x_act_buffer = np.zeros((0, x_act.shape[1]))
            self.x_data_buffer = np.zeros((0, x_data.shape[1], x_data.shape[2], x_data.shape[3]))
            self.y_data_buffer = np.zeros((0), dtype=np.int64)
        self.x_act_buffer = np.vstack((self.x_act_buffer, x_act))
        self.x_data_buffer = np.vstack((self.x_data_buffer, x_data))
        self.y_data_buffer = np.hstack((self.y_data_buffer, y))
        self.buffer_windows_count += 1


    def concatenate(self, x_act_1, x_data_1, y_1, x_act_2, x_data_2, y_2):
        if len(x_act_1) == 1 and len(x_act_2) == 1:
            # no concatenation required
            return x_act_1, x_data_1, y_1
        else:
            if len(x_act_1) == 1:
                x_act_1 = np.zeros((0, x_act_2.shape[1]))
                x_data_1 = np.zeros((0, x_data_2.shape[1], x_data_2.shape[2], x_data_2.shape[3]))
                y_1 = np.zeros((0), dtype=np.int64)
            x_act = np.vstack((x_act_1, x_act_2))
            x_data = np.vstack((x_data_1, x_data_2))
            y = np.hstack((y_1, y_2))
            return x_act, x_data, y



    def get_buffer_data(self):
        return self.x_act_buffer, self.x_data_buffer, self.y_data_buffer



    def clear_buffer_data(self):
        if self.num_window_buffers > 0:
            # clear if the number of buffers has been reached
            if self.buffer_windows_count >= self.num_window_buffers:
                self.x_act_buffer = np.array([0])
                self.x_data_buffer = np.array([0])
                self.y_data_buffer = np.array([0])
                self.buffer_windows_count = 0
        else:
            self.x_act_buffer = np.array([0])
            self.x_data_buffer = np.array([0])
            self.y_data_buffer = np.array([0])
            self.buffer_windows_count = 0


    def concatenate_previous_buffer_data(self, x_act, x_data, y):
        if len(self.x_act_prev_buffer) == util.getParameter('StreamBatchLength') * self.num_previous_buffers:
            # window buffer reached - remove first window
            self.x_act_prev_buffer = self.x_act_prev_buffer[len(x_act):]
            self.x_data_prev_buffer = self.x_data_prev_buffer[len(x_act):]
            self.y_data_prev_buffer = self.y_data_prev_buffer[len(x_act):]
        if len(self.x_act_prev_buffer) == 1:
            self.x_act_prev_buffer = np.zeros((0, x_act.shape[1]))
            self.x_data_prev_buffer = np.zeros((0, x_data.shape[1], x_data.shape[2], x_data.shape[3]))
            self.y_data_prev_buffer = np.zeros((0), dtype=np.int64)
        self.x_act_prev_buffer = np.vstack((self.x_act_prev_buffer, x_act))
        self.x_data_prev_buffer = np.vstack((self.x_data_prev_buffer, x_data))
        self.y_data_prev_buffer = np.hstack((self.y_data_prev_buffer, y))
        self.buffer_prev_windows_count += 1



    def get_previous_buffer_data(self):
        return self.x_act_prev_buffer, self.x_data_prev_buffer, self.y_data_prev_buffer



    def clear_previous_buffer_data(self):
        if self.num_previous_buffers > 0:
            # clear if the number of buffers has been reached
            if self.buffer_prev_windows_count >= self.num_previous_buffers:
                self.x_act_prev_buffer = np.array([0])
                self.x_data_prev_buffer = np.array([0])
                self.y_data_prev_buffer = np.array([0])
                self.buffer_prev_windows_count = 0
        else:
            self.x_act_prev_buffer = np.array([0])
            self.x_data_prev_buffer = np.array([0])
            self.y_data_prev_buffer = np.array([0])
            self.buffer_prev_windows_count = 0


    def get_drift(self, dnn_predict, streamClas_predict, drift_detector):
        if int(dnn_predict) == int(streamClas_predict):
            drift_input = 0
        else:
            drift_input = 1

        drift_result = 'N'
        drift_detector.add_element(
            drift_input)  # must pass in a 0 or 1 here. 0 = correctly classified, 1 = incorrectly classified
        if drift_detector.detected_warning_zone():
            drift_result = 'W'
        elif drift_detector.detected_change():
            drift_result = 'C'

        return str(drift_input), drift_result

    def processStreamInstances(self):
        # only used if instances need to be processed on a separate thread
        temp = 'not implemented'

    def stopProcessing(self):
        self.adapt = False

    def getAnalysisParams(self):
        parameters = []
        # return a list of variable parameters in the format [[name,value],[name,value]...[name,value]]
        parameters.append(['none', 0])
        return parameters