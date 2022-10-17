from core import dnn, extract
import util
from modules_detect import drift_wcn_true as drift
from skmultiflow.drift_detection.ddm import DDM
from skmultiflow.trees import HoeffdingAdaptiveTreeClassifier
import threading
import copy
from core import analyse
import time
import os
import datetime
from core import dnn as dnnModel
from core.pytorch_wrapper import DNNPytorchAdaptive
from core import dataset
from joblib import dump, load
import numpy as np
from modules_adapt import clusterer

class Analysis:
    def __init__(self):
        self.classifiers = []  # trained on true label at training time
        self.streamClassifierNames = []
        self.threads = []
        self.drift_detectors = []
        self.adaptive_dnn = None
        self.run_update_thread = False
        self.update_thread_started = False
        self.adapt = False
        self.adapt_lock = threading.Lock()
        self.model_lock = threading.Lock()
        self.data_lock = threading.Lock()
        self.clus_lock = threading.Lock()
        self.act_vote = False
        self.buffer_windows_count = 0
        self.num_window_buffers = 0
        self.x_act_buffer = np.array([0])
        self.x_data_buffer = np.array([0])
        self.y_data_buffer = np.array([0])
        self.num_previous_buffers = 0      # saves the previous mem buffers from previous drifts and uses them in all subsequent adaptions
        self.buffer_prev_windows_count = 0
        self.x_act_prev_buffer = np.array([0])
        self.x_data_prev_buffer = np.array([0])
        self.y_data_prev_buffer = np.array([0])
        self.adaption_thread_error = False
        self.adaption_error_message = ''
        self.adaption_times = []
        self.is_adapted = False
        self.use_artificial_drift_detection = False # instead of detecting drift, it uses true values to determine when drift has actually started
        self.adapt_every_n_windows = 0    # once drift has been detected, adapt every n windows
        self.window_count = 0
        self.instance_count = 0
        self.drift_triggered = False
        self.drift_triggered_window = 0
        self.drift_trigger_count = 0
        self.augment = False
        self.use_clustering = False
        self.use_anomaly_detection = False
        self.clusterer = clusterer.SamKnn()
        self.x_train = None
        self.y_train = None
        self.x_train_act = None
        self.y_train_act = None
        self.sequential_adapt_override = False
        self.use_class_buffer = False

    def setup_activation_classifiers(self, x_train_batch, act_train_batch, y_train_batch):
        self.x_train_act = act_train_batch
        self.y_train_act = y_train_batch

        if self.use_clustering:
            self.clus_lock.acquire()
            self.clusterer.partial_fit(x_train_batch, act_train_batch, y_train_batch)
            self.clus_lock.release()

        if self.use_anomaly_detection:
            self.hstrees.fit(act_train_batch, y_train_batch)

        self.classifiers.append(HoeffdingAdaptiveTreeClassifier())

        class_file_names = []
        file_loaded = []
        for i, c in enumerate(self.classifiers):
            class_name = type(c).__name__
            self.streamClassifierNames.append(class_name)
            class_dir = os.path.join(dnn.getDnnModelPath(), 'act_classifiers')
            if not os.path.exists(class_dir):
                os.mkdir(class_dir)
            class_file_name = os.path.join(class_dir,
                                           dnn.getDnnModelName().replace('.plt', '')) + '_' + class_name + '_' + \
                              util.getParameter('LayerActivationReduction')[-1] + '.joblib'
            class_file_names.append(class_file_name)
            if os.path.exists(class_file_name):
                self.classifiers[i] = load(class_file_name)
                file_loaded.append(True)
            else:
                file_loaded.append(False)
                # start model update thread ready for when the stream batches start arriving
                thread = threading.Thread(target=c.fit, args=(act_train_batch, y_train_batch))
                self.threads.append(thread)

        util.thisLogger.logInfo('streamClassifierNames=%s' % (';'.join(self.streamClassifierNames)))
        for i, t in enumerate(self.threads):
            t.start()
            util.thisLogger.logInfo('Training thread for %s activation classifier started' % (self.streamClassifierNames[i]))
            time.sleep(1)

        for t in self.threads:
            t.join()
            time.sleep(1)

        for i, (c, f, file_load) in enumerate(zip(self.classifiers, class_file_names, file_loaded)):
            if file_load == False:  # if the classifier was not loaded from file, save it to file
                dump(c, f)

        util.thisLogger.logInfo('%s activation classifiers trained' % (len(self.classifiers)))
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
        for i, classifier in enumerate(self.classifiers):
            y_predict = classifier.predict(x)
            bool_array = y == y_predict
            num_correct = np.count_nonzero(bool_array)
            accuracies.append(num_correct / len(y))
        return np.array(accuracies)

    def setup_drift_detectors(self):
        self.drift_detectors.append(DDM())
        # self.drift_detectors.append(DDM())
        if len(self.drift_detectors) != len(self.classifiers):
            raise Exception('The number of drift detectors must match the number of classifiers')

    def get_training_data(self):
        # get original training data
        mapOriginalYValues = util.getParameter('MapOriginalYValues')
        if len(mapOriginalYValues) == 0:
            self.x_train, self.y_train, _, _ = dataset.getFilteredData(isMap=False, do_normalize=True)
        else:
            self.x_train, self.y_train, _, _ = dataset.getFilteredData(isMap=True, do_normalize=True)
        return self.x_train, self.y_train

    def get_discrepancy_data(self):
        # get original training data
        mapOriginalYValues = util.getParameter('MapOriginalYValues')
        if len(mapOriginalYValues) == 0:
            x_train_orig, y_train_orig, _, _ = dataset.getOutOfFilteredData(isMap=False)
        else:
            x_train_orig, y_train_orig, _, _ = dataset.getOutOfFilteredData(isMap=True)
        return x_train_orig, y_train_orig

    def setup_adaptive_dnn(self, adaption_strategies, buffer_type='none'):
        self.adaptive_dnn = DNNPytorchAdaptive(adaption_strategies=adaption_strategies, buffer_type=buffer_type)
        self.adaptive_dnn.load((dnnModel.getFullDnnModelPathAndName()))
        x_train_orig, y_train_orig = self.get_training_data()
        self.adaptive_dnn.fit(x_train_orig, y_train_orig)  # applies extra training if required


    def start_model_update_thread(self):
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
                util.thisLogger.logInfo('number of adaption activations: %s' % (len(activations)))
                self.clear_buffer_data()

                self.data_lock.release()

                classifier_copies = []
                # re-train on window data
                start = datetime.datetime.now()
                for classifier in self.classifiers:
                    copy_method = getattr(classifier, "get_copy", None)
                    if callable(copy_method):
                        classifier_copies.append(classifier.get_copy())
                    else:
                        classifier_copies.append(copy.deepcopy(classifier))
                adaptive_dnn_copy = self.adaptive_dnn.get_copy()

                if self.act_vote:
                    adaptive_dnn_copy.add_activations(activations, ydata,
                                                      classifier_copies[-1])  # add original classifier to adaptive dnn

                if self.augment:
                    ydata_2d = np.reshape(ydata, (ydata.shape[0], 1))
                    xdata, ydata = dataset.augment_extend_data(xdata, ydata_2d, 2)
                    ydata = np.reshape(ydata, (ydata.shape[0]))

                if self.use_clustering:
                    self.clus_lock.acquire()
                    ydata = self.clusterer.predict_act(activations)
                    util.thisLogger.logInfo('More samples on classes: %s' % (np.unique(ydata)))
                    if self.use_class_buffer:
                        xdata, activations, ydata = self.clusterer.get_buffer_data(ydata, 100)
                    self.clus_lock.release()

                threads = []
                # update adaptive DNN copy
                adaptive_dnn_copy.partial_fit(xdata, ydata)

                # update streaming classifier copy
                for classifier_copy in classifier_copies:
                    classifier_copy.partial_fit(np.asarray(activations), np.asarray(ydata))

                for i, t in enumerate(threads):
                    t.start()
                    time.sleep(1)

                for t in threads:
                    t.join()
                    time.sleep(1)

                if self.act_vote:
                    adaptive_dnn_copy.add_activations(activations, ydata, classifier_copies[-1])  # add adapted classifier to adaptive dnn

                self.adaptive_dnn = adaptive_dnn_copy
                self.classifiers = classifier_copies
                util.thisLogger.logInfo('Models updated')

                stop = datetime.datetime.now()
                time_diff = stop - start
                util.thisLogger.logInfo('DnnAdaptionTime=%s' % (time_diff.total_seconds()))
            else:
                self.data_lock.release()
        except Exception as e:
            util.thisLogger.logInfo('Error in adaption thread: %s'%(e))
            self.adaption_thread_error = True
            self.adaption_error_message = str(e)
            raise(e)



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


    def get_training_data_sample(self):
        idxs = np.random.choice(self.x_train.shape[0], self.num_window_buffers*util.getParameter('StreamBatchLength'), replace=False)
        x_sample = self.x_train[idxs]
        y_sample = self.y_train[idxs]

        # get activations
        act_sample = extract.getActivationData2(self.adaptive_dnn, x_sample, y_sample)

        return act_sample, x_sample, y_sample


    def processUnseenStreamBatch(self, xdata_unseen_batch, act_unseen_batch, dnnPredict_batch, true_values, true_discrep):
        # stream is processed by CVFDT (Hoeffding adaptive tree) - Concept drift adapting Hoeffding Tree
        # batch from the stream is provided to this function
        # result consists of a list of strings of the values: ND (Non-Discrepancy), D (Discrepancy) and NONE (No result determined)

        self.window_count += 1
        self.instance_count = self.window_count*len(xdata_unseen_batch)

        # Predict the unseen instances via  the classifier
        classifier_predictions = []
        # self.model_lock.acquire()

        # predict unseen instances via the DNN
        if self.act_vote:
            self.adaptive_dnn.add_activations(act_unseen_batch)

        dnnPredict_batch = analyse.getPredictions(self.adaptive_dnn, xdata_unseen_batch, self.adaptive_dnn.all_classes)

        # streaming classifier
        for i, classifier in enumerate(self.classifiers):
            classifier.partial_fit(act_unseen_batch, dnnPredict_batch)
            y_classifier_batch = classifier.predict(act_unseen_batch)
            classifier_predictions.append(y_classifier_batch)

        # self.model_lock.release()

        true_label_str = ''
        dnn_predict_str = ''
        for d, t in zip(dnnPredict_batch, true_values):
            true_label_str += str(int(t))
            dnn_predict_str += str(int(d))
        util.thisLogger.logInfo('True_label:   %s' % (true_label_str))
        util.thisLogger.logInfo('DNN_Predict:  %s' % (dnn_predict_str))
        correct = [y for i,y in enumerate(dnnPredict_batch) if true_values[i] == int(y)]
        acc = len(correct) / len(true_values)
        util.thisLogger.logInfo('DNN_Predict Acc: %s' % (acc))

        if self.use_clustering:
            self.clus_lock.acquire()
            ydata = self.clusterer.predict_act(act_unseen_batch)
            self.clus_lock.release()
            util.thisLogger.logInfo('ClusA Predict:%s' % (''.join([str(x) for x in ydata])))
            correct = [y for i, y in enumerate(ydata) if true_values[i] == int(y)]
            acc = len(correct) / len(true_values)
            util.thisLogger.logInfo('Clus A Acc: %s' % (acc))

        if self.use_anomaly_detection:
            levels = self.hstrees.eval(act_unseen_batch, true_values)
            util.thisLogger.logInfo('HSTree lev:%s' % (''.join([str(x) for x in levels])))

        util.thisLogger.logInfo('')

        all_results = []
        majority_result = []
        all_classifier_predicts = []
        drift_votes = 0
        for i, classifier in enumerate(self.classifiers):
            stream_classifier_predict_str = ''
            drift_input_str = ''
            drift_output_str = ''
            all_classifier_predicts.append(classifier_predictions[i])
            for a, d, c, t in zip(act_unseen_batch, dnnPredict_batch, classifier_predictions[i], true_values):
                stream_classifier_predict_str += str(int(c))
                drift_input, drift_result = self.get_drift(d, c, self.drift_detectors[i])
                drift_input_str += drift_input
                drift_output_str += drift_result
            util.thisLogger.logInfo('CLAS_Predict  %s:  %s' % (i, stream_classifier_predict_str))
            util.thisLogger.logInfo('DRIFT_Input   %s:  %s' % (i, drift_input_str))
            util.thisLogger.logInfo('DRIFT_Output  %s:  %s' % (i, drift_output_str))

            classifier_discrepancies = drift.convertDriftToDiscrepancies(list(drift_output_str), true_discrep)
            all_results.append(classifier_discrepancies)

            # accumulate the discrepancies from each streaming classifier for the result
            if len(majority_result) == 0:
                majority_result = classifier_discrepancies
            else:
                for d, discrep in enumerate(classifier_discrepancies):
                    if discrep == 'D':
                        majority_result[d] = 'D'

            if 'D' in classifier_discrepancies:
                drift_votes += 1
                drift_detected = True

        classifier_predictions.clear()

        self.is_adapted = False
        apply_drift = True
        # if apply_drift:

        # used for investigations ----------------------------------------------------------
        if self.use_artificial_drift_detection:
            if 'CE' in true_discrep or 'CD' in true_discrep:
                self.drift_triggered = True
                # self.drift_trigger_count += 1
                if self.drift_trigger_count == 0:
                    self.drift_triggered_window = self.window_count

            if self.drift_triggered and self.window_count % self.adapt_every_n_windows == 0:
                # store true data
                self.concatenate_buffer_data(act_unseen_batch, xdata_unseen_batch, true_values)

            if self.drift_triggered and self.window_count%self.adapt_every_n_windows == 0:
                # if drift has occurred, adapt every n windows
                drift_votes = 1
            else:
                drift_votes = 0
        #-------------------------------------------------------------------------------------

        if drift_votes > 0:
            self.drift_trigger_count += 1

            self.data_lock.acquire()
            self.concatenate_buffer_data(act_unseen_batch, xdata_unseen_batch, true_values)

            util.thisLogger.logInfo('adapt set to true')

            if self.use_clustering:
                self.clus_lock.acquire()
                # self.clusterer.partial_fit_act(act_unseen_batch, true_values)
                self.clusterer.partial_fit(xdata_unseen_batch, act_unseen_batch, true_values)
                self.clus_lock.release()

            self.data_lock.release()

            # adapt lock will be taken if doing adaption sequentially
            sequential_adapt = False
            if sequential_adapt or (not sequential_adapt and self.drift_trigger_count == 1):
                self.adapt_dnn()
                # predict unseen instances via the DNN
                if self.act_vote:
                    self.adaptive_dnn.add_activations(act_unseen_batch)
                dnnPredict_batch = analyse.getPredictions(self.adaptive_dnn, xdata_unseen_batch, self.adaptive_dnn.all_classes)
                # if the batch contains old and new values, use the streaming classifier predictions
                for i, classifier in enumerate(self.classifiers):
                    y_classifier_batch = classifier.predict(act_unseen_batch)
                    classifier_predictions.append(y_classifier_batch)
                self.is_adapted = True

                if not sequential_adapt:
                    self.adapt = True
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
        adapt_class = []
        discrepancyType = util.getParameter('DataDiscrepancy')

        if self.is_adapted:
            for r, d, t, c in zip(majority_result, dnnPredict_batch, true_values, classifier_predictions[0]):
                if t in util.getParameter('DataClasses'):
                    # original class
                    adapt_class.append(c)
                else:
                    adapt_class.append(d)
                if r == 'D':
                    adapt_discrepancy.append(discrepancyType)
                if r == 'N':
                    adapt_discrepancy.append('ND')
        else:
            for r, x, a, d, t in zip(majority_result, xdata_unseen_batch, act_unseen_batch, dnnPredict_batch,
                                     true_values):
                adapt_class.append(d)
                if r == 'D':
                    adapt_discrepancy.append(discrepancyType)
                if r == 'N':
                    adapt_discrepancy.append('ND')

        y_classifier_batch = []
        for idx in range(len(all_classifier_predicts[0])):
            combined_classes = []
            for c in all_classifier_predicts:
                combined_classes.append(str(c[idx]))
            y_classifier_batch.append(';'.join(combined_classes))
            # y_classifier_batch.append(str(combined_classes[0]))

        # discrepancy results for each analysis classifier separately
        # using majority vote instead
        all_discrep_batch = []
        for idx in range(len(all_results[0])):
            combined_discreps = []
            for c in all_results:
                combined_discreps.append(str(c[idx]))
            all_discrep_batch.append(';'.join(combined_discreps))

        return majority_result, y_classifier_batch, adapt_discrepancy, adapt_class


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