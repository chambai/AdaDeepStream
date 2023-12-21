from experiment.run_interface import Run
import numpy as np
import os
import util
from framework import train, extract, analyse
import datetime


class DeepStreamFramework():
  def __init__(self):
    self.data_dir = ''
    self.run_data_file = ''

class DeepStreamBase(Run, DeepStreamFramework):
  def __init__(self, dnn, reduction_name, evaluator, simulator):
    super().__init__()
    self.dnn = dnn
    self.reduction_name = reduction_name
    self.max_values = []
    self.evaluator = evaluator
    self.simulator = simulator
    self.batch_index = 1
    self.is_adapt = False


  def load_modules(self):
    # was not in main. Must be loaded somewhere else. Does this automatically run when a sub class is called?
    pass

  def get_activations(self, X, y):
    # get activations
    if os.path.exists(self.run_data_file):
      # get from file
      data = np.load(self.run_data_file, allow_pickle=True)
      activations = data['a']
      xData = data['x']
      yData = data['y']
    else:
      # extract from DNN and save
      activations, xData, yData = train.getActivations(X, -1, self.dnn, y)
      np.savez(self.run_data_file, a=activations, x=xData, y=yData, allow_pickle=True)

    # normalize
    activations, self.max_values = util.normalizeFlatValues(activations, True)
    return xData, yData, activations

  def fit(self, X, y):
    X, y, activations = self.get_activations(X, y)
    self.load_modules()
    analyse.module.setup(self.dnn, activations, y, X)

  def partial_fit(self, X):
    util.thisLogger.logInfo('Processing window: %s' % (self.batch_index))
    window_start_time = datetime.datetime.now()

    dnn_predicts = None
    if not self.is_adapt:
      # only predict here if it is not adapting as the new classes have to be taken into account in the module
      dnn_predicts = analyse.getPredictions(self.dnn, X)

    acts = extract.getActivations(self.dnn, X)
    acts, _ = util.normalizeFlatValues(acts, False, self.max_values)

    # drift detection
    if self.simulator is not None and self.simulator.is_simulate_drift_detection():
      dnn_predicts, sc_predicts, drift_result = self.simulator.simulate_drift_detection(self.batch_index)
    else:
      dnn_predicts, sc_predicts, drift_result = analyse.module.detect(X, acts, dnn_predicts)

    window_end_time = datetime.datetime.now()
    self.evaluator.drift_det_window_times = np.append(self.evaluator.drift_det_window_times,
                                                 round((window_end_time - window_start_time).total_seconds() * 1000))

    if 'D' in drift_result:
      util.thisLogger.logInfo('Drift detected')

    if not self.is_adapt and self.evaluator is not None:
      self.evaluator.add_stream_results(acts, dnn_predicts, drift_result)  # add results, ready for evaluation

    self.batch_index += 1
    return acts, dnn_predicts, sc_predicts, drift_result

  def stop(self):
    if hasattr(analyse.module, 'stopProcessing'):
        analyse.module.stopProcessing()

    # evaluate
    if self.evaluator is not None:
      self.evaluator.adapt_times = analyse.module.adapt_times
      self.evaluator.evaluate_stream()
      self.evaluator.evaluate_timings()






class DeepStreamAdaptBase(DeepStreamBase):
  def __init__(self, dnn, lr, reduction_name, adaptation_name, evaluator, simulator):
    super().__init__(dnn, reduction_name, evaluator, simulator)
    self.lr = lr
    self.adaptation_name = adaptation_name
    self.is_adapt = True

  def load_modules(self):
    super().load_modules() # load reduction module. Is this needed?
    analyse.loadModule('modules_detect_adapt.' + self.adaptation_name, self.data_dir)


  def partial_fit(self, X):
    acts, dnn_predicts, sc_predicts, drift_result = super().partial_fit(X)   # detect drift

    # if drift is detected, get true labels for this window
    true_classes = None
    if 'D' in drift_result:
        true_classes = self.simulator.get_true_classes(X, self.batch_index-1)

    adapt_result = analyse.module.adapt(X, acts, dnn_predicts, drift_result, sc_predicts, true_classes)

    if self.evaluator is not None:
      self.evaluator.add_stream_results(acts, dnn_predicts, adapt_result)  # add results, ready for evaluation

    return acts, dnn_predicts, sc_predicts, adapt_result


  def stop(self):
    self.evaluator.adapt_times = analyse.module.adapt_times
    super().stop()



class DeepStreamOs(DeepStreamBase):
  def __init__(self, dnn, reduction_name, evaluator=None, simulator=None):
    super().__init__(dnn, reduction_name, evaluator, simulator)



class AdaDeepStream(DeepStreamAdaptBase):
  def __init__(self, dnn, lr, reduction_name, adaptation_name, evaluator=None, simulator=None):
    super().__init__(dnn, lr, reduction_name, adaptation_name, evaluator, simulator)



class DeepStreamEnsemble(DeepStreamAdaptBase):
  def __init__(self, dnn, lr, reduction_name, adaptation_name, evaluator=None, simulator=None):
    super().__init__(dnn, lr, reduction_name, adaptation_name, evaluator, simulator)




class DeepStreamCompare(Run):
  # for comparison modules (they don't use activations), i.e. RSB, CPE, TENT
  def __init__(self, dnn, adaptation_name, evaluator, simulator):
    self.evaluator = evaluator
    self.simulator = simulator
    self.dnn = dnn
    self.adaptation_name = adaptation_name
    self.batch_index = 1

  def load_modules(self):
      analyse.loadModule('modules_compare.' + self.adaptation_name)

  def fit(self, X, y):
    self.load_modules()
    analyse.module.setup(self.dnn, X, y)

  def partial_fit(self, X):
    window_start_time = datetime.datetime.now()
    true_classes = self.simulator.get_true_classes(X, self.batch_index)
    adapt_result = analyse.module.adapt(X, true_classes)
    dnn_predicts = analyse.getPredictions(self.dnn, X)
    acts = [0 for i in dnn_predicts]  # simulate activations
    sc_predicts = dnn_predicts  # simulate streaming classifier results

    window_end_time = datetime.datetime.now()
    self.evaluator.drift_det_window_times = np.append(self.evaluator.drift_det_window_times,
                                                      round(
                                                        (window_end_time - window_start_time).total_seconds() * 1000))

    if self.evaluator is not None:
      self.evaluator.add_stream_results(acts, dnn_predicts, adapt_result)  # add results, ready for evaluation

    self.batch_index += 1
    return acts, dnn_predicts, sc_predicts, adapt_result

  def stop(self):
    if hasattr(analyse.module, 'stop'):
      analyse.module.stop()

    # evaluate
    if self.evaluator is not None:
      self.evaluator.evaluate_stream()
      self.evaluator.evaluate_timings()