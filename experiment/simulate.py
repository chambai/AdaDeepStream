import util
import numpy as np

# simulates drift detection and returns true values when drift is detected
class Simulator():
    def __init__(self):
        self.is_setup = False
        self.sim_data = []
        self.unseen_batches = []

    def setup(self, sim_data):
        self.sim_data = sim_data
        # split unseen instance object list into batches
        batch_length = util.getParameter('StreamBatchLength')
        self.unseen_batches = [self.sim_data[i:i + batch_length] for i in
                               range(0, len(self.sim_data), batch_length)]
        self.is_setup = True


    def is_simulate_drift_detection(self):
        return util.getParameter('SimulateDriftDetection')


    def simulate_drift_detection(self, batch_index):
        drift_result = []
        true_discrep = np.array([u.discrepancyName for u in self.unseen_batches[batch_index - 1]])
        for td in true_discrep:
            if td != 'ND':
                drift_result.append('D')
            else:
                drift_result.append('N')
        dnn_predicts = [u.correctResult for u in self.unseen_batches[batch_index - 1]]
        sc_predicts = dnn_predicts
        return dnn_predicts, sc_predicts, drift_result


    def get_true_classes(self, X, batch_index):
        # in real-world app, replace with i.e. user interface to allow entry of true values
        true_classes = np.array([u.correctResult for u in self.unseen_batches[batch_index - 1]])
        true_values_str = ''
        for t in true_classes:
            true_values_str += str(int(t))
        # util.thisLogger.logInfo('True_Values:   %s' % (true_values_str))
        return true_classes