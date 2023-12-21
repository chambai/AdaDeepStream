from skmultiflow.lazy import SAMKNNClassifier
import numpy as np

class SamKnn:
    def __init__(self):
        self.classifier_act = SAMKNNClassifier(n_neighbors=5,
                                                 weighting='distance', stm_size_option='maxACCApprox',
                                                 max_window_size=1000, use_ltm=True)
        self.data_buffers = {}


    class BufferData:
        def __init__(self, x_data, act_data, y_data):
            self.x_data = x_data
            self.act_data = act_data
            self.y_data = y_data

        def extend(self, x_data, act_data, y):
            self.x_data = np.vstack((self.x_data, x_data))
            self.act_data = np.vstack((self.act_data, act_data))
            self.y_data = np.hstack((self.y_data, y))

        def get_buffer_sample(self, n_samples):
            idxs = np.random.choice(np.arange(self.x_data.shape[0]), size=n_samples, replace=True)
            return self.x_data[idxs], self.act_data[idxs], np.take(self.y_data, idxs)


    def partial_fit_act(self, X, y):
        self.classifier_act.partial_fit(X, y)


    def partial_fit(self, X_data, X_act, y):
        # stores x data and activation data in class buffers and fits to clusterer
        self.classifier_act.partial_fit(X_act, y)
        self.__partial_fit_buffers(X_data, X_act, y)


    def __partial_fit_buffers(self, X_data, X_act, y):
        # add known instances
        y_key = np.array(y).astype(int)
        for k, v in self.data_buffers.items():
            idxs = np.where(y_key == k)[0]
            v.extend(X_data[idxs], X_act[idxs], np.take(y, idxs))

        # add new classes
        for n in np.unique(y_key):
            if n not in self.data_buffers.keys():
                idxs = np.where(y_key == n)[0]
                self.data_buffers[n] = self.BufferData(X_data[idxs], X_act[idxs], np.take(y, idxs))
                # maybe augment new data and add it here


    def get_buffer_data(self, y, n_samples):
        # returns a selection of original x data and the corresponding activation data based on the y values provided
        x_data = []
        act_data = []
        y_data = []
        y_key = np.array(y).astype(int)
        # print('true y values: %s'%(np.unique(y_key)))
        for c, v in self.data_buffers.items():
            if c not in y_key:
                n_samples = 20
            else:
                n_samples = 100
                x, a, y = self.data_buffers[c].get_buffer_sample(n_samples)
                x_data.extend(x)
                act_data.extend(a)
                y_data.extend(y)

        return x_data, act_data, y_data

    def predict_act(self, X):
        predicts = self.classifier_act.predict(X)
        predicts = predicts.astype(int)
        return predicts


class SamKnn1:
    def __init__(self):
        self.classifier_act = SAMKNNClassifier(n_neighbors=5,
                                                 weighting='distance', stm_size_option='maxACCApprox',
                                                 max_window_size=1000, use_ltm=True)


    def fit(self, X, y):
        self.classifier_act.partial_fit(X, y)

    def partial_fit(self, X, y):
        self.classifier_act.partial_fit(X, y)

    def predict(self, X):
        predicts = self.classifier_act.predict(X)
        predicts = predicts.astype(int)
        return predicts