import util
import numpy as np

class DsExternalInterface:
    def __init__(self):
       self.class_map = {}


    def setup_class_map(self, y_data):
        unique_classes = np.sort(np.unique(y_data))
        for base_class, zero_index_class in zip(unique_classes, np.arange(len(unique_classes))):
            self.class_map[base_class] = zero_index_class


    def transform_input_data(self, X_data, y_data=None, channels_first=False, y_data_2d=False):
        if y_data is None:
            y = y_data
        else:
            # add new classes in order as they need to be translated correctly into zero index classes
            for label in np.unique(y_data):
                if label not in self.class_map:
                    self.class_map[label] = max(self.class_map.values()) + 1
            y = util.transformDataIntoZeroIndexClasses(y_data, list(self.class_map.keys()), list(self.class_map.values()))
            if y_data_2d:
                y = np.reshape(y, (-1, 1))

        if channels_first:
            # move channels as that is what pytorch requires
            X = np.reshape(X_data, (X_data.shape[0], X_data.shape[3], X_data.shape[2], X_data.shape[1]))
        else:
            X = X_data

        return X, y

    def transform_ouput_data(self, y_data):
        y = util.transformZeroIndexDataIntoClasses(y_data, list(self.class_map.keys()))
        return y