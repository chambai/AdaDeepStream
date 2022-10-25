class DNN:
    def __init__(self):
        self.model = None

    def get_class_name(self):
        return type(self).__name__

    def load(self, model_file=''):
        pass

    def summary(self):
        pass

    def train(self, dataset_name, classes, in_x_train, in_y_train, in_x_test, in_y_test, dataMap='', raise_accuracy_exception=False, batch_size = 32, model=None, epochs = None, update_for_transfer_learning=True, save_model=True, test_model=True, early_stopping_patience=4, reduce_lr_percent=0):
        pass

    def predict(self, X):
        pass

    def get_num_layers(self):
        pass

    def get_layer_output(self, layer_num, X):
        pass

    def get_layer_name(self, layer_num, X):
        pass





