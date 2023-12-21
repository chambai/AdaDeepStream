# base class for DNN wrapper classes
class DNN:  # Wrapper for DNNs from different libraries so the calling code has the same interface no matter if it is a Keras or torch DNN
    def __init__(self):
        self.model = None

    def get_class_name(self):
        return type(self).__name__

    def load(self, model_file=''):
        pass

    def load_model(self, model):
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

    def get_adaption_variables(self, adaption_strategies):
        do_partial_fit = True
        always_update = False
        add_adaption_layers = False
        limit_adaption_layers = False
        reduce_lr_percent = 0
        nearest_class_mean = False
        scr = False
        gdumb_rv = False
        act_vote = False
        lin = False
        for adaption_strategy in adaption_strategies:
            if adaption_strategy == 'none':
                do_partial_fit = False
            elif adaption_strategy == 'always_update':
                always_update = True
            elif adaption_strategy == 'add_adaption_layers':
                add_adaption_layers = True
            elif adaption_strategy == 'limit_adaption_layers':
                limit_adaption_layers = True
            elif adaption_strategy == 'reduce_lr_percent':
                reduce_lr_percent = 10  # reduce learning rate by 10 percent
            elif adaption_strategy == 'nearest_class_mean':
                nearest_class_mean = True  # use nearest class mean instead of softmax
            elif adaption_strategy == 'scr':
                scr = True
            elif adaption_strategy == 'gdumb_rv':
                gdumb_rv = True
            elif adaption_strategy == 'act_vote':
                act_vote = True
            elif adaption_strategy == 'lin':
                lin = True
            else:
                raise Exception('unhandled adaption_strategy of %s' % (adaption_strategy))

        return do_partial_fit, always_update, add_adaption_layers, limit_adaption_layers, reduce_lr_percent, nearest_class_mean, scr, gdumb_rv, act_vote, lin





