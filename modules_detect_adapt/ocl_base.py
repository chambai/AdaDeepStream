from modules_detect_adapt.ads_base import Analysis
import util
import torch
from framework.dnn_models import DNNPytorch
from framework import dataset


class AnalysisOcl(Analysis):
    def __init__(self):
        super(AnalysisOcl, self).__init__()

    def get_training_data(self):
        # get original training data
        mapOriginalYValues = util.getParameter('MapOriginalYValues')
        if len(mapOriginalYValues) == 0:
            x_train_orig, y_train_orig, _, _ = dataset.getFilteredData(isMap=False, do_normalize=False)
        else:
            x_train_orig, y_train_orig, _, _ = dataset.getFilteredData(isMap=True, do_normalize=False)
        return x_train_orig, y_train_orig

    def set_dnn_output_features(self, model, num_classes):
        # adjust model so it will work in the OCI world - set the output layer to the total number of classes expected
        num_features = model.classifier[-1].in_features
        # delete original output layer
        del model.classifier[-1]
        # re-add final output layer
        num_class_modules = len([m for m in model.classifier.modules()])
        model.classifier.add_module(str(num_class_modules), torch.nn.Linear(num_features, num_classes))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return model



    def get_optimizer(self, model, model_name, lr=None, wd=None):
        # get optimizer
        dnn_pt = DNNPytorch()
        optimizer = dnn_pt.get_optimizer(model, model_name, lr=lr, wd=wd)
        return optimizer

    def get_optimizer_params(self):
        dnn_pt = DNNPytorch()
        lr, wd = dnn_pt.get_optimizer_params(util.getParameter('Dnn'), is_ocl=True)
        return lr, wd


    def setup_adaptive_dnn(self, adaption_strategies, x_train_orig, y_train_orig):
        super.setup_adaptive_dnn()
        self.adaptive_dnn.fit(x_train_orig, y_train_orig)  # applies extra training if required



