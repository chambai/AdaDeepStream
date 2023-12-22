from torch.utils.data import Dataset
# from external.ocl.models.resnet import Reduced_ResNet18
from framework import pretrain_dnn, dataset
from framework.dnn_wrapper import DNN
from torch.utils.data import TensorDataset
import os
import numpy as np
import matplotlib.pyplot as plt
import util
import torch
import torch.optim as optim
from torchvision.models.mobilenet import mobilenet_v2
from torchvision.models.vgg import vgg16, vgg16_bn
from torch.optim.lr_scheduler import StepLR
from torch.nn import CrossEntropyLoss
import time
import uuid
from collections import Counter

# wrapper classes for DNNs to make them adaptive
class DNNPytorch(DNN):
    def __init__(self):
        DNN.__init__(self)
        self.device = None
        self.model = None

    def load(self, model_file, raise_load_exception=True):
        is_loaded = False
        try:
            torch.cuda.empty_cache()
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model = torch.load(model_file)
            self.model = self.model.to(self.device)
            is_loaded = True
        except Exception as e:
            if raise_load_exception:
                raise e

    def load_model(self, model):
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def print_summary(self):
        pass

    def pytorch_train2(self, model, device, train_loader, optimizer, epoch, dataset_name, loss_func):
        model.train()
        optimizer.zero_grad()

        for batch_idx, (data, act_data, target) in enumerate(train_loader):
            target = target.long()
            outputs = model(data.to(device))
            loss = loss_func(outputs, target.to(device))

        loss.backward()
        optimizer.step()
        return 0.0, 0.0

    def pytorch_train(self, model, device, train_loader, optimizer, epoch, dataset_name, loss_func):
        show_log = False
        correct = 0
        model.train()
        for batch_idx, (data, act_data, target) in enumerate(train_loader):
            target = target.type(torch.LongTensor)
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            # see if there is a get_activations method
            requires_activations = False
            op = getattr(model, "add_activations", None)
            if callable(op):
                requires_activations = True

            if requires_activations:
                model.add_activations(act_data)

            output = model(data)

            _, preds = torch.max(output, 1)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()
            if show_log:
                util.thisLogger.logInfo('Train Epoch: %s [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

        loss /= len(train_loader)
        epoch_acc = correct / len(train_loader.dataset)
        percent_acc = 100. * correct / len(train_loader.dataset)

        util.thisLogger.logInfo('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            loss, correct, len(train_loader.dataset), percent_acc))

        return loss.item(), epoch_acc

    def __pytorch_test(self, model, device, test_loader, dataset_name, loss_func):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, act_data, target in test_loader:
                target = target.type(torch.LongTensor)
                data, target = data.to(device), target.to(device)

                # see if there is a get_activations method
                requires_activations = False
                op = getattr(model, "add_activations", None)
                if callable(op):
                    requires_activations = True

                if requires_activations:
                    model.add_activations(act_data)

                output = model(data)
                _, preds = torch.max(output, 1)
                test_loss = loss_func(output, target)
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader)
        epoch_acc = correct / len(test_loader.dataset)
        percent_acc = 100. * correct / len(test_loader.dataset)

        util.thisLogger.logInfo('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            percent_acc))

        return test_loss.item(), epoch_acc, percent_acc

    def get_model(self, base_model_name):
        if base_model_name == 'mobilenet':
            model = mobilenet_v2(pretrained=True)  # mobilenet v1 is not available in pytorch
        elif base_model_name == 'vgg16':
            model = vgg16(pretrained=True)
        else:
            raise Exception('unhandled model name of %s ' % (base_model_name))
        return model

    def get_optimizer_params(self, base_model_name, reduce_lr_percent=0, is_ocl=False):
        if base_model_name == 'mobilenet':
            lr = 0.0001
            if is_ocl:
                lr = 0.01  # this value works best with OCL
            wd = 0
            if reduce_lr_percent > 0:
                lr = (lr / 100) * reduce_lr_percent  # reduce lr by percentage given
        elif base_model_name == 'vgg16' or base_model_name == 'vgg16bn':
            lr = 0.01
            wd = 0
        else:
            raise Exception('unhandled base model name of %s' % (base_model_name))
        return lr, wd

    def get_optimizer(self, model, base_model_name, reduce_lr_percent=0, lr=None, wd=None):
        init_lr, wd = self.get_optimizer_params(base_model_name)
        if base_model_name == 'mobilenet':
            if lr is None:
                lr = init_lr
            init_wd = 0
            if wd is None:
                wd = init_wd
            if reduce_lr_percent > 0:
                lr = (lr / 100) * reduce_lr_percent  # reduce lr by percentage given
            optimizer = optim.Adamax(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-07)
        elif base_model_name == 'vgg16' or base_model_name == 'vgg16bn':
            if lr is None:
                lr = init_lr
            init_wd = 0
            if wd is None:
                wd = init_wd
            if reduce_lr_percent > 0:
                lr = (lr / 100) * reduce_lr_percent  # reduce lr by percentage given
            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
        else:
            raise Exception('unhandled model name of %s ' % (base_model_name))
        return optimizer

    def train(self, dataset_name, classes, in_x_train, in_y_train, in_x_test, in_y_test, dataMap='',
              raise_accuracy_exception=False, batch_size=256, model=None, epochs=None,
              update_for_transfer_learning=True, save_model=True, test_model=True, early_stopping_patience=4,
              reduce_lr_percent=0, in_x_train_act=None):
        util.thisLogger.logInfo('early_stopping_patience set to %s' % (early_stopping_patience))
        torch.cuda.empty_cache()  # clear GPU memory

        # set parameters-------------------------------
        base_model_name = util.getParameter('Dnn')
        full_model_name = pretrain_dnn.getFullDnnModelPathAndName()
        if model is None:
            model = self.model
        if epochs is None:
            epochs = util.getParameter('DnnTrainEpochs')
        reduce_lr_gamma = 0.7
        kwargs = {'batch_size': batch_size}

        num_classes = len(classes)

        # make sure the arrays are numpy arrays as this makes the conversion to/from 0 and 1s work correctly
        y_train = np.array(in_y_train, dtype=np.int16)
        y_test = np.array(in_y_test, dtype=np.int16)
        x_train = np.array(in_x_train, dtype=np.float32)
        x_test = np.array(in_x_test, dtype=np.float32)
        in_x_test_act = None

        if in_x_train_act is None:
            torch_x_train_act = np.full((x_train.shape[0], 0), 0)
            torch_x_test_act = np.full((x_test.shape[0], 0), 0)
        else:
            x_train_act = np.array(in_x_train_act, dtype=np.float32)
            x_test_act = np.array(in_x_test_act, dtype=np.float32)
            torch_x_train_act = torch.from_numpy(x_train_act)
            torch_x_test_act = torch.from_numpy(x_test_act)

        # Map data if mappings specified
        if dataMap != '':
            classes = np.unique(in_y_test)
            num_classes = len(classes)
            sourceClasses = np.sort(np.unique(y_train))
            y_train = util.transformDataIntoZeroIndexClasses(y_train, sourceClasses)
            y_test = util.transformDataIntoZeroIndexClasses(y_test, sourceClasses)
        else:
            # convert training data to zero's and 1's as 'to categorical' needs this
            sourceClasses = np.sort(np.unique(y_test))
            targetClasses = np.arange(len(np.unique(y_test)))
            y_train = util.transformDataIntoZeroIndexClasses(y_train, sourceClasses, targetClasses)
            y_test = util.transformDataIntoZeroIndexClasses(y_test, sourceClasses, targetClasses)
            pass

        util.thisLogger.logInfo('model will be stored in %s' % (full_model_name))
        util.thisLogger.logInfo('model training for %s epochs' % (epochs))

        # model starts here--------------
        # set GPU
        util.thisLogger.logInfo('Is GPU available to torch? %s' % (torch.cuda.is_available()))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            kwargs.update({'num_workers': 4, 'pin_memory': True})
        util.thisLogger.logInfo(device)

        torch_x_train = torch.from_numpy(x_train)
        torch_x_train_perm = torch_x_train.permute(0, 3, 1, 2)
        torch_x_test = torch.from_numpy(x_test)
        torch_x_test_perm = torch_x_test.permute(0, 3, 1, 2)

        train_loader = torch.utils.data.DataLoader(
            TensorDataset(torch.Tensor(torch_x_train_perm), torch.Tensor(torch_x_train_act), torch.Tensor(y_train)),
            shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            TensorDataset(torch.Tensor(torch_x_test_perm), torch.Tensor(torch_x_test_act), torch.Tensor(y_test)),
            shuffle=False, **kwargs)
        util.thisLogger.logInfo('Number of train batches: {} Number of test batches: {}'.format(len(train_loader), len(test_loader)))

        # Note: we do not need to specify a softmax layer as the softmax is integrated with the nn.CrossEntropyLoss function
        model = self.get_model(base_model_name)
        optimizer = self.get_optimizer(model, base_model_name)

        if update_for_transfer_learning:
            # add an extra convolutional layer at the begining so we don't have to increase the size of the image
            num_input_channels = torch_x_train_perm.shape[1]
            if num_input_channels != 3:
                first_conv_layer = [
                    torch.nn.Conv2d(num_input_channels, 3, kernel_size=3, stride=1, padding=1, dilation=1,
                                    groups=num_input_channels, bias=True)]
                first_conv_layer.extend(list(model.features))
                model.features = torch.nn.Sequential(*first_conv_layer)
                # disable grad for pre-trained layers only (train the new first conv layer and the classifier layers
                for param in model.features[1:].parameters():
                    param.requires_grad = False

            # add an extra linear layer to get the output classes from 1000 to 10
            model.classifier[-1] = torch.nn.Linear(in_features=model.classifier[-1].in_features,
                                                   out_features=num_classes)

        model = model.to(device)

        early_stopping_patience = early_stopping_patience  # stop if the same test acc value is encountered consecutively
        history = {}
        history['accuracy'] = []
        history['val_accuracy'] = []
        history['loss'] = []
        history['val_loss'] = []
        losses = []
        accuracies = []
        last_percent_acc_value = 0
        last_percent_acc_count = 1
        loss_func = CrossEntropyLoss()
        scheduler = StepLR(optimizer, step_size=1, gamma=reduce_lr_gamma)
        for epoch in range(1, epochs + 1):
            train_epoch_loss, train_epoch_acc = self.pytorch_train(model, device, train_loader, optimizer, epoch,
                                                                   dataset_name, loss_func)
            if test_model:
                test_epoch_loss, test_epoch_acc, percent_acc = self.__pytorch_test(model, device, test_loader,
                                                                                   dataset_name, loss_func)
            else:
                test_epoch_loss = 0
                test_epoch_acc = 0
                percent_acc = 0

            scheduler.step()

            percent_acc = int(percent_acc)
            losses.append(train_epoch_loss)
            losses.append(test_epoch_loss)
            accuracies.append(train_epoch_acc)
            accuracies.append(test_epoch_acc)
            history['accuracy'].append(train_epoch_acc)
            history['loss'].append(train_epoch_loss)
            history['val_accuracy'].append(test_epoch_acc)
            history['val_loss'].append(test_epoch_loss)

            if dataset_name == 'cifar100':
                use_early_stopping = True
            else:
                use_early_stopping = True

            if use_early_stopping:
                # if it says at same value for the number of early_stopping_patience, end training
                if last_percent_acc_value == percent_acc:
                    last_percent_acc_count += 1
                else:
                    last_percent_acc_value = percent_acc
                    last_percent_acc_count = 1
                util.thisLogger.logInfo('accuracy percentage: %s, count: %s' % (percent_acc, last_percent_acc_count))
                if last_percent_acc_count == early_stopping_patience:
                    util.thisLogger.logInfo(
                        'training early stopped at %s test accuracy as there were %s consecutive values' % (
                        percent_acc, early_stopping_patience))
                    break;

        scores_train = [losses[0], accuracies[0]]
        scores_test = [losses[1], accuracies[1]]

        if save_model:
            # save full model
            torch.save(model, full_model_name)

            # save features checkpoints
            torch.save({
                # 'epoch': EPOCH,
                'model_state_dict': model.features.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_func,
            }, full_model_name.replace('.plt', '_features.ckpt'))

            # save classifier checkpoints
            torch.save({
                # 'epoch': EPOCH,
                'model_state_dict': model.classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_func,
            }, full_model_name.replace('.plt', '_classifier.ckpt'))

            # --------------------------------------------
            stats = 'DnnTrainLoss=%s, DnnTrainAccuracy=%s' % (scores_train[0], scores_train[1])
            stats = stats + ', DnnTestLoss=%s, DnnTestAccuracy=%s' % (scores_test[0], scores_test[1])

            plt.interactive(False)

            # Plot training & validation accuracy values
            plt.plot(history['accuracy'])
            plt.plot(history['val_accuracy'])
            plt.title('Model accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
            # plt.show()   # uncomment to display plot in window

            # Plot training & validation loss values
            plt.plot(history['loss'])
            plt.plot(history['val_loss'])
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
            # plt.show()   # uncomment to display plot in window

            # Save model and weights
            util.thisLogger.logInfo('Saved trained model at %s ' % (full_model_name))
            plt.savefig(full_model_name + '.png')
            plt.close()
        else:
            stats = 'DnnTrainLoss=%s\nDnnTrainAccuracy=%s' % ('0', '0')
            stats = stats + '\nDnnTestLoss=%s\nDnnTestAccuracy=%s' % ('0', '0')

        self.model = model
        self.device = device

        torch.cuda.empty_cache()  # clear GPU memory
        return stats

    def predict(self, X):
        result_np = []
        self.model = self.model.to(self.device)
        self.model.eval()
        num_chunks = X.shape[0] // 128
        x = torch.Tensor(X).to(self.device)
        if len(X.shape) == 4:
            x = x.permute(0, 3, 1, 2).to(self.device)
        #  split into chunks otherwise may get CUDA memory issues
        pred = torch.empty((0)).to(self.device)
        if num_chunks > 0:
            chunks = torch.chunk(x, num_chunks)
            for chunk in chunks:
                pred_chunk = self.model(chunk)
                pred_chunk = pred_chunk.argmax(dim=1, keepdim=True)
                pred = torch.cat((pred, pred_chunk))
                del chunk
                del pred_chunk
            del chunks
            torch.cuda.empty_cache()
        else:
            pred_chunk = self.model(x)
            pred_chunk = pred_chunk.argmax(dim=1, keepdim=True)
            pred = torch.cat((pred, pred_chunk))

        pred_np = pred.cpu().detach().numpy()
        result_np.append(pred_np)
        result_np = np.array(result_np)
        result_np = np.reshape(result_np, (-1))

        # if util.getParameter('DataDiscrepancy') == 'CD':
        if 'ocl' not in util.getParameter('AnalysisModule'):
            classes = np.sort(np.unique(util.getParameter('DataClasses')))
            if util.has_sub_classes():
                super_classes, _ = util.map_classes_y(classes)  # map sub classes to super classes
                classes = np.unique(super_classes)
            result_np = util.transformZeroIndexDataIntoClasses(result_np, classes)

        result_np = result_np.astype(int)
        return result_np

    def get_num_layers(self):
        layers = self.get_layer_names()
        return len(layers)

    def get_layer_output(self, layer_num, X, return_as_tensor=False):
        # define helper function
        def get_features(name):
            def hook(model, input, output):
                features[name] = output.detach()

            return hook

        layers = [module[1] for module in self.model.named_modules() if not isinstance(module, torch.nn.Sequential)]
        layer = layers[layer_num]
        handle = layer.register_forward_hook(get_features('feats'))

        x = X
        if torch.is_tensor(X):
            if len(X.shape) == 4:
                x = X.permute(0, 3, 1, 2)
        else:
            if len(X.shape) == 4:
                x = torch.Tensor(X).permute(0, 3, 1, 2).to(self.device)

        # activation extraction
        features = {}
        # forward pass [with feature extraction]
        preds = self.model(x)
        # move predictions to cpu
        preds.detach().cpu().numpy()
        # because of the forward pass and registered hook, features (activations)
        # will appear in the features dictionary
        acts = features['feats']
        if not return_as_tensor:
            acts = acts.detach().cpu().numpy()
            if len(acts.shape) == 4:
                acts = np.reshape(acts, (acts.shape[0], acts.shape[3], acts.shape[2], acts.shape[1]))
        handle.remove()
        return acts

    def get_layer_name(self, layer_num):
        layer_name = self.get_layer_names()[layer_num]
        return layer_name

    def get_layer_names(self):
        layer_names = ['%s.%s' % (module[0], type(module[1]).__name__) for module in self.model.named_modules() if
                       not isinstance(module, torch.nn.Sequential)]
        return layer_names

#-------------------------------------------------------------------------------------------------------------------
class DNNPytorchAdaptive(DNNPytorch):
    def __init__(self, do_partial_fit=True, adaption_strategies=[], buffer_type='none', freeze_layer_type='classification', epochs=3):
        DNNPytorch.__init__(self)
        self.adaption_strategies = adaption_strategies
        adaption_vars = self.get_adaption_variables(adaption_strategies)
        self.do_partial_fit = adaption_vars[0]
        self.always_update = adaption_vars[1]
        self.add_adaption_layers = adaption_vars[2]
        self.limit_adaption_layers = adaption_vars[3]
        self.reduce_lr_percent = adaption_vars[4]
        self.nearest_class_mean = adaption_vars[5]
        self.gdumb_rv = adaption_vars[7]
        self.act_vote = adaption_vars[8]
        self.lin = adaption_vars[9]
        self.freeze_layer_type = freeze_layer_type
        self.epochs = epochs

        self.all_classes = None
        self.class_map = None
        self.new_x_train = np.array([0])
        self.new_x_test = np.array([0])
        self.new_y_train = np.array([0])
        self.new_y_test = np.array([0])
        self.num_added_layers = 0
        self.adaption_layers_added = False
        self.mgr = None
        self.activations = None
        self.act_models = []
        self.new_classes = []
        self.class_buffers = None
        self.buffer_type = buffer_type  # class, reactive_subspace, none
        self.start_epoch = 0  # start from epoch 0 or last checkpoint epoch
        self.criterion = torch.nn.CrossEntropyLoss()
        pass

    def fit(self, X, y):
        # todo: check if this is used or not
        self.all_classes = np.sort(np.unique(y)).tolist()
        self.class_map = {}
        for i, c in enumerate(self.all_classes):
            self.class_map[c] = i

    def add_buffer_data(self, X, y):
        if self.buffer_type == 'class':
            self.class_buffers.add_no_weights(X, y)
            self.partial_fit(X,y)
        elif self.buffer_type == 'reactive_subspace':
            self.class_buffers.add(np.moveaxis(X, -1, 1), y, np.ones(len(y)))
            self.partial_fit(X, y)
        elif 'none':
            pass

    def get_copy(self):
        temp_model_name = 'temp_model_%s_%s.plt' % ('DNNPytorchAdaptive', uuid.uuid4())
        torch.save(self.model, temp_model_name)
        copy = DNNPytorchAdaptive(do_partial_fit=True, adaption_strategies=self.adaption_strategies, freeze_layer_type=self.freeze_layer_type, epochs=self.epochs)
        copy.load(temp_model_name)
        copy.all_classes = self.all_classes
        copy.class_map = self.class_map
        copy.new_x_train = self.new_x_train
        copy.new_x_test = self.new_x_test
        copy.new_y_train = self.new_y_train
        copy.new_y_test = self.new_y_test
        copy.num_added_layers = self.num_added_layers
        copy.adaption_layers_added = self.adaption_layers_added
        copy.mgr = self.mgr
        copy.activations = self.activations
        copy.act_models = self.act_models
        copy.new_classes = self.new_classes
        copy.class_buffers = self.class_buffers
        copy.buffer_type = self.buffer_type
        self.start_epoch = self.start_epoch
        self.criterion = self.criterion
        os.remove(temp_model_name)
        return copy

    # @jit(nopython=False)
    def partial_fit(self, x, y, adapt_blocks=[]):

        if self.do_partial_fit:
            previous_classes = self.all_classes.copy()
            previous_num_classes = len(self.all_classes)
            # add any additional classes
            new_classes = []
            [new_classes.append(c) for c in np.unique(y) if c not in self.all_classes]
            new_classes = np.array(new_classes)
            class_count = max(self.class_map.values())
            for i, c in enumerate(new_classes):
                self.class_map[c] = i + 1 + class_count

            [self.all_classes.append(c) for c in np.unique(y) if c not in self.all_classes]
            new_num_classes = len(self.all_classes)

            if self.always_update:
                self.model = self.__adapt(self.model, previous_classes, self.class_map, x, y, adapt_blocks=adapt_blocks)
            else:
                # only call if new classes detected
                if new_num_classes > previous_num_classes:
                    self.model = self.__adapt(self.model, previous_classes, self.class_map, x, y, adapt_blocks=adapt_blocks)

    def __adapt(self, model, previous_classes, class_map, x_train, y_train, adapt_blocks=[]):
        data_map = dataset.getDataMapAsString()
        start = time.time()
        y_train = np.array(y_train, dtype=np.int16)
        x_test = x_train
        y_test = y_train

        dnn_name = util.getParameter('Dnn')
        discrep = util.getParameter('DataDiscrepancy')
        if 'vgg' in dnn_name:
            batch_size = 8
            if len(x_train) > 100:
                if discrep == 'CE':
                    epochs = self.epochs # 6 for CD, 3 for CE
                if discrep == 'CD':
                    if 'cifar100' in util.getParameter('DatasetName'):
                        epochs = self.epochs
                    else:
                        epochs = self.epochs
            else:
                if discrep == 'CE':
                    epochs = self.epochs
                if discrep == 'CD':
                    epochs = self.epochs
        else:
            raise Exception('Number of epochs not set up for adaptation for %s'%(dnn_name))

        # util.thisLogger.logInfo('AdaptationEpochs=%s'%(epochs))
        num_classes = len(list(class_map.keys()))
        num_units = 8

        x_train = np.array(x_train, dtype=np.float32)
        x_test = np.array(x_test, dtype=np.float32)

        self.new_x_train = x_train
        self.new_x_test = x_test
        self.new_y_train = y_train
        self.new_y_test = y_test


        # if there's a new class, freeze existing layers and add the new layer
        previous_num_classes = len(previous_classes)

        if self.num_added_layers == 3 and self.limit_adaption_layers:
            self.add_adaption_layers = False

        self.model = model
        dnn_name = util.getParameter('Dnn')
        if 'mobilenet' in dnn_name:
            if len(self.model.classifier) == 2:
                del self.model.classifier[-1]
                del self.model.classifier[-1]
                num_class_modules = len([m for m in self.model.classifier.modules()])
                if 'mobilenet' in util.getParameter('Dnn'):
                    self.model.classifier.add_module('0', torch.nn.Linear(1280, 1280))
                    self.model.classifier.add_module('1', torch.nn.ReLU(inplace=True))
                    self.model.classifier.add_module('2', torch.nn.Linear(1280, 1280))
                    self.model.classifier.add_module('3', torch.nn.ReLU(inplace=True))
                    self.num_class_modules = len([m for m in self.model.classifier.modules()])
                    self.model.classifier.add_module('4', torch.nn.Linear(1280, num_classes))

        if self.freeze_layer_type == 'all':
            for param in self.model.parameters():
                param.requires_grad = True  # unfreeze all parameters
            for param in self.model.classifier.parameters():
                param.requires_grad = True  # unfreeze all classifier layers
        elif self.freeze_layer_type == 'classification':
            for param in self.model.parameters():
                param.requires_grad = False  # freeze all parameters
            for param in self.model.classifier.parameters():
                param.requires_grad = True  # unfreeze all classifier layers
        elif self.freeze_layer_type == 'block':
            # only unfreeze layers in the specified block
            block_indexes = self.get_block_indexes(adapt_blocks)
            if len(adapt_blocks) == 0:
                util.thisLogger.logInfo("WARNING: freeze_layer_type is 'block' and there are no adapt_blocks set. Adaptation will not occur")
            for param in self.model.parameters():
                param.requires_grad = False  # freeze all parameters
            for param in self.model.classifier.parameters():
                param.requires_grad = False  # freeze all classifier layers
            for i, param in enumerate(self.model.parameters()): # TODO: check this is right
                if i in block_indexes:
                    param.requires_grad = True  # unfreeze all classifier layers
        else:
            raise Exception('unhandled freeze_layer_type of %s'%(self.freeze_layer_type))


        if self.add_adaption_layers and num_classes > previous_num_classes:
            self.model = self.__get_new_model(model, num_units, num_classes)
        elif self.add_adaption_layers == False and not self.lin and num_classes > previous_num_classes:
            # do not add layers to the model
            self.model = self.get_new_model_final_layer(model, num_classes)
        elif self.add_adaption_layers == False and self.lin and num_classes > previous_num_classes:
            self.model = self.get_new_model_final_layer(model, num_classes)

        if self.lin:
            if self.buffer_type == 'class':
                sampled_x_train, sampled_y_train = self.class_buffers.sample_no_weights(x_train, y_train,)
                self.class_buffers.add_no_weights(x_train, y_train)
                x_train_ext = np.vstack((x_train, sampled_x_train))
                y_train_ext = np.hstack((y_train, sampled_y_train))
                self.model = self.do_train(self.model, self.device, self.get_train_loader(x_train_ext, y_train_ext, self.class_map), epochs=epochs)
                self.class_buffers.add_no_weights(x_train, y_train)
            elif self.buffer_type == 'none':
                self.model = self.do_train(self.model, self.device, self.get_train_loader(x_train, y_train, self.class_map), epochs=epochs)

        end = time.time()
        return self.model

    def get_block_indexes(self, block_idxs):
        indexes = []

        dnn = util.getParameter('Dnn')
        for block_idx in block_idxs:
            # get block mapping data
            full_filename = os.path.join(os.getcwd(), 'modules_extract', 'data',
                                         'cbir_blockmap_%s_blockcbir' % (dnn)) + '.csv'
            if os.path.isfile(full_filename):
                with open(full_filename, 'r') as f:
                    lines = f.readlines()

            layer_numbers = []
            for i,l in enumerate(lines[3:]): # start from the third line to correlate with the layer numbers in the model
                block_name = 'block' + str(block_idx + 1)   # name of the block in modules_extract\data\cbir_blockmap_vgg16_torch.csv
                if block_name in l:
                    layer_numbers.append(i)

            if len(layer_numbers) > 0:
                indexes.extend(np.arange(layer_numbers[0],layer_numbers[-1]+1))

        util.thisLogger.logInfo('Adaptation Block Indexes: %s'%(indexes))
        return indexes


    def __get_new_model(self, model, num_units, num_classes):
        # delete original output layer
        del model.classifier[-1]

        if self.add_adaption_layers and self.adaption_layers_added:
            num_class_modules = len([m for m in model.classifier.modules()])
            if 'mobilenet' in util.getParameter('Dnn'):
                model, num_class_modules = self.__add_layers_mnet(1, model, num_class_modules)
                self.adaption_layers_added = True
            elif 'vgg' in util.getParameter('Dnn'):
                model, num_class_modules = self.__add_layers_vgg(1, model, num_class_modules)
                self.adaption_layers_added = True

        if not self.adaption_layers_added:
            num_class_modules = len([m for m in model.classifier.modules()])
            if 'mobilenet' in util.getParameter('Dnn'):
                model, num_class_modules = self.__add_layers_mnet(2, model, num_class_modules)
                self.adaption_layers_added = True
            elif 'vgg' in util.getParameter('Dnn'):
                model, num_class_modules = self.__add_layers_vgg(2, model, num_class_modules)
                self.adaption_layers_added = True



        # re-add final output layer
        num_class_modules = len([m for m in model.classifier.modules()])
        num_class_modules += 1
        model.classifier.add_module(str(num_class_modules), torch.nn.Linear(512, num_classes))

        self.num_added_layers += 1
        return model

    def get_new_model_final_layer(self, model, num_classes):
        num_features = 0
        dnn_name = util.getParameter('Dnn')
        if 'mobilenet' in dnn_name:
            num_features = 1280
        elif 'vgg' in dnn_name:
            num_features = 4096
        else:
            raise Exception('unhandled Dnn of %s' % (dnn_name))

        # delete original output layer
        del model.classifier[-1]

        # re-add final output layer
        num_class_modules = len([m for m in model.classifier.modules()])
        num_class_modules += 1
        model.classifier.add_module(str(num_class_modules), torch.nn.Linear(num_features, num_classes))

        self.num_added_layers += 1
        return model

    def __add_layers_vgg(self, num_layers, model, num_class_modules):
        if num_layers == 2:
            model.classifier.add_module(str(num_class_modules), torch.nn.Linear(4096, 512))
            num_class_modules += 1
            model.classifier.add_module(str(num_class_modules), torch.nn.ReLU(inplace=True))
            num_class_modules += 1
        if num_layers == 1 or num_layers == 2:
            model.classifier.add_module(str(num_class_modules), torch.nn.Linear(512, 512))
            num_class_modules += 1
            model.classifier.add_module(str(num_class_modules), torch.nn.ReLU(inplace=True))
            if num_layers == 1:
                model.classifier[-2].weight = model.classifier[-4].weight
        return model, num_class_modules

    def __add_layers_mnet(self, num_layers, model, num_class_modules):
        model.classifier.add_module(str(num_class_modules), torch.nn.Linear(1280, 512))
        num_class_modules += 1
        model.classifier.add_module(str(num_class_modules), torch.nn.ReLU(inplace=True))
        num_class_modules += 1
        model.classifier.add_module(str(num_class_modules), torch.nn.Linear(512, 512))
        num_class_modules += 1
        model.classifier.add_module(str(num_class_modules), torch.nn.ReLU(inplace=True))
        return model, num_class_modules


    def __convert_zero_index_into_classes(self, y, class_map):
        for k, v in class_map.items():
            y[y == v] = k
        return y

    def add_activations(self, activations, y=None, act_model=None):
        if y is not None:
            [self.new_classes.append(c) for c in np.unique(y) if c not in self.all_classes]
            self.new_classes = self.new_classes

        self.activations = activations

        if act_model is not None:
            if len(self.act_models) == 2:
                del self.act_models[0]
            self.act_models.append(act_model)

    def get_train_loader(self, x_train, y_train, class_map):
        y_train = np.array(y_train, dtype=np.int16)
        x_train = np.array(x_train, dtype=np.float32)
        sourceClasses = list(class_map.keys())
        targetClasses = list(class_map.values())
        y_train = util.transformDataIntoZeroIndexClasses(y_train, sourceClasses, targetClasses)
        torch_x_train = torch.from_numpy(x_train)
        torch_x_train_perm = torch_x_train
        if len(x_train.shape) == 4:
            torch_x_train_perm = torch_x_train.permute(0, 3, 1, 2)
        kwargs = {'batch_size': len(x_train)}
        if torch.cuda.is_available():
            kwargs.update({'num_workers': 1, 'pin_memory': True})
        train_loader = torch.utils.data.DataLoader(
            TensorDataset(torch.Tensor(torch_x_train_perm), torch.Tensor(y_train)), shuffle=True, **kwargs)
        return train_loader


    def do_train(self, net, device, trainloader, epochs):
        net = net.to(device)
        dnn = util.getParameter('Dnn')
        # optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=LR, momentum=0.9, weight_decay=5e-4)
        if dnn == 'vgg16':
            lr = 0.01
            # lr = 0.1
            optimizer = optim.SGD(net.parameters(), lr=lr)
        elif dnn == 'mobilenet':
            lr = 0.0001
            optimizer = optim.Adamax(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-07)
        else:
            raise Exception('unhandled dnn of %s' % (dnn))

        acc_epoch_count = 0
        for epoch in range(self.start_epoch, self.start_epoch + epochs):

            if epoch % 50 == 0 and epoch > 50:
                lr = lr * 0.1
                if dnn == 'vgg16':
                    optimizer = optim.SGD(net.parameters(), lr=lr)
                elif dnn == 'mobilenet':
                    optimizer = optim.Adamax(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-07)
                else:
                    raise Exception('unhandled dnn of %s' % (dnn))
                util.thisLogger.logInfo('LR now is %s'%str(lr))

            # Perform 1 training epoch
            loss_epoch, acc_epoch = self.train(net, device, epoch, optimizer, trainloader)
            if acc_epoch == 100:
                acc_epoch_count += 1
            if acc_epoch_count == 3:
                break;

        return net

    def train(self, net, device, epoch, optimizer, trainloader):
        print_train_data = False
        util.thisLogger.logInfo('Epoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs = inputs.to(device)
            targets = targets.type(torch.LongTensor)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)

            # Forward predictionbs
            outputs = net.forward(inputs)
            prediction = net(inputs)

            # Convert labels to match the order seen by the classifier
            targets_converted = targets

            # Compute loss
            loss = self.criterion(outputs, targets_converted)

            # Backward + update
            loss.backward()

            optimizer.step()

            if print_train_data:
                # Printing stuff
                train_loss += loss.item()
                _, predicted = prediction.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets_converted).sum().item()
                if batch_idx % 200 == 0:
                    util.thisLogger.logInfo(batch_idx, len(trainloader), 'Training Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                 % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        if print_train_data:
            result = (train_loss / (batch_idx + 1)), 100. * correct / total
        else:
            result = 0.0, 0.0

        return result

    def predict_base(self, X):
        result_np = []
        try:
            self.model = self.model.to(self.device)
        except:
            util.thisLogger.logInfo('problem with model')
        self.model.eval()
        num_chunks = X.shape[0]//128
        x = torch.Tensor(X).to(self.device)
        if len(X.shape) == 4:
            x = x.permute(0, 3, 1, 2).to(self.device)
        #  split into chunks otherwise may get CUDA memory issues
        pred = torch.empty((0)).to(self.device)
        if num_chunks > 0:
            chunks = torch.chunk(x, num_chunks)
            for chunk in chunks:
                pred_chunk = self.model(chunk)
                pred_chunk = pred_chunk.argmax(dim=1, keepdim=True)
                pred = torch.cat((pred, pred_chunk))
                del chunk
                del pred_chunk
            del chunks
            torch.cuda.empty_cache()
        else:
            pred_chunk = self.model(x)
            pred_chunk = pred_chunk.argmax(dim=1, keepdim=True)
            pred = torch.cat((pred, pred_chunk))

        pred_np = pred.cpu().detach().numpy()
        result_np.append(pred_np)
        result_np = np.array(result_np)
        result_np = np.reshape(result_np, (-1))

        if 'ocl' not in util.getParameter('AnalysisModule'):
            classes = np.sort(np.unique(util.getParameter('DataClasses')))
            if util.has_sub_classes():
                super_classes, _ = util.map_classes_y(classes) # map sub classes to super classes
                classes = np.unique(super_classes)
            result_np = util.transformZeroIndexDataIntoClasses(result_np, self.all_classes)

        return result_np


    def predict(self, X):
        result = []
        if len(self.act_models) == 0:
            result = self.predict_base(X)
        else:
            # majority vote on DNN, activation DNN and previous activation DNN
            predictions = []
            dnn_predict = np.array(DNNPytorch.predict(self, X), dtype=np.int64)
            for act_model in self.act_models:
                predictions.append(act_model.predict(self.activations))

            inv_class_map = {v: k for k, v in self.class_map.items()}
            for d, a1, a2 in zip(dnn_predict, predictions[0], predictions[1]):
                d = np.int64(inv_class_map[d])
                pred = None
                if d != a2 and d in self.new_classes and a2 in self.new_classes:
                    # take DNN prediction
                    pred = d
                elif d != a2 and d not in self.new_classes and a2 not in self.new_classes:
                    # majority vote on d, a1, a2. if no majority, use a2
                    count = Counter([d, a1, a2])
                    pred = count.most_common(1)[0][0]

                elif d != a2:
                    # if a2 agrees with a1, use a2
                    if a1 == a2:
                        pred = a2
                    else:
                        # if a2 does not agree with A1, use DNN
                        pred = d
                else:
                    # use DNN
                    pred = d
                result.append(pred)
            result = self.__convert_zero_index_into_classes(np.array(result), self.class_map)

        result = result.astype(int)
        return result






































