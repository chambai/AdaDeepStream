from core.dnn_wrapper import DNN
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
import os
import numpy as np
import matplotlib.pyplot as plt
import util
import torch
import torch.optim as optim
from torchvision import transforms
from torchvision.models.mobilenet import mobilenet_v2
from torchvision.models.vgg import vgg16
from external.ocl.models.resnet import Reduced_ResNet18
from torch.optim.lr_scheduler import StepLR
from torch.nn import CrossEntropyLoss
from core import pretrain_dnn, dataset
import time
import uuid
from collections import Counter

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
        return is_loaded

    def print_summary(self):
        util.thisLogger.logInfo('Summary not implemented for Pytorch DNNs yet!')


    def pytorch_train(self, model, device, train_loader, optimizer, epoch, dataset_name, loss_func):
        show_log = False
        log_interval = 10
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
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

        loss /= len(train_loader)
        epoch_acc = correct / len(train_loader.dataset)
        percent_acc = 100. * correct / len(train_loader.dataset)

        print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            loss, correct, len(train_loader.dataset),percent_acc))

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

        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            percent_acc))

        return test_loss.item(), epoch_acc, percent_acc

    def get_model(self, base_model_name):
        if base_model_name == 'mobilenet':
            model = mobilenet_v2(pretrained=True) # mobilenet v1 is not available in pytorch
        elif base_model_name == 'vgg16':
            model = vgg16(pretrained=True)
        elif base_model_name == 'resnet18':
            model = Reduced_ResNet18(len(util.getParameter('DataClasses')))
        else:
            raise Exception('unhandled model name of %s ' % (base_model_name))
        return model

    def get_optimizer_params(self, base_model_name, reduce_lr_percent=0, is_ocl=False):
        if base_model_name == 'mobilenet':
            lr = 0.0001
            if is_ocl:
                lr = 0.01 # this value works best with OCL
            wd = 0
            if reduce_lr_percent > 0:
                lr = ( lr / 100 ) * reduce_lr_percent # reduce lr by percentage given
        elif base_model_name == 'vgg16':
            lr = 0.01
            wd = 0
        else:
            raise Exception('unhandled base model name of %s'%(base_model_name))
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
                lr = ( lr / 100 ) * reduce_lr_percent # reduce lr by percentage given
            optimizer = optim.Adamax(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-07)
        elif base_model_name == 'vgg16':
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

    def train(self, dataset_name, classes, in_x_train, in_y_train, in_x_test, in_y_test, dataMap='', raise_accuracy_exception=False, batch_size = 256, model=None, epochs=None, update_for_transfer_learning=True, save_model=True, test_model=True, early_stopping_patience=4, reduce_lr_percent=0, in_x_train_act=None):
        torch.cuda.empty_cache() # clear GPU memory

        # set parameters-------------------------------
        base_model_name = util.getParameter('Dnn')
        full_model_name = pretrain_dnn.getFullDnnModelPathAndName()
        # batch_size = 32
        if model is None:
            model = self.model
        if epochs is None:
            epochs = 50
        reduce_lr_gamma = 0.7
        kwargs = {'batch_size': batch_size}
        # ----------------------------------------------

        # set parameters-------------------------------
        # Only train and test on the specified classes
        classesName = ''.join(str(x) for x in classes)
        numClassesStr = str(len(classes))
        num_classes = len(classes)

        # make sure the arrays are numpy arrays as this makes the conversion to/from 0 and 1s work correctly
        y_train = np.array(in_y_train, dtype=np.int16)
        y_test = np.array(in_y_test, dtype=np.int16)
        x_train = np.array(in_x_train, dtype=np.float32)
        x_test = np.array(in_x_test, dtype=np.float32)

        if in_x_train_act is None:
            torch_x_train_act = np.full((x_train.shape[0],0),0)
            torch_x_test_act = np.full((x_test.shape[0], 0), 0)
        else:
            x_train_act = np.array(in_x_train_act, dtype=np.float32)
            x_test_act = np.array(in_x_test_act, dtype=np.float32)
            torch_x_train_act = torch.from_numpy(x_train_act)
            torch_x_test_act = torch.from_numpy(x_test_act)

        # source classes
        sourceClasses = []
        targetClasses = []

        # Map data if mappings specified
        if dataMap != '':
            classes = np.unique(in_y_test)
            num_classes = len(classes)
            dataMap = '_map' + dataMap
            sourceClasses = util.getParameter('MapOriginalYValues')
            targetClasses = util.getParameter('MapNewYValues')
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
        print('Is GPU available to torch? %s' % (torch.cuda.is_available()))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            kwargs.update({'num_workers': 4, 'pin_memory': True})

        torch_x_train = torch.from_numpy(x_train)
        torch_x_train_perm = torch_x_train.permute(0, 3, 1, 2)
        torch_x_test = torch.from_numpy(x_test)
        torch_x_test_perm = torch_x_test.permute(0, 3, 1, 2)

        # setup data
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        train_loader = torch.utils.data.DataLoader(TensorDataset(torch.Tensor(torch_x_train_perm), torch.Tensor(torch_x_train_act), torch.Tensor(y_train)), shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(TensorDataset(torch.Tensor(torch_x_test_perm), torch.Tensor(torch_x_test_act), torch.Tensor(y_test)), shuffle=False, **kwargs)
        print('Number of train batches: {} Number of test batches: {}'.format(len(train_loader), len(test_loader)))


        # Note: we do not need to specify a softmax layer as the softmax is integrated with the nn.CrossEntropyLoss function
        model = self.get_model(base_model_name)
        optimizer = self.get_optimizer(model, base_model_name)

        if update_for_transfer_learning:
            # add an extra convolutional layer at the begining so we don't have to increase the size of the image
            num_input_channels = torch_x_train_perm.shape[1]
            if num_input_channels != 3:
                first_conv_layer = [torch.nn.Conv2d(num_input_channels, 3, kernel_size=3, stride=1, padding=1, dilation=1,
                                                    groups=num_input_channels, bias=True)]
                first_conv_layer.extend(list(model.features))
                model.features = torch.nn.Sequential(*first_conv_layer)
                # disable grad for pre-trained layers only (train the new first conv layer and the classifier layers
                for param in model.features[1:].parameters():
                    param.requires_grad = False

            # add an extra linear layer to get the output classes from 1000 to 10
            model.classifier[-1] = torch.nn.Linear(in_features=model.classifier[-1].in_features, out_features=num_classes)

        model = model.to(device)


        early_stopping_patience = early_stopping_patience # stop if the same test acc value is encountered consecutively
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
            train_epoch_loss, train_epoch_acc = self.pytorch_train(model, device, train_loader, optimizer, epoch, dataset_name, loss_func)
            if test_model:
                test_epoch_loss, test_epoch_acc, percent_acc = self.__pytorch_test(model, device, test_loader, dataset_name, loss_func)
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
                    util.thisLogger.logInfo('training early stopped at %s test accuracy as there were %s consecutive values'%(percent_acc, early_stopping_patience))
                    break;

        scores_train = [losses[0], accuracies[0]]
        scores_test = [losses[1], accuracies[1]]

        if save_model:
            # save full model
            torch.save(model, full_model_name)

            #--------------------------------------------
            stats = 'DnnTrainLoss=%s\nDnnTrainAccuracy=%s' % (scores_train[0], scores_train[1])
            stats = stats + '\nDnnTestLoss=%s\nDnnTestAccuracy=%s' % (scores_test[0], scores_test[1])

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

            # Save model and weights
            print('Saved trained model at %s ' % (full_model_name))
            plt.savefig(full_model_name + '.png')
            plt.close()
        else:
            stats = 'DnnTrainLoss=%s\nDnnTrainAccuracy=%s' % ('0', '0')
            stats = stats + '\nDnnTestLoss=%s\nDnnTestAccuracy=%s' % ('0', '0')

        self.model = model
        self.device = device

        y_train = util.transformZeroIndexDataIntoClasses(y_train, classes)
        y_test = util.transformZeroIndexDataIntoClasses(y_test, classes)

        if raise_accuracy_exception:
            if scoresTrain[1] < 0.9 or scoresTest[1] < 0.8:
                raise Exception('DNN stats too low:\n' + stats)

        torch.cuda.empty_cache()  # clear GPU memory
        return stats

    def predict(self, X):
        result_np = []
        self.model = self.model.to(self.device)
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
        return result_np

    def get_num_layers(self):
        layers = self.get_layer_names()
        return len(layers)

    def get_layer_output(self, layer_num, X):
        # define helper function
        def get_features(name):
            def hook(model, input, output):
                features[name] = output.detach()
            return hook

        layers = [module[1] for module in self.model.named_modules() if not isinstance(module, torch.nn.Sequential)]
        layer = layers[layer_num]
        layer.register_forward_hook(get_features('feats'))

        # move data to device
        x = torch.Tensor(X).permute(0, 3, 1, 2).to(self.device)

        # activation extraction
        features = {}
        # forward pass [with feature extraction]
        preds = self.model(x)
        # move predictions to cpu
        preds.detach().cpu().numpy()
        # because of the forward pass and registered hook, features (activations)
        # will appear in the features dictionary
        acts = features['feats'].detach().cpu().numpy()
        if len(acts.shape) == 4:
            acts = np.reshape(acts, (acts.shape[0], acts.shape[3], acts.shape[2], acts.shape[1]))
        return acts

    # replaces model.layers[layerNum].name
    def get_layer_name(self, layer_num):
        layer_name = self.get_layer_names()[layer_num]
        return layer_name

    def get_layer_names(self):
        layer_names = ['%s.%s'%(module[0],type(module[1]).__name__) for module in self.model.named_modules() if not isinstance(module, torch.nn.Sequential)]
        return layer_names

#-------------------------------------------------------------------------------------------------------------------
class DNNPytorchAdaptive(DNNPytorch):
    def __init__(self):
        DNNPytorch.__init__(self)
        self.all_classes = None
        self.class_map = None
        self.new_x_train = np.array([0])
        self.new_x_test = np.array([0])
        self.new_y_train = np.array([0])
        self.new_y_test = np.array([0])
        self.activations = None
        self.act_models = []
        self.new_classes = []
        self.start_epoch = 0  # start from epoch 0 or last checkpoint epoch
        self.criterion = torch.nn.CrossEntropyLoss()
        pass

    def fit(self, X, y):
        # todo: check if this is used or not
        self.all_classes = np.sort(np.unique(y)).tolist()
        self.class_map = {}
        for i, c in enumerate(self.all_classes):
            self.class_map[c] = i

    def get_copy(self):
        temp_model_name = 'temp_model_%s_%s.plt' % ('DNNPytorchAdaptive', uuid.uuid4())
        torch.save(self.model, temp_model_name)
        copy = DNNPytorchAdaptive()
        copy.load(temp_model_name)
        copy.all_classes = self.all_classes
        copy.class_map = self.class_map
        copy.new_x_train = self.new_x_train
        copy.new_x_test = self.new_x_test
        copy.new_y_train = self.new_y_train
        copy.new_y_test = self.new_y_test
        copy.activations = self.activations
        copy.act_models = self.act_models
        copy.new_classes = self.new_classes
        self.start_epoch = self.start_epoch
        self.criterion = self.criterion
        os.remove(temp_model_name)
        return copy

    def partial_fit(self, x, y):
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

        self.model = self.__adapt(self.model, previous_classes, self.class_map, x, y)

    def __adapt(self, model, previous_classes, class_map, x_train, y_train):
        data_map = dataset.getDataMapAsString()
        start = time.time()
        y_train = np.array(y_train, dtype=np.int16)
        x_test = x_train
        y_test = y_train

        dnn_name = util.getParameter('Dnn')
        batch_size = 8
        if len(x_train) > 100:
            if 'vgg' in dnn_name:
                epochs = 3
            else:
                epochs = 18
        else:
            if 'vgg' in dnn_name:
                epochs = 3
            else:
                epochs = 18
        num_classes = len(list(class_map.keys()))
        input_shape = len(x_train[0])
        num_units = 8

        x_train = np.array(x_train, dtype=np.float32)
        x_test = np.array(x_test, dtype=np.float32)

        self.new_x_train = x_train
        self.new_x_test = x_test
        self.new_y_train = y_train
        self.new_y_test = y_test


        # if there's a new class, freeze existing layers and add the new layer
        previous_num_classes = len(previous_classes)

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

        for param in self.model.parameters():
            param.requires_grad = False  # freeze all parameters
        for param in self.model.classifier.parameters():
            param.requires_grad = True  # unfreeze all classifier layers

        self.model = self.get_new_model_final_layer(model, num_classes)

        self.model = self.do_train(self.model, self.device, self.get_train_loader(x_train, y_train, self.class_map), epochs=epochs)


        end = time.time()
        print(end - start)
        return self.model

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

        return model

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
        model_name = 'DEEP NCM'
        iters = []
        losses_training = []
        accuracy_training = []
        accuracies_test = []
        lr = 0
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
                print('LR now is ' + str(lr))

            # Perform 1 training epoch
            loss_epoch, acc_epoch = self.train(net, device, epoch, optimizer, trainloader)
            if acc_epoch == 100:
                acc_epoch_count += 1
            if acc_epoch_count == 3:
                break;

        return net

    def train(self, net, device, epoch, optimizer, trainloader):
        print_train_data = False
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            print('\nBatch: %d' % batch_idx)
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
                    print()
                    print('TRAINING')
                    print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                 % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        if print_train_data:
            result = (train_loss / (batch_idx + 1)), 100. * correct / total
        else:
            result = 0.0, 0.0

        return result


    def predict(self, X):
        result = []
        if len(self.act_models) == 0:
            result = DNNPytorch.predict(self, X)
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

        return result






































