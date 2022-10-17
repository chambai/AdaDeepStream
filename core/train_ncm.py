'''Train CIFAR10 with PyTorch.'''
# from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
import numpy as np
import util

start_epoch = 0  # start from epoch 0 or last checkpoint epoch

criterion = nn.CrossEntropyLoss()

activation = {}
hook_handle = None
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

# Training, single epoch
def train(net, device, epoch, optimizer, trainloader):
    global hook_handle
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = inputs.to(device)
        targets = targets.type(torch.LongTensor)
        targets = targets.to(device)
        optimizer.zero_grad()

        # Initialize classifier (if novel classes are present)

        hook_handle = net.classifier[-2].register_forward_hook(get_activation('final_hidden'))
        fh_output = net(inputs)
        fh_act = activation['final_hidden']

        # Forward predictionbs
        outputs = net.forward(inputs)
        prediction = net(inputs)

        # Convert labels to match the order seen by the classifier
        targets_converted = net.classifier[-1].convert_labels(targets).to(outputs.device)

        # Compute loss
        loss = criterion(prediction, targets_converted)

        # Backward + update
        loss.backward()
        optimizer.step()

        # Update means
        net.classifier[-1].update_means_decay(fh_act, targets)

        hook_handle.remove()

        # Printing stuff
        train_loss += loss.item()
        _, predicted = prediction.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets_converted).sum().item()
        if batch_idx % 200 == 0:
            print()
            print('TRAINING')
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return (train_loss / (batch_idx + 1)), 100. * correct / total


def test(net, device, epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = inputs.to(device)
            outputs = net.forward(inputs)
            outputs = net.predict(outputs)
            targets_converted = net.linear.convert_labels(targets).to(outputs.device)
            loss = criterion(outputs, targets_converted)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets_converted).sum().item()
            if batch_idx % 100 == 0:
                print()
                print('TEST')
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    print(acc)
    return acc


def loop(net, device, trainloader, epochs=200):
    global hook_handle
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
        raise Exception('unhandled dnn of %s'%(dnn))

    for epoch in range(start_epoch, start_epoch + epochs):

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
        loss_epoch, acc_epoch = train(net, device, epoch, optimizer, trainloader)

        hook_handle.remove()
    return net

def do_train_ncm(net, device, trainloader, epochs):
    net = net.to(device)
    return loop(net, device, trainloader, epochs=epochs)

def progress_bar(current, total, msg=None):
    print(msg)


def get_train_loader(x_train, y_train, class_map):
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
    train_loader = torch.utils.data.DataLoader(TensorDataset(torch.Tensor(torch_x_train_perm), torch.Tensor(y_train)), shuffle=True, **kwargs)
    return train_loader

