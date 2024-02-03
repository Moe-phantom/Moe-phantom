import argparse
import os
import time
import torch
import matplotlib.pyplot as plt 
import numpy as np
import json

from torch import nn, optim
from torchvision import datasets, models, transforms
from collections import OrderedDict
from PIL import Image

def validate_parameters():
    print("Validating parameters")
    if (arguments.gpu and not torch.cuda.is_available()):
        raise Exception("--gpu option enabled...but no GPU detected")
    if(not os.path.isdir(arguments.data_directory)):
        raise Exception('Directory does not exist!')
    data_dir = os.listdir(arguments.data_directory)
    if (not set(data_dir).issubset({'test','train','valid'})):
        raise Exception('Missing: test, train, or valid sub-directories')
    if arguments.arch not in ('vgg', 'densenet', None):
        raise Exception('Please choose one of: vgg or densenet')

def process_data(data_dir):
    print("Processing data into iterators")
    train_dir, test_dir, valid_dir = data_dir 
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    modified_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])    
    train_datasets = datasets.ImageFolder(train_dir, transform=modified_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=modified_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=modified_transforms)

    # Criteria: Data batching
    train_loaders = torch.utils.data.DataLoader(train_datasets, batch_size=32, shuffle=True)
    valid_loaders = torch.utils.data.DataLoader(valid_datasets, batch_size=32, shuffle=True)
    test_loaders = torch.utils.data.DataLoader(test_datasets, batch_size=32, shuffle=True)
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    loaders = {'train': train_loaders, 'valid': valid_loaders, 'test': test_loaders, 'labels': cat_to_name}
    return loaders

def retrieve_data():
    print("Retrieving data")
    train_dir = arguments.data_directory + '/train'
    test_dir = arguments.data_directory + '/test'
    valid_dir = arguments.data_directory + '/valid'
    data_dir = [train_dir, test_dir, valid_dir]
    return process_data(data_dir)

def construct_model(data):
    print("Building model object")
    if (arguments.arch is None):
        arch_type = 'vgg'
    else:
        arch_type = arguments.arch
    if (arch_type == 'vgg'):
        model = models.vgg19(pretrained=True)
        input_node = 25088
    elif (arch_type == 'densenet'):
        model = models.densenet121(pretrained=True)
        input_node = 1024
    if (arguments.hidden_units is None):
        hidden_units = 4096
    else:
        hidden_units = arguments.hidden_units
    for param in model.parameters():
        param.requires_grad = False
    hidden_units = int(hidden_units)
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_node, hidden_units)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1)),
    ]))
    model.classifier = classifier
    return model

def calculate_accuracy(model, loader, device='cpu'):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total  

def train_model(model, data):
    print("Training model")
    
    print_every = 40
    
    if (arguments.learning_rate is None):
        learn_rate = 0.001
    else:
        learn_rate = arguments.learning_rate
    if (arguments.epochs is None):
        epochs = 3
    else:
        epochs = arguments.epochs
    if (arguments.gpu):
        device = 'cuda'
    else:
        device = 'cpu'
    
    learn_rate = float(learn_rate)
    epochs = int(epochs)
    
    train_loader = data['train']
    valid_loader = data['valid']
    test_loader = data['test']
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)
    
    steps = 0
    model.to(device)
    
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(train_loader):
            steps += 1
            
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()     
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                valid_accuracy = calculate_accuracy(model, valid_loader, device)
                print("Epoch: {}/{}... ".format(e + 1, epochs),
                      "Loss: {:.4f}".format(running_loss / print_every),
                      "Validation Accuracy: {}".format(round(valid_accuracy, 4)))            
                running_loss = 0
    print("DONE TRAINING!")
    test_result = calculate_accuracy(model, test_loader, device)
    print('Final accuracy on test set: {}'.format(test_result))
    return model

def save_trained_model(model):
    print("Saving trained model")
    if (arguments.save_dir is None):
        save_dir = 'check.pth'
    else:
        save_dir = arguments.save_dir
    checkpoint = {
        'model': model.cpu(),
        'features': model.features,
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
    }
    torch.save(checkpoint, save_dir)
    return 0

def create_deep_learning_model():
    validate_parameters()
    data = retrieve_data()
    model = construct_model(data)
    model = train_model(model, data)
    save_trained_model(model)
    return None

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a neural network with one of many options!')
    parser.add_argument('data_directory', help='Data directory (required)')
    parser.add_argument('--save_dir', help='Directory to save a neural network.')
    parser.add_argument('--arch', help='Models to use OPTIONS[vgg, densenet]')
    parser.add_argument('--learning_rate', help='Learning rate')
    parser.add_argument('--hidden_units', help='Number of hidden units')
    parser.add_argument('--epochs', help='Epochs')
    parser.add_argument('--gpu', action='store_true', help='GPU')
    args = parser.parse_args()
    return args

def main_execution():
    print("Creating a deep learning model")
    global arguments
    arguments = parse_arguments()
    create_deep_learning_model()
    print("Model finished!")

main_execution()
