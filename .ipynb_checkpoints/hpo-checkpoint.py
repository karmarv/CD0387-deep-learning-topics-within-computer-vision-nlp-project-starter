#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse
import time
import os
import sys
import logging

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader, criterion, device):
    '''
    EDIT: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
    '''
    model.eval()
    running_loss, running_corrects = 0, 0
    for inputs, labels in test_loader:
        inputs=inputs.to(device)
        labels=labels.to(device)
        outputs=model(inputs)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    total_loss = running_loss // len(test_loader)
    total_acc = running_corrects // len(test_loader)
    print(f'Test set: Accuracy: {running_corrects}/{len(test_loader.dataset)} = {100*total_acc}%),\t Testing Loss: {total_loss}')

def train(model, train_loader, validation_loader, criterion, optimizer, epochs, device):
    '''
    EDIT: Complete this function that can take a model and
          data loaders for training and will get train the model
    '''
    best_loss=1e6
    image_dataset={'train':train_loader, 'valid':validation_loader}
    loss_counter=0
    
    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            if phase=='train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in image_dataset[phase]:
                inputs=inputs.to(device)
                labels=labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                if phase=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss // len(image_dataset[phase])
            epoch_acc = running_corrects // len(image_dataset[phase])
            if phase=='valid':
                if epoch_loss<best_loss:
                    best_loss=epoch_loss
                else:
                    loss_counter+=1
            logger.info("Epoch {},\t {} loss: {:.4f}, acc: {:.4f}, best loss: {:.4f}\n".format(epoch, phase, epoch_loss, epoch_acc, best_loss))
        if loss_counter==1:
            break            
    return model
    
def net():
    '''
    EDIT: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False   
    num_features=model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 512),
                             nn.ReLU(inplace=True), 
                             nn.Linear(512,133))
    return model

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    train_dir = os.path.join(data, 'train')
    test_dir = os.path.join(data, 'test')
    val_dir =os.path.join(data, 'valid')

    training_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()])
    valid_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()])
    testing_transform = transforms.Compose([
        transforms.RandomResizedCrop((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])
  
    train_set = torchvision.datasets.ImageFolder(root=train_dir, transform=training_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_set = torchvision.datasets.ImageFolder(root=test_dir, transform=testing_transform)
    test_loader  = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
    validation_set = torchvision.datasets.ImageFolder(root=val_dir, transform=valid_transform)
    validation_loader  = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True) 
    
    return train_loader, test_loader, validation_loader

def main(args):
    '''
    EDIT: Initialize a model by calling the net function
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on Device {device}")
    model=net()
    model=model.to(device)
 
    '''
    EDIT: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    
    '''
    EDIT: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    logger.info(int(args.batch_size))
    train_loader, test_loader, validation_loader = create_data_loaders(args.data, int(args.batch_size))
    logger.info("Start Model Training")
    model=train(model, train_loader, validation_loader, loss_criterion, optimizer, args.epochs, device)
    
    '''
    EDIT: Test the model to see its accuracy
    '''
    logger.info("Testing Model")
    test(model, test_loader, loss_criterion, device)
    
    '''
    EDIT: Save the trained model
    '''
    logger.info("Saving Model")
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    EDIT: Specify any training args that you might need
    '''
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for training (default: 32)",
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=0.1, 
        metavar="LR", 
        help="learning rate (default: 0.1)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="N",
        help="number of epochs to train (default: 5)",
    )
    
    parser.add_argument("--model-dir", type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    args=parser.parse_args()
    
    main(args)
