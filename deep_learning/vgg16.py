import copy

import torchvision.utils

import deep_learning.functions as functions
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np

from torchvision import models
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image

import torch.optim as optim
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CMFD_VGG16(nn.Module):
    def __init__(self, vgg_pt):
        super(CMFD_VGG16, self).__init__()
        self.vgg_pt = vgg_pt
        self.new_layers = nn.Sequential(nn.Linear(25088, 1028),
                                        nn.Dropout(p=0.4),
                                        nn.BatchNorm1d(1028),
                                        nn.Linear(1028, 2))

    def forward(self, x):
        x = self.vgg_pt(x)
        x = self.new_layers(x)
        return x


def VGG():
    # VGG-16 takes 224x224 images as input

    # Use the best up-to-date weights
    vgg_pt = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES)

    # Drop the classification part
    # vgg_pt.classifier = nn.Sequential(*[vgg_pt.classifier[i] for i in range(0)])
    vgg_pt.classifier = nn.Sequential()
    # vgg_pt.features = nn.Sequential(*[vgg_pt.features[i] for i in range(17)])

    # Freeze all the layers
    for param in vgg_pt.parameters():
        param.requires_grad = False

    # Add new classification layer to model for cross entropy classification
    my_vgg = CMFD_VGG16(vgg_pt=vgg_pt)

    return my_vgg


def make_train_step(model, loss_fn, optimizer):
    def train_step(x, y):
        optimizer.zero_grad()  # Clear the gradients
        model.train()
        yres = model(x)  # Compute model output

        # print(yres)
        # print(y.flatten().long())

        loss = loss_fn(yres, y.flatten().long())  # Calculate loss
        loss.backward()  # Backpropagating the error
        optimizer.step()  # Update parameters (weights)
        return loss.item(), yres
    return train_step


def start():
    print(DEVICE)

    if DEVICE == "cuda":
        torch.cuda.synchronize()  # wait for move to complete
        start_timer = torch.cuda.Event(enable_timing=True)
        end_timer = torch.cuda.Event(enable_timing=True)
    else:
        start_timer, end_timer = None, None

    model = VGG()
    model = model.to(DEVICE)
    print(model)
    # loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = nn.CrossEntropyLoss()
    # print(model.new_layers[3])
    optimizer = optim.SGD(model.parameters(), lr=0.00002, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=0.00002)

    dataset, class_weights = functions.load_data()

    # Train size -> 80%
    # Test size -> 20%
    train_size = int(len(dataset) * 8 / 10)
    test_size = int(len(dataset) - train_size)

    print(f'Total amount of images: {len(dataset)}')
    print(f'Train size: {train_size}')
    print(f'Test size: {test_size}')

    train_data, test_data = random_split(dataset, [train_size, test_size])

    targets = [label for _, label in train_data]

    class_weights_all = class_weights[targets]
    weighted_sampler = torch.utils.data.WeightedRandomSampler(weights=class_weights_all,
                                                              num_samples=len(class_weights_all),
                                                              replacement=True)

    train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=False, sampler=weighted_sampler)
    test_loader = DataLoader(dataset=test_data, batch_size=50, shuffle=True)

    # functions.visualize_image(model, test_loader)
    # functions.visualize_model(model, test_loader)

    epochs = 100

    if DEVICE == "cuda":
        start_timer.record()

    print("Training started...")
    model, losses, val_losses = train(model, loss_fn, optimizer, epochs,
                                      train_loader, test_loader, train_size, test_size)

    if DEVICE == "cuda":
        end_timer.record()
        torch.cuda.synchronize()

    total_time = round(start_timer.elapsed_time(end_timer) / 1000.0, 2)

    print(f'Total training time: {total_time}s')

    torch.save(model.state_dict(), 'vgg16_best.pt')
    # functions.draw_plot(losses, val_losses)
    # functions.visualize_model(model, test_loader)

    # Evaluate best model
    functions.evaluate_model(test_loader, model)


def train(model, loss_fn, optimizer, epochs, train_loader, test_loader, train_size, test_size):
    all_accuracies = []
    losses, val_losses = [], []
    loss = 0
    best_accuracy = -np.Inf
    best_model = None
    train_step = make_train_step(model, loss_fn, optimizer)

    # Early stopping
    last_loss = 100
    patience = 5
    triggertimes = 0

    for epoch in range(epochs):
        correct = 0
        running_loss = 0

        for counter, (x_batch, y_batch) in enumerate(train_loader, 1):
            # unsqeeze the tensor to add another dimension
            x_batch = x_batch.to(torch.float32)
            x_batch = x_batch.to(DEVICE)  # move to gpu
            y_batch = y_batch.to(torch.float32).unsqueeze(-1)
            y_batch = y_batch.to(DEVICE)  # move to gpu

            loss, output = train_step(x_batch, y_batch)
            running_loss += loss * x_batch.size(0)
            # print("loss: " + str(loss))
            # output = torch.sigmoid(output)
            _, preds = torch.max(output.data, 1)
            # print(output)

            if counter % 10 == 0 or counter == len(train_loader):
                print('[{}/{}, {}/{}] loss: {:.8}'.format(epoch, epochs, counter, len(train_loader), loss))

            # print(preds)
            # print(y_batch.flatten().long())
            correct += (preds == y_batch.flatten().long()).float().sum()

        losses.append(running_loss / train_size)
        accuracy = 100 * correct / train_size
        all_accuracies.append(accuracy)

        val_loss = validation(model, loss_fn, test_loader, test_size)
        val_losses.append(val_loss)

        # Early stopping
        if val_loss < last_loss:
            triggertimes = 0
        else:
            triggertimes += 1
            print(f'Trigger times: {triggertimes}')

            if triggertimes >= patience:
                print(f'Early stopping!')
                print(f'Best accuracy: {best_accuracy}')
                if best_accuracy < max(all_accuracies):
                    best_model = copy.deepcopy(model)
                return best_model, losses, val_losses

        last_loss = val_loss

        print('epoch {}, loss: {}, val loss: {}, acc: {}'.format(epoch, loss, val_loss, accuracy))

        if epoch < 1:
            best_accuracy = max(all_accuracies)
            best_model = copy.deepcopy(model)
        else:
            if best_accuracy > max(all_accuracies):
                continue
            else:
                best_accuracy = max(all_accuracies)
                best_model = copy.deepcopy(model)

    print(f'Best accuracy: {best_accuracy}')
    return best_model, losses, val_losses


def validation(model, loss_fn, test_loader, test_size):
    running_val_loss = 0
    # Evaluate the model with test data
    with torch.no_grad():
        for x_val, y_val in test_loader:
            x_val = x_val.to(torch.float32)
            x_val = x_val.to(DEVICE)
            y_val = y_val.to(torch.float32).unsqueeze(-1)
            y_val = y_val.to(DEVICE)

            model.eval()

            yhat = model(x_val)
            val_loss = loss_fn(yhat, y_val.flatten().long()).detach().cpu().numpy()
            running_val_loss += val_loss * x_val.size(0)

    return running_val_loss / test_size
