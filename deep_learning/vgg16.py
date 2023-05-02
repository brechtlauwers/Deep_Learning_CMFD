import copy
import deep_learning.functions as functions

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np

from pytorch_model_summary import summary

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image

import torch.optim as optim
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CMFD_VGG16(nn.Module):
    def __init__(self, vgg_pt):
        super(CMFD_VGG16, self).__init__()
        self.vgg_pt = vgg_pt
        self.new_layers = nn.Sequential(nn.Linear(4096, 1))

    def forward(self, x):
        x = self.vgg_pt(x)
        x = self.new_layers(x)
        return x


def VGG():
    # VGG-16 takes 224x224 images as input

    # transform = transforms.Compose([transforms.Resize((224, 224)),
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    #
    # dataset = datasets.ImageFolder("/home/brechtl/Pictures/Data/MICC/MICC-F2000", transform=transform)
    #
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Run this to test data loader
    # images, labels = next(iter(dataloader))
    # save_image(images[0], "original.png")

    vgg_pt = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    # Cut layers from the model

    # vgg_pt_prediction = vgg_pt.features(images[0].unsqueeze(0))
    # vgg_pt_prediction = torch.squeeze(vgg_pt_prediction)

    # print(vgg_pt_prediction)
    # print(vgg_pt_prediction.shape)
    # print(vgg_pt.classifier)
    # print(summary(vgg_pt, images[0].unsqueeze(0), show_input=False))
    # print(vgg_pt)

    # gray_scale = torch.sum(vgg_pt_prediction, 0)
    # gray_scale = gray_scale / vgg_pt_prediction.shape[0]
    # gray_scale = gray_scale.detach().numpy()
    # print(gray_scale.shape)
    # plt.imshow(gray_scale)
    # plt.savefig("feature_map.png", bbox_inches='tight', pad_inches=0)

    vgg_pt.classifier = nn.Sequential(*[vgg_pt.classifier[i] for i in range(6)])

    for param in vgg_pt.parameters():
        param.requires_grad = False

    # vgg_pt.classifier._modules['6'] = nn.Linear(4096, 1)
    # vgg_pt.classifier._modules['7'] = nn.Sigmoid

    my_vgg = CMFD_VGG16(vgg_pt=vgg_pt)
    # images, labels = next(iter(dataloader))
    # print(summary(my_vgg, images[0].unsqueeze(0), show_input=False))
    # print(my_vgg)

    return my_vgg


def make_train_step(model, loss_fn, optimizer):
    def train_step(x, y):
        optimizer.zero_grad()  # Clear the gradients

        model.train()

        yres = model(x)  # Compute model output
        loss = loss_fn(yres, y)  # Calculate loss
        loss.backward()  # Backpropagating the error
        optimizer.step()  # Update parameters (weights)
        return loss.item(), yres
    return train_step


def start():
    print(DEVICE)

    model = VGG()
    model = model.to(DEVICE)
    # print(model)
    loss_fn = nn.BCEWithLogitsLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.0001)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    train_step = make_train_step(model, loss_fn, optimizer)

    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    dataset = datasets.ImageFolder("/home/brechtl/Pictures/Data/MICC/MICC-F2000", transform=transform)
    train_data, test_data = random_split(dataset, [1600, 400])

    train_loader = DataLoader(dataset=train_data, batch_size=80, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=40, shuffle=True)

    loss = None
    losses = []
    val_losses = []
    all_accuracies = []
    epochs = 30
    best_accuracy = -np.Inf
    best_model = None

    # functions.evaluate_model(test_loader, model)

    for epoch in range(epochs):
        correct = 0
        for x_batch, y_batch in train_loader:
            # unsqeeze the tensor to add another dimension
            x_batch = x_batch.to(torch.float32)
            x_batch = x_batch.to(DEVICE)  # move to gpu
            y_batch = y_batch.to(torch.float32).unsqueeze(-1)
            y_batch = y_batch.to(DEVICE)  # move to gpu

            loss, output = train_step(x_batch, y_batch)
            losses.append(loss)
            # print("loss: " + str(loss))
            # print(output)
            output = torch.sigmoid(output)
            output = output > 0.5
            # print(output)

            correct += (output == y_batch).float().sum()

        accuracy = 100 * correct / len(train_data)
        all_accuracies.append(accuracy)

        # Evaluate the model with test data
        with torch.no_grad():
            for x_val, y_val in test_loader:
                x_val = x_val.to(torch.float32)
                x_val = x_val.to(DEVICE)
                y_val = y_val.to(torch.float32).unsqueeze(-1)
                y_val = y_val.to(DEVICE)

                model.eval()

                yhat = model(x_val)
                val_loss = loss_fn(yhat, y_val)
                val_losses.append(val_loss.item())

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

    torch.save(best_model.state_dict(), 'vgg16_best.pt')
    functions.draw_plot(losses, val_losses)
    functions.evaluate_model(test_loader, model)
