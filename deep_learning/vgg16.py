import torch
import torch.nn as nn

from pytorch_model_summary import summary

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image

import torch.optim as optim
from torch.utils.data import DataLoader


class CMFD_VGG16(nn.Module):
    def __init__(self, vgg_pt):
        super(CMFD_VGG16, self).__init__()
        self.vgg_pt = vgg_pt
        self.new_layers = nn.Sequential(nn.Linear(4096, 1))
        # self.new_layers = nn.Sequential(nn.Linear(4096, 1))

    def forward(self, x):
        x = self.vgg_pt(x)
        x = self.new_layers(x)
        return x


def VGG():
    # VGG-16 takes 224x224 images as input

    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    dataset = datasets.ImageFolder("/home/brechtl/Pictures/Data/MICC/MICC-F2000", transform=transform)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Run this to test your data loader
    images, labels = next(iter(dataloader))
    save_image(images[0], "original.png")

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

    for param in vgg_pt.parameters():
        param.requires_grad = False

    # vgg_pt.classifier._modules['6'] = nn.Linear(4096, 1)
    # vgg_pt.classifier._modules['7'] = nn.Sigmoid

    vgg_pt.classifier = nn.Sequential(*[vgg_pt.classifier[i] for i in range(5)])

    my_vgg = CMFD_VGG16(vgg_pt=vgg_pt)

    # print(my_vgg)

    return my_vgg


def make_train_step(model, loss_fn, optimizer):
    def train_step(x, y):
        model.train()
        optimizer.zero_grad()  # Clear the gradients
        yres = model(x)  # Compute model output
        loss = loss_fn(y, yres)  # Calculate loss
        loss.backward()  # Backpropagating the error
        optimizer.step()  # Update parameters (weights)
        return loss.item()
    return train_step


def train():
    model = VGG()
    loss_fn = nn.BCEWithLogitsLoss()
    # loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    train_step = make_train_step(model, loss_fn, optimizer)

    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor()])

    dataset = datasets.ImageFolder("/home/brechtl/Pictures/Data/MICC/MICC-F2000", transform=transform)
    train_data, test_data = random_split(dataset, [1600, 400])

    train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True)

    loss = None
    losses = []
    val_losses = []

    for epoch in range(50):
        for x_batch, y_batch in train_loader:
            # unsqeeze the tensor to add another dimension
            x_batch = x_batch.to(torch.float32)
            y_batch = y_batch.to(torch.float32).unsqueeze(-1)

            loss = train_step(x_batch, y_batch)
            losses.append(loss)
            print(loss)

        # Evaluate the model with test data
        with torch.no_grad():
            for x_val, y_val in test_loader:
                x_val = x_val.to(torch.float32)
                y_val = y_val.to(torch.float32).unsqueeze(-1)

                model.eval()

                yhat = model(x_val)
                val_loss = loss_fn(y_val, yhat)
                val_losses.append(val_loss.item())

        print('epoch {}, MSE: {}, val loss: {}'.format(epoch, loss, val_loss))