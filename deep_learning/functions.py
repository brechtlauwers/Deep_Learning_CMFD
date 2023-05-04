import torch
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
from collections import Counter
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_data():
    dataset_l = []
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # MICC-F2000
    dataset_l.append(datasets.ImageFolder("/home/brechtl/Pictures/Data/MICC/MICC-F2000", transform=transform))
    # MICC-F220
    dataset_l.append(datasets.ImageFolder("/home/brechtl/Pictures/Data/MICC/MICC-F220", transform=transform))
    # CoMoFoD
    dataset_l.append(datasets.ImageFolder("/home/brechtl/Pictures/Data/CoMoFoD_small_v2/", transform=transform))
    # CASIAv1
    dataset_l.append(datasets.ImageFolder("/home/brechtl/Pictures/Data/CASIA/CASIA1/", transform=transform))
    # CASIAv2
    dataset_l.append(datasets.ImageFolder("/home/brechtl/Pictures/Data/CASIA/CASIA2/", transform=transform))
    # GRIP
    dataset_l.append(datasets.ImageFolder("/home/brechtl/Pictures/Data/GRIP/", transform=transform))

    dataset = torch.utils.data.ConcatDataset(dataset_l)

    total = []
    for x in dataset_l:
        total.append(dict(Counter(x.targets)))

    c = Counter()
    for x in total:
        c.update(x)

    c["Forged"] = c.pop(0)
    c["Original"] = c.pop(1)
    print(c)

    return dataset


def evaluate_model(test_dl, model):
    correct = 0
    predicted_labels, gt_labels = [], []
    total_data_points = len(test_dl.dataset)

    with torch.no_grad():
        for inputs, targets in test_dl:
            # evaluate the model on the test set
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            yhat = model(inputs)
            output = torch.sigmoid(yhat)
            output = output > 0.5

            correct += (output.flatten().long() == targets).sum().item()
            predicted_labels = np.append(predicted_labels, output.cpu().detach().numpy().flatten())
            gt_labels = np.append(gt_labels, targets.cpu().detach().numpy().flatten())

    accuracy = 100 * correct / total_data_points

    precision = precision_score(gt_labels, predicted_labels)
    recall = recall_score(gt_labels, predicted_labels)
    f1 = f1_score(gt_labels, predicted_labels)
    print(f'accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1-score: {f1}')


def draw_plot(losses, val_losses):
    plt.plot(losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.savefig("loss_graph.png", bbox_inches='tight', pad_inches=0)
