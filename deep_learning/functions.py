import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    print(f'acc: {accuracy}, precision: {precision}, recall: {recall}, f1-score: {f1}')


def draw_plot(losses, val_losses):
    plt.plot(losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.savefig("loss_graph.png", bbox_inches='tight', pad_inches=0)
