import torch
import torch.utils.data
import torchvision.utils
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torchvision.utils import save_image
from pytorch_model_summary import summary
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from collections import Counter
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_data():
    dataset_l = []
    # transform = transforms.Compose([transforms.Resize((224, 224)),
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    transform = transforms.Compose([transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])

    # MICC-F2000
    # dataset_l.append(datasets.ImageFolder("/home/brechtl/Pictures/Data/MICC/MICC-F2000", transform=transform))
    # MICC-F220
    dataset_l.append(datasets.ImageFolder("/home/brechtl/Pictures/Data/MICC/MICC-F220", transform=transform))
    # CoMoFoD
    # dataset_l.append(datasets.ImageFolder("/home/brechtl/Pictures/Data/CoMoFoD_small_v2/", transform=transform))
    # CASIAv1
    # dataset_l.append(datasets.ImageFolder("/home/brechtl/Pictures/Data/CASIA/CASIA1/", transform=transform))
    # CASIAv2
    # dataset_l.append(datasets.ImageFolder("/home/brechtl/Pictures/Data/CASIA/CASIA2/", transform=transform))
    # GRIP
    # dataset_l.append(datasets.ImageFolder("/home/brechtl/Pictures/Data/GRIP/", transform=transform))

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

    # Calculate the weights to create a balanced dataset
    class_count = [i for i in c.values()]
    class_weights = 1.0 / torch.tensor(class_count, dtype=torch.float16)
    print(f'Class weights: {class_weights}')

    return dataset, class_weights


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

            _, preds = torch.max(yhat.data, 1)

            correct += (preds == targets.flatten().long()).sum().item()
            predicted_labels = np.append(predicted_labels, preds.cpu().detach().numpy().flatten())
            gt_labels = np.append(gt_labels, targets.cpu().detach().numpy().flatten())

    accuracy = 100 * correct / total_data_points

    precision = precision_score(gt_labels, predicted_labels)
    recall = recall_score(gt_labels, predicted_labels)
    f1 = f1_score(gt_labels, predicted_labels)
    print("Evaluation on test set: ")
    print(f'accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1-score: {f1}')

    cf_matrix = confusion_matrix(gt_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(cf_matrix, display_labels=["Forged", "Original"])
    disp.plot()
    plt.savefig("confusion_matrix.png")


def draw_plot(losses, val_losses):
    plt.clf()
    plt.plot(losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.savefig("loss_graph.png", bbox_inches='tight', pad_inches=0)


def visualize_image(model, test_loader):
    images, labels = next(iter(test_loader))
    save_image(images[0], "original.png")

    vgg_pt_prediction = model.vgg_pt.features(images[0].unsqueeze(0))
    vgg_pt_prediction = torch.squeeze(vgg_pt_prediction)
    gray_scale = torch.sum(vgg_pt_prediction, 0)
    gray_scale = gray_scale / vgg_pt_prediction.shape[0]
    gray_scale = gray_scale.detach().numpy()
    plt.imshow(gray_scale)
    plt.savefig("feature_map.png", bbox_inches='tight', pad_inches=0)


def visualize_model(model, test_loader, num_images=6):
    class_names = ["Forged", "Original"]

    with torch.no_grad():
        for x_val, y_val in test_loader:
            x_val = x_val[:num_images].to(torch.float32)
            x_val = x_val.to(DEVICE)
            y_val = y_val[:num_images].to(torch.float32).unsqueeze(-1).flatten().long()
            y_val = y_val.to(DEVICE)
            print(summary(model, x_val[0].unsqueeze(0), show_input=False))

            outputs = model(x_val)
            _, preds = torch.max(outputs.data, 1)
            predicted_labels = [preds[j] for j in range(x_val.size()[0])]

            print("Ground truth:")
            out = torchvision.utils.make_grid(x_val)
            out = out.cpu().numpy().transpose((1, 2, 0))
            plt.axis("off")
            print(out)
            plt.imshow(out)
            print(y_val)
            plt.title([class_names[x] for x in y_val])
            plt.savefig("gt_visualized.png", bbox_inches='tight', pad_inches=0)

            print("Prediction:")
            out = torchvision.utils.make_grid(x_val)
            out = out.cpu().numpy().transpose((1, 2, 0))
            plt.axis("off")
            plt.imshow(out)
            plt.title([class_names[x] for x in predicted_labels])
            plt.savefig("pred_visualized.png", bbox_inches='tight', pad_inches=0)

            break
