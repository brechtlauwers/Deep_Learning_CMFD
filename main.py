import cv2
import os

from statistical.sift import SiftExtractor
from statistical.matching import FeatureMatching
import deep_learning.vgg16 as vgg16
import deep_learning.random_forest as rf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay


def main():
    # statistical()
    # deep_learning()
    random_forest()


def statistical():
    # FORGED images
    forged_dir_list = [# "/home/brechtl/Pictures/Data/MICC/MICC-F2000/forged/",
                       # "/home/brechtl/Pictures/Data/MICC/MICC-F220/forged/",
                       # "/home/brechtl/Pictures/Data/CoMoFoD_small_v2/forged/",
                       # "/home/brechtl/Pictures/Data/CASIA/CASIA1/forged/",
                       # "/home/brechtl/Pictures/Data/CASIA/CASIA2/forged/",
                       # "/home/brechtl/Pictures/Data/GRIP/forged/"
                       ]
    correct = 0
    predicted_labels, gt_labels = [], []
    total_data_points = 0
    counter = 0

    for directory in forged_dir_list:
        total_data_points += len(os.listdir(directory))
        for filename in os.listdir(directory):
            image = os.path.join(directory + filename)
            if os.path.exists(image):
                image = cv2.imread(image)
                sift = SiftExtractor(image)
                kp, desc = sift.extract_features()
            else:
                kp, desc = None, None

            if desc is not None:
                matching = FeatureMatching(kp, desc)
                forgery = matching.locateForgery(image)
            else:
                forgery = None

            counter += 1

            if forgery is None:
                print(f"No forgery found, {counter}/{total_data_points}")
                pred = 0
            else:
                print(f"Forgery found, {counter}/{total_data_points}")
                pred = 1

            correct += (pred == 1)
            predicted_labels = np.append(predicted_labels, pred)
            gt_labels = np.append(gt_labels, 1)



    # ORIGINAL images
    original_dir_list = ["/home/brechtl/Pictures/Data/MICC/MICC-F2000/original/",
                         # "/home/brechtl/Pictures/Data/MICC/MICC-F220/original/",
                         # "/home/brechtl/Pictures/Data/CoMoFoD_small_v2/original/",
                         # "/home/brechtl/Pictures/Data/CASIA/CASIA1/original/",
                         # "/home/brechtl/Pictures/Data/CASIA/CASIA2/original/",
                         # "/home/brechtl/Pictures/Data/GRIP/original/"
                         ]

    for directory in original_dir_list:
        total_data_points += len(os.listdir(directory))
        for filename in os.listdir(directory):
            image = os.path.join(directory + filename)

            if os.path.exists(image):
                image = cv2.imread(image)
                sift = SiftExtractor(image)
                kp, desc = sift.extract_features()
            else:
                kp, desc = None, None

            if desc is not None:
                matching = FeatureMatching(kp, desc)
                forgery = matching.locateForgery(image)
            else:
                forgery = None

            counter += 1

            if forgery is None:
                print(f"No forgery found, {counter}/{total_data_points}")
                pred = 0
            else:
                print(f"Forgery found, {counter}/{total_data_points}")
                pred = 1

            correct += (pred == 0)
            predicted_labels = np.append(predicted_labels, pred)
            gt_labels = np.append(gt_labels, 0)


    accuracy = 100 * correct / total_data_points

    precision = precision_score(gt_labels, predicted_labels)
    recall = recall_score(gt_labels, predicted_labels)
    f1 = f1_score(gt_labels, predicted_labels)
    print("Evaluation metrics: ")
    print(f'accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1-score: {f1}')

    cf_matrix = confusion_matrix(gt_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(cf_matrix, display_labels=["Forged", "Original"])
    disp.plot()
    plt.savefig("sift_confusion_matrix.png")


def random_forest():
    rf.start()


def deep_learning():
    vgg16.start()


if __name__ == '__main__':
    main()

