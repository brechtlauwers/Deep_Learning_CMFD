import torch
from sklearn.cluster import MiniBatchKMeans
from torchvision import models
import os
from torchvision.transforms import transforms, InterpolationMode
from PIL import Image
from numpy import save, load
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,\
    confusion_matrix, ConfusionMatrixDisplay
import cv2
from statistical.sift import SiftExtractor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from statistical.matching import FeatureMatching
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# ORIGINAL images
original_dir_list = [# "/home/brechtl/Pictures/Data/MICC/MICC-F2000/original/",
                     "/home/brechtl/Pictures/Data/MICC/MICC-F220/original/",
                     # "/home/brechtl/Pictures/Data/CoMoFoD_small_v2/original/",
                     # "/home/brechtl/Pictures/Data/CASIA/CASIA1/original/",
                     # "/home/brechtl/Pictures/Data/CASIA/CASIA2/original/"
                     ]

# FORGED images
forged_dir_list = [# "/home/brechtl/Pictures/Data/MICC/MICC-F2000/forged/",
                   # "/home/brechtl/Pictures/Data/MICC/MICC-F220/forged/",
                   # "/home/brechtl/Pictures/Data/CoMoFoD_small_v2/forged/",
                   # "/home/brechtl/Pictures/Data/CASIA/CASIA1/forged/",
                   # "/home/brechtl/Pictures/Data/CASIA/CASIA2/forged/",
                   # "/home/brechtl/Pictures/Data/GRIP/forged/"
                   ]


def start():
    # extract_features_vgg()
    features_list = extract_features_sift()
    # features_list = load("sift_features.npy", allow_pickle=True)
    # features_list = load("vgg16_features.npy", allow_pickle=True)
    labels = load("labels.npy", allow_pickle=True)
    # features_list = features_list.reshape((features_list.shape[0], features_list.shape[2]))

    kmeans = MiniBatchKMeans(n_clusters=100)
    kmeans.fit(np.concatenate(features_list))

    encoded_features = np.zeros((len(features_list), 100))
    for i, desc in enumerate(features_list):
        label = kmeans.predict(desc)
        hist, _ = np.histogram(label, bins=100, range=(0, 100))
        encoded_features[i] = hist

    # scaler = StandardScaler()
    # features_list = scaler.fit_transform(features_list)

    X_train, X_test, y_train, y_test = train_test_split(encoded_features, labels[:encoded_features.shape[0]],
                                                        test_size=0.2, random_state=42, shuffle=True)

    # X_train, X_test, y_train, y_test = train_test_split(features_list, labels[:features_list.shape[0]],
    #                                                     test_size=0.2,  shuffle=True)

    rf = RandomForestClassifier(n_estimators=300,
                                min_samples_split=2,
                                min_samples_leaf=4,
                                max_depth=5,
                                class_weight="balanced",
                                verbose=2)

    # rf = RandomForestClassifier(class_weight="balanced",
    #                             random_state=42,
    #                             verbose=2)

    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Accuracy: {:.2f}%".format(accuracy * 100))
    print("Precision: {:.2f}%".format(precision * 100))
    print("Recall: {:.2f}%".format(recall * 100))
    print("F1 Score: {:.2f}%".format(f1 * 100))

    cf_matrix = confusion_matrix(y_test, y_pred)
    print(cf_matrix)

    disp = ConfusionMatrixDisplay(cf_matrix, display_labels=["Forged", "Original"])
    disp.plot()
    plt.savefig("confusion_matrix_rf.png")

    # param_grid = {
    #     'n_estimators': [100, 200, 300],
    #     'max_depth': [5, 10, 15, 20],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4]
    # }
    #
    # rf = RandomForestClassifier(class_weight="balanced")
    # grid_search = RandomizedSearchCV(rf, param_distributions=param_grid, cv=5, scoring='f1', verbose=2)
    # grid_search.fit(X_train, y_train)
    # print("Best hyperparameters: ", grid_search.best_params_)
    # print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))


def extract_features_vgg():
    vgg_pt = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    # vgg_pt = torch.nn.Sequential(*list(vgg_pt.children())[:-1])
    vgg_pt.classifier = torch.nn.Sequential()
    vgg_pt.eval()

    transform = transforms.Compose([
        # transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    feature_list = []
    label_list = []

    for directory in original_dir_list:
        for filename in os.listdir(directory):
            image_path = os.path.join(directory + filename)
            image = Image.open(image_path).convert("RGB")
            input_tensor = transform(image)
            features = image_features(vgg_pt, input_tensor.unsqueeze(0))
            feature_list.append(features.numpy())
            label_list.append(0)
        print("Done -> " + directory)

    for directory in forged_dir_list:
        for filename in os.listdir(directory):
            image_path = os.path.join(directory + filename)
            image = Image.open(image_path).convert("RGB")
            input_tensor = transform(image)
            features = image_features(vgg_pt, input_tensor.unsqueeze(0))
            feature_list.append(features.numpy())
            label_list.append(1)
        print("Done -> " + directory)

    save('vgg16_features.npy', feature_list)
    save('labels.npy', label_list)


def extract_features_sift():
    feature_list = []
    for directory in original_dir_list:
        for filename in os.listdir(directory):
            image = os.path.join(directory, filename)
            image = cv2.imread(image)
            sift = SiftExtractor(image)
            kp, desc = sift.extract_features()
            if desc is not None:
                feature_list.append(desc)
            print(filename)
        print("Done -> " + directory)

    for directory in forged_dir_list:
        for filename in os.listdir(directory):
            image = os.path.join(directory, filename)
            image = cv2.imread(image)
            sift = SiftExtractor(image)
            kp, desc = sift.extract_features()
            if desc is not None:
                feature_list.append(desc)
            print(filename)
        print("Done -> " + directory)

    features_list = np.array(feature_list, dtype=object)
    # features_list = features_list.astype(int)
    # features_list = np.asarray(feature_list, dtype=object)
    # save('sift_features.npy', features_list)
    print("Successfully saved!")
    return features_list


def image_features(model, image):
    with torch.no_grad():
        features = model(image)
        features = torch.flatten(features, start_dim=1)
        return features
