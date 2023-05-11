import torch
from torchvision import models
import os
from torchvision.transforms import transforms, InterpolationMode
from PIL import Image
from numpy import save


def start():
    extract_features()


def extract_features():
    vgg_pt = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    vgg_pt = torch.nn.Sequential(*list(vgg_pt.children())[:-1])
    vgg_pt.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # ORIGINAL images
    original_dir_list = ["/home/brechtl/Pictures/Data/MICC/MICC-F2000/original/",
                         "/home/brechtl/Pictures/Data/MICC/MICC-F220/original/",
                         "/home/brechtl/Pictures/Data/CoMoFoD_small_v2/original/",
                         "/home/brechtl/Pictures/Data/CASIA/CASIA1/original/",
                         "/home/brechtl/Pictures/Data/CASIA/CASIA2/original/"
                         ]

    # FORGED images
    forged_dir_list = ["/home/brechtl/Pictures/Data/MICC/MICC-F2000/forged/",
                       "/home/brechtl/Pictures/Data/MICC/MICC-F220/forged/",
                       "/home/brechtl/Pictures/Data/CoMoFoD_small_v2/forged/",
                       "/home/brechtl/Pictures/Data/CASIA/CASIA1/forged/",
                       "/home/brechtl/Pictures/Data/CASIA/CASIA2/forged/",
                       "/home/brechtl/Pictures/Data/GRIP/forged/"
                       ]

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


def image_features(model, image):
    with torch.no_grad():
        features = model(image)
        features = torch.flatten(features, start_dim=1)
        return features
