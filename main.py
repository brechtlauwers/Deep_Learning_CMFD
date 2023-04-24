import cv2
from statistical.surf import SurfExtractor
from statistical.matching import FeatureMatching
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def main():
    deep_learning()


def statistical():
    image = cv2.imread("/home/brechtl/Pictures/Data/CoMoFoD_small_v2/029_F_JC9.jpg")

    surf = SurfExtractor(image)
    kp, desc = surf.extract_features()
    matching = FeatureMatching(kp, desc)
    p1, p2 = matching.match()

    if p1 is None:
        print("No tampering was found")
        return False

    clusters, p1, p2 = matching.hierarchical_clustering(p1, p2, 'ward', 2.2)

    if len(clusters) == 0 or len(p1) == 0 or len(p2) == 0:
        print("No tampering was found")
        return False

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    matching.plot_image(image, p1, p2, clusters)


def deep_learning():
    gt = []
    with open("/home/brechtl/Pictures/Data/MICC/MICC-F2000/groundtruthDB_2000.txt") as GT_file:
        for line in GT_file:
            gt += [line.split()]

    transform = transforms.Compose([transforms.Resize(300),
                                    transforms.ToTensor()])

    dataset = datasets.ImageFolder("/home/brechtl/Pictures/Data/MICC/MICC-F2000", transform=transform)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


if __name__ == '__main__':
    main()
