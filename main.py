import cv2

from statistical.surf import SurfExtractor
from statistical.matching import FeatureMatching
import deep_learning.vgg16 as vgg16

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def main():
    # statistical()
    deep_learning()


def statistical():
    image = cv2.imread("/home/brechtl/Pictures/Data/CoMoFoD_small_v2/029_F_JC9.jpg")

    surf = SurfExtractor(image)
    kp, desc = surf.extract_features()
    print(desc)
    print(desc.shape)
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
    # model = vgg16.VGG()
    # loss_fn = nn.BCELoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001)  # momentum=0.9
    # vgg16.make_train_step(model, loss_fn, optimizer)
    vgg16.train()


if __name__ == '__main__':
    main()
