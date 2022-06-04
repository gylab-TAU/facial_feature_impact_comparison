import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from data_prep.Datasets.image_and_text_dataset import ImageAndTextDataset
from modelling.models.context_vgg16 import context_vgg16

# PATH_IMAGE_FOLDER_FOR_CHECKING = "raw_test_dataset"
# PATH_CSV = "vector_test.csv"
PATH_IMAGE_FOLDER_FOR_CHECKING = "/home/context/dataset/tmptest"
PATH_CSV = "/home/context/dataset/classes_with_occupation_vectors_test.csv"

# simple running to see if the code in context_vgg16 make sense:

if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.Resize(size=(224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = ImageAndTextDataset(path=PATH_IMAGE_FOLDER_FOR_CHECKING,
                                  transforms=transform,
                                  target_transforms=None,
                                  vector_csv_path=PATH_CSV)
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=2,
        shuffle=True)
    net = context_vgg16(True, True, num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(10):
        for images, labels, context_vectors in dataloader:
            optimizer.zero_grad()
            outputs = net(images, context_vectors)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(loss.item())