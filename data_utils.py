from torchvision import transforms, datasets
from torch.utils.data import DataLoader

class Binarize():
    """
    Binarizes the values of some PyTorch tensor.
    Values above the threshold are turned into ones,
    and below said threshold are turned to zero.
    """
    def __init__(self, threshold=0.5):
        self.threshold = threshold
    def __call__(self, tensor):
        return (tensor > self.threshold).type(tensor.type())


def get_MNIST_data(batch_size, threshold=0.5):
    """
    Downloads (if necessary), the MNIST dataset and
    returns the train and test datasets and dataloaders
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         Binarize(threshold)]
    )
    #if binarize[0] == True:
    #    transformations.append(Binarize(binarize[1]))

    train_dataset = datasets.MNIST(
        "./data",
        train=True,
        download=True,
        transform=transform 
    )
    test_dataset = datasets.MNIST(
        "./data",
        train=False,
        download=True,
        transform=transform
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size
    )

    return train_dataset, test_dataset, train_dataloader, test_dataloader


def get_FashionMNIST_data(batch_size, binarize=(True, 0.5)):
    """
    Downloads (if necessary), the MNIST dataset and
    returns the train and test datasets and dataloaders
    """
    transformations = [transforms.ToTensor()]
    if binarize[0]:
        transformations.append(Binarize(binarize[1]))

    train_dataset = datasets.FashionMNIST(
        "./data",
        train=True,
        download=True,
        transform=transformations
    )
    test_dataset = datasets.FashionMNIST(
        "./data",
        train=False,
        download=True,
        transform=transformations
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size
    )

    return train_dataset, test_dataset, train_dataloader, test_dataloader
