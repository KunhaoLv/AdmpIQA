import torch
import torchvision
from torchvision import transforms

from datasets.datasets import ImageQualityDataset


class ImageQualityDataLoader:
    """Data loader class for IQA databases"""
    def __init__(self, dataset_name, dataset_path, subset_indices, patch_num=1, batch_size=1, is_train=True):
        print(f"Loading dataset: {dataset_name}")
        self.batch_size = batch_size
        self.is_train = is_train

        # Define transformations
        if self.is_train:
            self.transform = transforms.Compose([
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        # Load the dataset
        if dataset_name == 'CSIQ':
            self.dataset = ImageQualityDataset(base_dir=dataset_path, subset_indices=subset_indices, 
                                               transform=self.transform, num_patches=patch_num)
        else:
            raise ValueError("Unsupported dataset specified!")

    def get_data_loader(self):
        """Returns the data loader for the dataset"""
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.is_train)

if __name__ == '__main__':
    # Example usage
    dataset_name = 'CSIQ'
    dataset_path = '/home/datasets/IQA_Datasets/CSIQ/DATA/'
    subset_indices = [1]  # Example subset indices
    patch_num = 1
    batch_size = 1
    is_train = True

    data_loader = ImageQualityDataLoader(dataset_name, dataset_path, subset_indices, patch_num, batch_size, is_train)
    dataloader = data_loader.get_data_loader()

    # Example iteration over the dataloader
    for batch in dataloader:
        images, qualities, prompts, paths = batch
        print(images.shape, qualities, prompts, paths)