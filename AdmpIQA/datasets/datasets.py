import os
import torch.utils.data as data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import clip
from clip.simple_tokenizer import SimpleTokenizer
from PIL import Image

def load_image(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class ImageQualityDataset(data.Dataset):
    def __init__(self, base_dir, subset_indices, transform=None, num_patches=1):
        self.base_dir = base_dir
        self.transform = transform
        self.num_patches = num_patches

        csv_file_path = os.path.join(base_dir, 'data.csv')
        dataframe = pd.read_csv(csv_file_path)

        self.image_names = dataframe['name'].tolist()
        self.quality_scores = np.array(dataframe['mos_quality']).astype(np.float32)
        self.prompts = dataframe['prompt'].tolist()

        self.samples = self._prepare_samples(subset_indices)

    def _prepare_samples(self, subset_indices):
        object_dict = {}
        for idx, prompt in enumerate(self.prompts):
            if prompt not in object_dict:
                object_dict[prompt] = [idx]
            else:
                object_dict[prompt].append(idx)

        sorted_keys = sorted(object_dict.keys())
        selected_indices = []
        for idx in subset_indices:
            selected_indices.extend(object_dict[sorted_keys[idx]])

        samples = []
        for idx in selected_indices:
            prompt = self.prompts[idx]
            formatted_prompt = self._format_prompt(prompt)
            for _ in range(self.num_patches):
                samples.append((os.path.join(self.base_dir, 'img', self.image_names[idx]),
                                self.quality_scores[idx], formatted_prompt))
        return samples

    def _format_prompt(self, prompt):
        parts = [item.strip() for item in prompt.split(',')]
        if len(parts) == 1:
            return f'This is a photo aligned with the prompt: {parts[0]}.'
        elif len(parts) == 2:
            if 'style' in parts[1]:
                return f'This is a photo aligned with the prompt: {parts[0]}, with style {parts[1]}.'
            else:
                return f'This is a photo aligned with the prompt: {parts[0]}, with detail {parts[1]}.'
        else:
            details = ', '.join(parts[1:-1])
            if 'style' in parts[-1]:
                return f'This is a photo aligned with the prompt: {parts[0]}, with details {details}, and style {parts[-1]}.'
            else:
                return f'This is a photo aligned with the prompt: {parts[0]}, with details {details}.'

    def __getitem__(self, index):
        image_path, quality, prompt = self.samples[index]
        image = load_image(image_path)
        if self.transform:
            image = self.transform(image)
        tokenized_prompt = clip.tokenize(prompt)[0]
        return image, quality / 5.0, tokenized_prompt, image_path

    def __len__(self):
        return len(self.samples)

if __name__ == '__main__':
    dataset_root = '/home/datasets/IQA_Datasets/CSIQ/DATA/'
    subset_indices = [1]
    num_patches_per_image = 1
    dataset = ImageQualityDataset(dataset_root, subset_indices, num_patches=num_patches_per_image)