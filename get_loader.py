import os
import pandas as pd
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms


# Load the English language model for SpaCy
spacy_model = spacy.load("en")


class VocabularyBuilder:
    def __init__(self, min_frequency):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.min_frequency = min_frequency

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenize_text(text):
        return [token.text.lower() for token in spacy_model.tokenizer(text)]

    def construct_vocabulary(self, sentences):
        word_freqs = {}
        index = 4

        for sentence in sentences:
            for word in self.tokenize_text(sentence):
                word_freqs[word] = word_freqs.get(word, 0) + 1
                if word_freqs[word] >= self.min_frequency:
                    if word not in self.stoi:
                        self.stoi[word] = index
                        self.itos[index] = word
                        index += 1

    def convert_to_indices(self, text):
        tokens = self.tokenize_text(text)
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokens]


class ImageCaptionDataset(Dataset):
    def __init__(self, image_directory, caption_file, transform=None, min_word_freq=5):
        self.image_directory = image_directory
        self.annotations = pd.read_csv(caption_file)
        self.transform = transform

        self.image_ids = self.annotations["image"]
        self.captions = self.annotations["caption"]

        self.vocab = VocabularyBuilder(min_word_freq)
        self.vocab.construct_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        image_id = self.image_ids[idx]
        image = Image.open(os.path.join(self.image_directory, image_id)).convert("RGB")

        if self.transform:
            image = self.transform(image)

        tokenized_caption = [self.vocab.stoi["<SOS>"]]
        tokenized_caption += self.vocab.convert_to_indices(caption)
        tokenized_caption.append(self.vocab.stoi["<EOS>"])

        return image, torch.tensor(tokenized_caption)


class PaddingCollate:
    def __init__(self, padding_index):
        self.padding_index = padding_index

    def __call__(self, batch):
        images = [item[0].unsqueeze(0) for item in batch]
        images = torch.cat(images, dim=0)
        captions = [item[1] for item in batch]
        padded_captions = pad_sequence(captions, batch_first=False, padding_value=self.padding_index)

        return images, padded_captions


def create_data_loader(
    image_folder,
    annotations_file,
    transform,
    batch_size=32,
    num_workers=8,
    shuffle=True,
    pin_memory=True,
):
    dataset = ImageCaptionDataset(image_folder, annotations_file, transform=transform)

    pad_idx = dataset.vocab.stoi["<PAD>"]

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=PaddingCollate(padding_index=pad_idx),
    )

    return data_loader, dataset


if __name__ == "__main__":
    image_transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    loader, dataset = create_data_loader(
        "flickr8k/images/", "flickr8k/captions.txt", transform=image_transform
    )

    for idx, (images, captions) in enumerate(loader):
        print(images.shape)
        print(captions.shape)
