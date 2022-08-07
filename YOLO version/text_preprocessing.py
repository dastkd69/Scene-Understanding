from PIL import Image
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import os
import matplotlib as plt
import natsort

train_data="/home/dastkd69/Minecraft/storage/val2017"
train_labels="/home/dastkd69/Minecraft/storage/annotations/instances_val2017.json"
max_width = 640
max_height = 640
# train_captions = "/home/dastkd69/Minecraft/storage/annotations/captions_train2017.json"

# test_data="/home/dastkd69/Minecraft/storage/val2017"
# test_labels="/home/dastkd69/Minecraft/storage/annotations/instances_val2017.json"
# test_captions = "/home/dastkd69/Minecraft/storage/annotations/captions_val2017.json"


class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image


transform = transforms.Compose([
                        transforms.Resize([255]),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(), 
                    ])

# train_set = datasets.CocoDetection(root = train_data, annFile = train_labels, transform = transforms.ToTensor())
# train_set = pad_sequence(train_set)
# test_set = datasets.CocoDetection(root = test_data, annFile = test_labels, transform = transforms.ToTensor())

dataset = CustomDataSet(train_data, transform=transform)

# image_set = [ F.pad(img, [0, max_width - img.size(2), 0, max_height - img.size(1)]) for img in next(iter(dataset))]


train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
# test_dataloader = DataLoader(test_set, batch_size=64, shuffle=True)



batch = iter(train_dataloader)

train_features, train_labels = batch.next()
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# img = train_features[0].squeeze()
# label = train_labels[0]
# plt.imshow(img, cmap="gray")
# plt.show()
# print(f"Label: {label}")

# sample_caption = 'A person doing a trick on a rail while riding a skateboard.'
# sample_tokens = nltk.tokenize.word_tokenize(str(sample_caption).lower())
# print(sample_tokens)
# sample_caption = []

# # start_word = data_loader.dataset.vocab.start_word
# # print('Special start word:', start_word)
# sample_tokens.append(nltk.tokenize.word_tokenize(str(caption for caption in dataset).lower())
# print(sample_tokens)
