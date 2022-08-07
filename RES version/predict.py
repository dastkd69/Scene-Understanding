from utils import get_loader, preprocess_img
from model import RioluNet
import matplotlib.pyplot as plt
import time

import torch


img = preprocess_img("1.jpg")
plt.imshow(img.permute(1,2,0))
image, dataset = get_loader()
model = RioluNet(256, 256, vocab_size = len(dataset.vocab), num_layers=1)
model.eval()
start = time.time()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# img = img.to(device=device)
# print(model.captionImage(image=img.unsqueeze(1), vocabulary=dataset.vocab))
# end = time.time() - start
# print(f"Inferred in: {end}")
print(img.shape)