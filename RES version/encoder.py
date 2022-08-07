import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torchvision.models as models
from torchvision.io import read_image


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        weights = models.ResNet50_Weights.DEFAULT
    
        #reconstruct model
        self.resnet = models.resnet50(weights = weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
       
        
    def forward(self, input):
        features = self.resnet(input)
        self.resnet.requires_grad_ = False
        return self.dropout(self.relu(features))




