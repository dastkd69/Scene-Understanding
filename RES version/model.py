import torch
import torch.nn as nn
import torchvision.models as models

from encoder import EncoderCNN
from decoder import DecoderRNN

    

class RioluNet(nn.Module):
  def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
    super(RioluNet, self).__init__()
    self.encoder = EncoderCNN(embed_size)
    self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

  def forward(self, images, captions):
    features = self.encoder(images)
    outputs = self.decoder(features, captions)
    return outputs

  def captionImage(self, image, vocabulary, maxlength=50):
        result_caption = []
        
        with torch.no_grad():
            x = self.encoder(image).unsqueeze(0)
            states = None
            
            for _ in range(maxlength):
                hiddens, states = self.decoder.lstm(x, states)
                output = self.decoder.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)  
                result_caption.append(predicted.item())
                x = self.decoder.embed(output).unsqueeze(0)
                
                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

        return [vocabulary.itos[idx] for idx in result_caption]

