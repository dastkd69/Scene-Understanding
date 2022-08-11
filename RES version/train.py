from spacy import load
import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.tensorboard import SummaryWriter

from model import RioluNet
from utils import get_loader, load_checkpoint, save_checkpoint
import os
import yaml



def train(weight):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_model = True
    load_model = False
    if weight in os.listdir("../weights/"):
        load_model = True
    loader, dataset = get_loader()

    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    #HyperParams
    embed_size = config["embed_size"]
    hidden_size = config["hidden_size"]
    vocab_size = len(dataset.vocab)
    num_layers = config["num_layers"]
    num_epochs = config["num_epochs"]
    learning_rate = 3e-4


    #Tensorboad
    writer = SummaryWriter("runs/flickr")
    step = 0

    print("Initialising network.....")
    model = RioluNet(embed_size=embed_size, hidden_size=hidden_size,vocab_size=vocab_size, num_layers=num_layers).to(device=device)
    print("Network initialised!")
    loss_fn = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimiser = optim.Adam(model.parameters(), lr = learning_rate)

    if load_model:
        step = load_checkpoint(torch.load(weight), model, optimiser)

    #Set model to train mode
    model.train()

    #Set up Epoch
    for epoch in range(num_epochs):
        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimiser": optimiser.state_dict(),
                "step": step,
            }
            save_checkpoint(checkpoint)

        
        print(f"Training...{epoch}")
        
        for idx, (imgs, captions) in enumerate(loader):
            imgs = imgs.to(device)
            captions = captions.to(device)
            
            outputs = model(imgs, captions[:-1])
            loss = loss_fn(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))
            
            writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1  

            optimiser.zero_grad()        
            loss.backward(loss)
            optimiser.step()
        
        print(f"Loss for epoch {epoch}: {loss}")

if __name__ == "__main__":
    train("../weights/flickr_resnet.pth.tar")



