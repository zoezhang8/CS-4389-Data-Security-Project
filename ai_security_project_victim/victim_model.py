import torch.nn as nn
 #imports PyTorch's neural network toolbox
 
class VictimNet(nn.Module):
    """784 -> 128 -> 64 -> 10 fully connected network.
 
    Outputs raw scores (logits). Apply softmax if you need probabilities.
    """
 #gives PyTorch's built in abilities: tracking its learnable numbers and saving/loading them.

    def __init__(self):
        super().__init__()
        #setup function, runs once model is created.
        self.net = nn.Sequential( #runs these layers in order
            nn.Flatten(),          # 1x28x28 image -> 784 numbers
            nn.Linear(784, 128),   # takes 784 input numbers and mixes them into 128 new numbers
            nn.ReLU(),
            nn.Linear(128, 64),    # output 10 numbers 
            nn.ReLU(),
            nn.Linear(64, 10),     # one score per digit 0-9
        )
 
    def forward(self, x):
        return self.net(x)