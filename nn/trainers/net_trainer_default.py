import os
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from pose_fpn.nn.optim.lr import adjust_lr

class NetTrainerDefault:
    def __init__(self, model, dataset, batch_size, num_workers, shuffle, device="cuda:0", ):
        self.model = model
        self.device = device
        self.model.to(device)
        self.train_loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

    def run(self, checkpoint, epochs, lr, lr_gamma):
        os.makedirs(checkpoint, exist_ok=True)
        # Loss and optimizer
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Epochs
        for epoch in range(epochs):
            lr = adjust_lr(optimizer, epoch, lr_gamma)
            last_loss = None
            for idx, (input, label) in tqdm(desc='\nEpoch: %d/%d | LR: %.8f' % (epoch + 1, epochs, lr),
                                            iterable=enumerate(self.train_loader),
                                            total=len(self.train_loader)):
                inputs, label = input.to(self.device).float(), label.to(self.device).float()
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = loss_fn(outputs, label)
                loss.backward()
                optimizer.step()
                last_loss = loss.data
            print('Epoch {} : loss {}'.format(epoch, last_loss))

    def save(self):
        pass


