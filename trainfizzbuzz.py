import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class fbnet(nn.Module):
    def __init__(self, in_channels, hidden_channels, output_channels):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_channels, hidden_channels),
                                nn.ReLU(),
                                # nn.Dropout(),
                                # nn.Linear(hidden_channels, hidden_channels),
                                # nn.ReLU(),
                                # nn.Dropout(),
                                nn.Linear(hidden_channels, output_channels),
                                nn.LogSoftmax(dim=1))
    def forward(self, x):
        return self.fc(x)


class fbdataset(Dataset):
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys
        assert len(self.xs) == len(self.ys)
    def __getitem__(self, index):
        return torch.FloatTensor(self.xs[index]), self.ys[index]
    def __len__(self):
        return len(self.xs)


def encode(num, digits=10):
    return [num >> i & 1 for i in range(digits)][::-1]

def labelnum(num):
    if (num % 3 == 0) and (num % 5 == 0):
        return 3
    if num % 5 == 0:
        return 2
    if num % 3 == 0:
        return 1
    return 0


model = fbnet(10, 1000, 4)
# model = model.cuda(0)

criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

training_range = range(101, 2**10)

trainx = [encode(i) for i in training_range]
trainy = [labelnum(i) for i in training_range]

traindataset = fbdataset(trainx, trainy)
trainloader = DataLoader(traindataset,
                         shuffle=True,
                         batch_size=128)

testing_range = range(1, 101)
testx = [encode(i) for i in testing_range]
testy = [labelnum(i) for i in testing_range]
testdataset = fbdataset(testx, testy)
testloader = DataLoader(testdataset,
                        batch_size=32)

for n in range(500):
    model.train()
    for data in trainloader:
        optimizer.zero_grad()
        datax, datay = data
        # datax = datax.cuda(0)
        # datay = datay.cuda(0)
        out = model(datax)
        loss = criterion(out, datay)
        loss.backward()
        optimizer.step()
    
    model.eval()
    equality = 0
    with torch.no_grad():
        for data in testloader:
            datax, datay = data
            # datax = datax.cuda(0)
            # datay = datay.cuda(0)
            out = model(datax)
            probs = torch.exp(out)
            pred = torch.argmax(probs, axis=1)
            equality += (pred == datay).sum()

    if n%10 == 0:
        print('epoch %s ... ' %n)
        acc = equality.cpu().numpy()/len(testdataset)
        print('acc on test dataset ... %.f%%' %(acc*100))




