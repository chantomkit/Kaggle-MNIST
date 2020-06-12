import os
import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

filename = "MNIST_model.pt"
# This determines what the program will execute.
# "train" mode is to train the neural network from zero, which will take a lot of time.
# "testAll" mode is to let the neural network goes through all the data in the testset, giving the overall performance.
# "testOne" mode is to pass one random data to the neural network and also displaying that data on screen.
mode = {"train":False, "testAll":False, "testOne": True}

# Getting toy datasets (only set download=True in the first run)
train = datasets.MNIST('', train=True, download=False, transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST('', train=False, download=False, transform=transforms.Compose([transforms.ToTensor()]))

# Divide datasets into batches with shuffling
trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

# Calling and loading the neural network
net = Net()
# print(net)
net.load_state_dict(torch.load(filename))
net.eval()

class trainNet():
    def train(self, mode):
        if mode.get("train") == True:
            loss_function = nn.CrossEntropyLoss()
            optimizer = optim.Adam(net.parameters(), lr=0.001)

            for epoch in range(3): # 3 full passes over the data
                for data in trainset:  # `data` is a batch of data
                    X, y = data  # X is the batch of features, y is the batch of targets.
                    net.zero_grad()  # sets gradients to 0 before loss calc. You will do this likely every step.
                    output = net(X.view(-1,28*28))  # pass in the reshaped batch (recall they are 28x28 atm)
                    loss = F.nll_loss(output, y)  # calc and grab the loss value
                    loss.backward()  # apply this loss backwards thru the network's parameters
                    optimizer.step()  # attempt to optimize weights to account for loss/gradients
                print(loss)  # print loss. We hope loss (a measure of wrong-ness) declines!
            torch.save(net.state_dict(), filename)
        else:
            print("Note that train mode is not activated.")

class testNet():
    def check(self, mode):
        if mode.get("testAll") == True and mode.get("testOne") == True:
            print("Note that both testAll and testOne mode are on, it may cause errors.")
            exit()

    def testAll(self, mode):
        if mode.get("testAll") == True:
            correct = 0
            total = 0
            with torch.no_grad():
                for data in testset:
                    X, y = data
                    output = net(X.view(-1,784))
                    for idx, i in enumerate(output):
                        if torch.argmax(i) == y[idx]:
                            correct += 1
                        total += 1

            print("Accuracy: ", round(correct/total, 3))
            print("Total samples: ", total)
        else:
            print("Note that testAll mode is not activated.")

    def testOne(self, mode):
        if mode.get("testOne") == True:
            with torch.no_grad():
                for data in testset:
                    X, y = data
                    output = net(X.view(-1,784))
                    for idx, i in enumerate(output):
                        print(f"Model Prediction: {torch.argmax(i)}\nLabel: {y[idx]}")
                        plt.imshow(X[0].view(28, 28), cmap='gray')
                        plt.show()
                        break
                    break
        else:
            print("Note that testOne mode is not activated.")

if __name__ == "__main__":
    trainNet.train(Net, mode)
    testNet.check(Net, mode)
    testNet.testAll(Net, mode)
    testNet.testOne(Net, mode)