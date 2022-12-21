import numpy as np
import torch
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


writer = SummaryWriter("runs/testCNN9")

Encodings = np.load('trainEncodings.npy')
df = pd.read_csv('training_data.csv', sep=',')

Labels = df['difficulty'].tolist()
le = LabelEncoder()
le.fit(["A1", "A2", "B1", "B2", "C1", "C2"])
Labels = np.array(le.transform(Labels))

trainEncodings, testEncodings = train_test_split(Encodings, test_size=0.2, shuffle=False, random_state=42)
trainLabels, testLabels = train_test_split(Labels, test_size=0.2, shuffle=False, random_state=42)


def DataBatchLoader(batchSize, mode):
    assert mode in ['train', 'test']
    if mode == 'train':
        encodings, labels = trainEncodings, trainLabels
    elif mode == 'test':
        encodings, labels = testEncodings, testLabels
    
    numBatches = len(encodings) // batchSize
    for i in range(numBatches):
        yield encodings[i*batchSize:(i+1)*batchSize], labels[i*batchSize:(i+1)*batchSize]

class MyNN(torch.nn.Module):
    def __init__(self):
        super(MyNN, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=7, out_channels=100, kernel_size=3)
        self.conv2 = torch.nn.Conv1d(in_channels=100, out_channels=100, kernel_size=3)
        self.conv3 = torch.nn.Conv1d(in_channels=100, out_channels=100, kernel_size=3)
        self.pool = torch.nn.MaxPool1d(kernel_size=2, stride=2)
        self.Lin0 = torch.nn.Linear(700, 300)
        self.Lin1 = torch.nn.Linear(300, 100)
        self.Lin2 = torch.nn.Linear(100, 6)
    def forward(self, x):
        # # remove the first 6 features
        x = x[:, :, 6:]
        # # remove the 300 last features
        x = x[:, :, :-300]

        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = torch.nn.functional.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.relu(self.Lin0(x))
        x = torch.nn.functional.relu(self.Lin1(x))
        x = torch.nn.functional.log_softmax(self.Lin2(x), dim=1)
        return x
    


net = MyNN()
device = torch.device('cuda')
net.to(device)
criterion = torch.nn.NLLLoss()

optimizer = torch.optim.Adam(net.parameters(), lr=10**(-5))
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99, verbose=True)

firstBatch = True
batchnum = 0
for epoch in range(1000):
    net.train()
    for batch in DataBatchLoader(batchSize=64, mode='train'):
        optimizer.zero_grad()
        encodings, labels = batch
        encodings = torch.FloatTensor(encodings).to(device)
        labels = torch.tensor(labels).to(device)
        if firstBatch:
            writer.add_graph(net, encodings)
            firstBatch = False
        outputs = net(encodings)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        writer.add_scalar('training loss', loss, batchnum)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / len(labels)
        writer.add_scalar('training accuracy', accuracy, batchnum)
        print('Epoch: {}, Batch: {}, Loss: {}, Accuracy: {}'.format(epoch, batchnum, loss, accuracy))
        batchnum += 1

    with torch.no_grad():
        net.eval()
        accuracies = []
        losses = []
        for batch in DataBatchLoader(batchSize=64, mode='test'):
            encodings, labels = batch
            encodings = torch.FloatTensor(encodings).to(device)
            labels = torch.tensor(labels).to(device)
            labels = labels.long()
            outputs = net(encodings)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            accuracy = correct / len(labels)
            accuracies.append(accuracy)
            losses.append(loss.item())
        writer.add_scalar('test loss', np.mean(losses), batchnum)
        writer.add_scalar('test accuracy', np.mean(accuracies), batchnum)
    