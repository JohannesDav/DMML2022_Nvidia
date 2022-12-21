import numpy as np
import torch
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


writer = SummaryWriter("runs/testLSTM14")

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
        self.lstm0 = torch.nn.LSTM(input_size=13, hidden_size=200, num_layers=3, batch_first=False, bidirectional=True)
        self.Lin0 = torch.nn.Linear(400, 400)
        self.Lin1 = torch.nn.Linear(400, 200)
        self.Lin2 = torch.nn.Linear(200, 60)
        self.Lin3 = torch.nn.Linear(60, 6)
    def forward(self, x):
        x = x[:,:,:13]
        x, _ = self.lstm0(x)
        x = x[-1]
        x = torch.nn.functional.relu(self.Lin0(x))
        x = torch.nn.functional.relu(self.Lin1(x))
        x = torch.nn.functional.relu(self.Lin2(x))
        x = torch.nn.functional.log_softmax(self.Lin3(x), dim=1)
        return x


net = MyNN()
device = torch.device('cuda')
net.to(device)
criterion = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=10**(-5))
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99, verbose=True)

firstBatch = True
batchnum = 0
for epoch in range(100):
    net.train()
    for batch in DataBatchLoader(batchSize=32, mode='train'):
        optimizer.zero_grad()
        encodings, labels = batch
        encodings = torch.FloatTensor(encodings).to(device)
        encodings = encodings.permute(1, 0, 2)
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
    
    accuracies = []
    losses = []
    with torch.no_grad():
        net.eval()
        for batch in DataBatchLoader(batchSize=128, mode='test'):
            encodings, labels = batch
            encodings = torch.FloatTensor(encodings).to(device)
            encodings = encodings.permute(1, 0, 2)
            labels = torch.tensor(labels).to(device)
            outputs = net(encodings)
            loss = criterion(outputs, labels)
            writer.add_scalar('test loss', loss, batchnum)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            accuracy = correct / len(labels)
            accuracies.append(accuracy)
            losses.append(loss.item())
        writer.add_scalar('test loss', np.mean(losses), batchnum)
        writer.add_scalar('test accuracy', np.mean(accuracies), batchnum)

    lr_scheduler.step()