import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim

train_dataframe = pd.read_csv('./dados_competicao/train.csv')
test = pd.read_csv('./dados_competicao/test.csv').values / 255
train_y = train_dataframe['label'].values
train_x = train_dataframe.drop(columns=["label"]).values / 255

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

train_x_tensor = torch.tensor(train_x, dtype=torch.float32)
train_x_tensor = train_x_tensor.view(-1, 1, 28, 28).to(device)
train_y_tensor = torch.tensor(train_y, dtype=torch.long).to(device)
test_tensor = torch.tensor(test, dtype=torch.float32)
test_tensor = test_tensor.view(-1, 1, 28, 28).to(device)

dataset = TensorDataset(train_x_tensor, train_y_tensor)
#trainloader com o conjunto de treino disponibilizado pela competição
trainloader = DataLoader(dataset, batch_size=64, shuffle=True)

#data augmentation
transform_MNIST = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor()
])
trainset_MNIST = datasets.MNIST(root='./trainset_MNIST', train=True, download=True, transform=transform_MNIST)
#trainloader com o conjunto de treino do torchvision (mais imagens)
trainloader_MNIST = torch.utils.data.DataLoader(trainset_MNIST, batch_size=64, shuffle=True)

def train(model, trainloader, epochs):
    lossFunction = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), 0.001)

    for e in range(epochs):
        print(f"EPOCA [{e+1}/{epochs}]")
        for imagens, rotulos in trainloader:
            imagens = imagens.to(device)
            rotulos = rotulos.to(device)

            output = model(imagens)
            loss = lossFunction(output, rotulos)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

from CNN import CNN

model = CNN().to(device)
model.train()

train(model, trainloader_MNIST, epochs=40)

model.eval()

acertos = 0
with torch.no_grad():
    for imagens, rotulos in trainloader:
        output = model(imagens)
        predict = torch.argmax(output, dim=1)
        acertos += (predict == rotulos).sum().item()

print("ACURACIA (treino) =", acertos/train_x.shape[0])

transform_MNIST = transforms.Compose([transforms.ToTensor()])
testset_MNIST = datasets.MNIST(root='./testset_MNIST', train=False, download=True, transform=transform_MNIST)
testloader_MNIST = torch.utils.data.DataLoader(testset_MNIST, batch_size=500, shuffle=False)

acertos = 0
with torch.no_grad():
    for imagens, rotulos in testloader_MNIST:
        imagens = imagens.to(device)
        rotulos = rotulos.to(device)
        output = model(imagens)
        predict = torch.argmax(output, dim=1)
        acertos += (predict == rotulos).sum().item()

print("ACURACIA (teste) =", acertos/10000)

""" Criando o arquivo de submissão """

predictions = list()
ids = list()

with torch.no_grad():
    for i in range(test_tensor.shape[0]):
        img = test_tensor[i]
        img = img.unsqueeze(0)
        output = model(img)
        predictions.append(torch.argmax(output, dim=1).item())
        ids.append(i+1)

results = pd.DataFrame({"ImageId": ids, "Label": predictions})
results.to_csv("submission.csv", index=False)
