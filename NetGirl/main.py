import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from net_girl import NetGirl
from net_girl_dataset import NetGirlDataset
from random import randint


loss_list = []
model = NetGirl(3, 3, 1)
dataset = NetGirlDataset()

optimizer = optim.RMSprop(params=model.parameters(), lr=0.01)
loss_func = torch.nn.MSELoss()

model.train()

for _ in range (10):
    model.train()
    for _ in range (100):
        x_train, y_train = dataset[randint(0, dataset.__len__() - 1)]
        y = model(x_train)
        y = y.squeeze()
        loss = loss_func(y, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()

    error = 0.0

    for i in range (dataset.__len__()):
        y = model(dataset.x_train[i])
        error = abs(dataset.y_train[i]) - abs(y.data)
        print(f"Выходное значение НС: {y.data} => {dataset.y_train[i]}")
    loss_list.append(error / dataset.__len__())
    print(f"Среднее отклонение: {error / dataset.__len__()}")

plt.plot(loss_list)
plt.grid()
plt.show()