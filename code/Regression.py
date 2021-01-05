import torch
from IPython import embed
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())

x,y = Variable(x), Variable(y)



class Net(torch.nn.Module):
    def __init__(self,n_features,n_hiddens,n_outputs):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_features,n_hiddens)
        self.predict = torch.nn.Linear(n_hiddens,n_outputs)

    def forward(self,x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

net = Net(1,10,1)

print(net)
optimizer = torch.optim.Adam(net.parameters(),lr=0.2)
loss_func = torch.nn.MSELoss()

for t in range(200):
    prediction = net(x)
    loss = loss_func(prediction,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(loss)
plt.plot(x,prediction.detach().numpy(),color='red')
plt.scatter(x.data.numpy(),y.data.numpy(),color='blue')
plt.show()