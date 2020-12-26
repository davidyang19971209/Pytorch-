import torch
import numpy as np
from torch.autograd import Variable

data = [[1,2],[3,4]]
tensor = torch.FloatTensor(data)

# print(np.matmul(data,data))
# print('\nnumpy:',np.matmul(data,data),'\nnumpy:',np.matmul(data,data))

variable = Variable(tensor,requires_grad=True)

print(tensor)
print(variable)

v_out = torch.mean(variable*variable)

v_out.backward()

print(variable)
print(variable.data)

