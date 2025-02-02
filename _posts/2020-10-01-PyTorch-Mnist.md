---
layout: post
title: PyTorch Basics
subtitle: MNIST classification
thumbnail-img: assets/img/pytorch.png
comments: false
---

<style>
.dropcap {
  margin: 5px 7px -30px 0;
  }
</style>

## Import MNIST

Let's get $7000$ examples of length $784 = 28\times 28$ of flattened grayscale square images of handwritten digits, stacked as
a tuple of $(X,Y)$ with Python shapes $(70000, 784)$ and $(70000,).$

```python
mnist = fetch_openml("mnist_784", return_X_y=True)
```

Convert them to allowed formats and do the train-test $(5:2)$ split.

```python
X = np.float32(mnist[0] / 256)  # (70000, 784)
Y = np.int64(mnist[1])  # (70000,)
x_train, y_train, x_valid, y_valid = X[:50000, :], Y[:50000], X[50000:, :], Y[50000:]
```

Take a look at the first training example!
```python
plt.imshow(x_train[0].reshape((28, 28)), cmap="gray")
```

<center><p><img src="/assets/img/mnist.png" alt="Profile pic" style="width:200px;border:0px solid black" data-toggle="tooltip" data-placement="auto" title="Looks like a '5'"></p></center>

## DataLoader
Some technical steps
1. Convert the data to "torch tensors"
2. Form a DataLoader (to simplify the iteration process over batches)

Notice that no shuffling is required for the validation data

```python
x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
train_ds = TensorDataset(x_train, y_train)
valid_ds = TensorDataset(x_valid, y_valid)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=128)
```
<span style="display:block; height: 0px;"></span>

## Model, Optimizer & Loss
Lets design the NN architecture by subclassing <code class="in"> nn.Module </code>, which manages the network parameters/weights, their gradients, etc. Create a simple linear layer <code class="in"> self.lin = nn.Linear(784, 10) </code> which implicitly is equivalent to creating a $784\times 10$ matrix of weights and a $10$-dim bias vector

```python
self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
self.bias = nn.Parameter(torch.zeros(10))
```
Both parameters are initially random, Xavier initialized, but need to be determined eventually. For now, let's not add any nonlinearities. Also incorporate the forward step, which will multiply the input by the matrix and add the bias, <code class="in">  xb @ self.weights + self.bias </code>. Hence we have

```python
class FeedForward(nn.Module):
  def __init__(self):
    super().__init__()
    self.lin = nn.Linear(784, 10) 

  def forward(self, xb):
    return self.lin(xb) 

model = FeedForward()
```

Pick the optimizer and the learning rate.

```python
opt = optim.Adam(model.parameters(), lr=.001)
```

Specify the loss. <code class="in"> F.cross_entropy </code> combines negative log likelihood loss and log softmax activation and works well for classification purposes.

```python
loss_func = F.cross_entropy 
```
<span style="display:block; height: 0px;"></span>

## Training
Iterate over all training images $15$ times, each time sampling batches from <code class="in">DataLoader</code>. 
For each batch, 
1. Make a prediction <code class="in">model(xb)</code>, which propagates <code class="in">xb</code> forward.
2. Compute the loss <code class="in">loss_func(pred, yb)</code>
3. Propagate backwards:
  - Update the gradients 
  - Optimize the weights, i.e. for each parameter <code class="in">p</code> do <code class="in"> p -= p.grad * lr </code>
  - Reset the gradient back to $0$ so it is ready for the next batch

```python
for epoch in range(15):
  for xb, yb in train_dl:       
    pred = model(xb)
    loss = loss_func(pred, yb) 

    loss.backward() 
    opt.step()      
    opt.zero_grad() 
  print(loss_func(model(xb), yb))
```
<span style="display:block; height: 0px;"></span>

## Validation
Easy to check the validation loss too (the lines <code class="in">model.train()</code> and <code class="in">model.eval()</code> ensure appropriate behavior in more complex cases)

```python
for epoch in range(15):
    model.train()
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

    # Validation
    model.eval()
    with torch.no_grad():
        valid_loss = sum(
            loss_func(model(xb), yb).tolist() for xb, yb in valid_dl
        ) / len(valid_dl)
```
Can also plot both losses over epochs
<center><p><img src="/assets/img/mnist_loss.png" alt="Profile pic" style="width:100%;border:0px solid black" data-toggle="tooltip" data-placement="auto" title="Losses over Epochs"></p></center>
<span style="display:block; height: 0px;"></span>

## Make a prediction

Plot a few validation images

```python
fig = plt.figure(figsize=(3, 3))
rows, cols = 3, 3
for i in range(0, cols * rows):
    fig.add_subplot(rows, cols, i + 1)
    plt.imshow(x_valid[i].reshape((28, 28)), cmap="gray")
plt.show()
print(torch.argmax(model(x_valid[0:9]), axis=1).reshape(3, 3))
```
<span style="display:block; height: 0px;"></span>

<center>
<img src="/assets/img/mnist_test.png" alt="Profile pic" style="width:150px;" data-toggle="tooltip" data-placement="auto"> <a style="padding: 2rem;">vs</a>  $\begin{matrix} [3, 8, 6]\\ [9, 6, 4]\\ [5, 3, 8] \end{matrix}$ <span style="padding: 1rem;"></span>
</center>


# CNN

Can use a Convolutional NN instead. Our inputs are $28\times 28\times 1$, where $1$ is the # of channels (only gray). 
The first two arguments of <code class="in">Conv2d</code> are <code class="in">in_channels=1</code> and <code class="in">out_channels=16</code>. 

A useful formula for keeping track of dimensions is 

$$n_{out} = \frac{n_{in} - k + 2p}{s} + 1$$ 

where $n$ is image's spatial dimension (height or weight).

- <code class="in">Conv2d(1,16,3,2,1)</code>: &nbsp; &nbsp;28x28x1 &nbsp; <i class="fas fa-arrow-alt-circle-right"></i> &nbsp;(28-3+2)/2+1 = 14.5&nbsp;  <i class="fas fa-arrow-alt-circle-right"></i>&nbsp; 14x14x16
- <code class="in">Conv2d(16,16,3,2,1)</code>: &nbsp; 14x14x16 &nbsp; <i class="fas fa-arrow-alt-circle-right"></i> &nbsp;(14-3+2)/2+1 = 7.5&nbsp;  <i class="fas fa-arrow-alt-circle-right"></i>&nbsp; 7x7x16
- <code class="in">Conv2d(16,10,3,2,1)</code>: &nbsp; 7x7x16 &nbsp; <i class="fas fa-arrow-alt-circle-right"></i> &nbsp;(7-3+2)/2+1 = 4&nbsp;  <i class="fas fa-arrow-alt-circle-right"></i>&nbsp; 4x4x10
- <code class="in">avg_pool2d(xb,4)</code>: &nbsp; &nbsp; &nbsp; 4x4x10 &nbsp; <i class="fas fa-arrow-alt-circle-right"></i> &nbsp;1x10

```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        return xb.view(-1, xb.size(1))

model = CNN()
```
<span style="display:block; height: 1rem;"></span>

# <i class="fas fa-code"></i> Full Code for CNN
```python
from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import torch.nn.functional as F

mnist = fetch_openml("mnist_784", return_X_y=True)
X = np.float32(mnist[0] / 256)  # (70000, 784)
Y = np.int64(mnist[1])  # (70000,)
x_train, y_train, x_valid, y_valid = X[:50000, :], Y[:50000], X[50000:, :], Y[50000:]
plt.imshow(x_train[0].reshape((28, 28)), cmap="gray")

x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
train_ds = TensorDataset(x_train, y_train)
valid_ds = TensorDataset(x_valid, y_valid)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=128)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        return xb.view(-1, xb.size(1))


model = CNN()
opt = optim.Adam(model.parameters(), lr=0.001)
loss_func = F.cross_entropy

# Training
for epoch in range(15):
    model.train()
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

    # Validation
    model.eval()
    with torch.no_grad():
        valid_loss = sum(
            loss_func(model(xb), yb).tolist() for xb, yb in valid_dl
        ) / len(valid_dl)
    print(epoch, valid_loss)

# True vs Pred
fig = plt.figure(figsize=(3, 3))
rows, cols = 3, 3
for i in range(0, cols * rows):
    fig.add_subplot(rows, cols, i + 1)
    plt.imshow(x_valid[i].reshape((28, 28)), cmap="gray")  # True
plt.show()
print(torch.argmax(model(x_valid[0:9]), axis=1).reshape(3, 3))  # Pred
```

