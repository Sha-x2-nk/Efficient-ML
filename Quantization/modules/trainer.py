import torch
import torch.nn as nn
import torch.nn.functional as F



def train_and_test(model, train_loader, test_loader, optimizer, device, epochs, callbacks=None):
  model.to(device)
  
  # training
  for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch, callbacks)
    accuracy = test(model, device, test_loader)
    print(f'Epoch {epoch}: Test Accuracy (No Quantization): {accuracy:.2f}%')

def train(model, device, train_loader, optimizer, epoch, callbacks = None):
  model.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = F.cross_entropy(output, target)
    loss.backward()
    optimizer.step()

    if callbacks is not None:
      for callback in callbacks:
        callback()

def test(model, device, test_loader):
  model.eval()
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      pred = output.argmax(dim=1, keepdim=True)
      correct += pred.eq(target.view_as(pred)).sum().item()
  accuracy = 100. * correct / len(test_loader.dataset)
  return accuracy