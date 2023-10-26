from torch import nn
import torch
import wandb # not avalable now

class train_model():
    def __init__(self,model:nn.Module, 
                 loss_func:nn.modules.loss,
                 optimizer:torch.optim,
                 device: str = "cpu") -> None:
        
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.device = device


    def train(self,
              trainloader:torch.utils.data.dataloader.DataLoader,
              verbose=False) -> None:
        
        size = len(trainloader.dataset)
        for i,data in enumerate(trainloader):
            inputs,target = data
            inputs = inputs.to(self.device)
            target = target.to(self.device)

            y_pred = self.model.forward(inputs)
            loss = self.loss_func(y_pred,target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if verbose:
                if i % 100 == 0:
                    loss = loss.item()
                    
                    print(f"loss {loss:>7f} [{(i + 1) * len(inputs):>5d}/{size:>5d}]")
    
    def test(self,
             testloader:torch.utils.data.dataloader.DataLoader,
             verbose=False) -> None:
        
        size = len(testloader.dataset)
        num_batches = len(testloader)
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in testloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += self.loss_func(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        if verbose:
            print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    def fit(self,
            trainloader:torch.utils.data.dataloader.DataLoader,
            testloader:torch.utils.data.dataloader.DataLoader,
            epochs:int = 4,
            verbose =False) -> None:
        
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            self.train(trainloader, self.model, self.loss_func, self.optimizer, verbose=verbose)
            self.test(testloader,  self.model, self.loss_func, verbose=verbose)
        print("Done!")

    def save_state(self,path=".../models",name = "model") -> None:
        torch.save(self.model.state_dict(), f"{path}/{name}.pth")