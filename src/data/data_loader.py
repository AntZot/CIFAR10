import torch
import torchvision
import torchvision.transforms as transforms



def load_train(root:str = "data/raw/",
               batch_size:int = 4,
               transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]), 
               num_workers: int = 2) -> torch.utils.data.DataLoader:

    train_set = torchvision.datasets.CIFAR10(root,train=True,transform=transform,download=True)

    return torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=num_workers)

def load_test(root:str = "data/raw/",
               batch_size:int = 4,
               transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]), 
               num_workers: int = 2) -> torch.utils.data.DataLoader:
    
    test_set = torchvision.datasets.CIFAR10(root,train=False,transform=transform,download=True)

    return torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=True,num_workers=num_workers)