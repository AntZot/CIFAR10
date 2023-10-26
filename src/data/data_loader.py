import torch
import torchvision
import torchvision.transforms as transforms

class Loader():
    def __init__(self,device="cpu") -> None:
        self.data_shape = ()
        self.batch_size = 4
        self.device = device
        self.classes = None
    

    def get_classes(self) -> list[str]:
        return self.classes 
    

    def allocate_mem(self) -> None:
        alloc = torch.zeros(self.data_shape,device=self.device)
    
    
    def load_train(self,
                root:str = "data/raw/",
                batch_size:int = 4,
                transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]), 
                num_workers: int = 2) -> torch.utils.data.DataLoader:

        train_set = torchvision.datasets.CIFAR10(root,train=True,transform=transform,download=True)

        if not self.data_shape or self.batch_size<batch_size:
            self.data_shape = train_set.data[0:batch_size].shape

        if not self.classes:
            self.get_classes = train_set.classes
        
        return torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=num_workers)


    def load_test(self,
                root:str = "data/raw/",
                batch_size:int = 4,
                transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]), 
                num_workers: int = 2) -> torch.utils.data.DataLoader:
        
        
        test_set = torchvision.datasets.CIFAR10(root,train=False,transform=transform,download=True)

        if not self.data_shape or self.batch_size<batch_size:
            self.data_shape = test_set.data[0:batch_size].shape

        if not self.classes:
            self.get_classes = test_set.classes

        return torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=True,num_workers=num_workers)

