import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import random_split

def make_dataset(option,img_size):# 1.4.1確認済
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if option.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='../data',download=True,transform=transform,train=True )
        testset  = torchvision.datasets.CIFAR10(root='../data',download=True,transform=transform,train=False)

    elif option.dataset == 'stl10':
        trainset = torchvision.datasets.STL10(root='../data',download=True,transform=transform,split='train+unlabeled',)
        testset  = torchvision.datasets.STL10(root='../data',download=True,transform=transform,split='test',)
    
    testset,validset = random_split(testset, [int(len(testset)/2),len(testset) - int(len(testset)/2)])

    if option.run_mode == 'debug':
        trainset,_ = random_split(trainset, [30,len(trainset) - 30])
        testset ,_ = random_split( testset, [30,len( testset) - 30])
        validset,_ = random_split(validset, [30,len(validset) - 30])

    trainloader = torch.utils.data.DataLoader(
        trainset,batch_size=option.train_batch_size,shuffle=True,num_workers=option.num_workers,pin_memory=True,drop_last=True
    )
    testloader = torch.utils.data.DataLoader(
        testset ,batch_size=option.test_batch_size,shuffle=False,num_workers=option.num_workers,pin_memory=True,drop_last=True
    )
    validloader = torch.utils.data.DataLoader(
        validset,batch_size=option.test_batch_size,shuffle=False,num_workers=option.num_workers,pin_memory=True,drop_last=True
    )

    return trainloader,testloader,validloader


