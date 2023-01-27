import torch
import os
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from PIL import Image
from torch.utils.data import DataLoader
import torch.distributed as dist

path = "/root/autodl-tmp/imagenet/"

def read_from_txt(type="meta/val.txt"):
    """return:{"img_name": "label"}
    注意label是int的格式
    """
    file = {}
    with open(path + type, "r") as f:
        data = f.readlines()
    for i in range(len(data)):
        for idx, cha in enumerate(data[i]):
            if cha == " ":
                file.update({data[i][:idx]:data[i][idx+1:-1]})
            else:
                pass
    return file

def label2onehot(file, num_classes=1000):
    """把read_from_txt()函数得到的file文件里的label变成独热码的形式
    args: 参考read_from_txt()函数
    return: label_onehot
    """
    label = file.values()
    len_label = len(label)
    label_list = list(label)
    for idx, i in enumerate(label_list):
        label_list[idx] = int(label_list[idx])
    
    label_onehot = torch.zeros((len_label, 1000))
    for idx, i in enumerate(label_list):
        label_onehot[idx, i-1] = 1
    return label_onehot

norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
trans_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    norm
])

trans_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    norm
])

class MyImageNet(Dataset):
    def __init__(self, types="val", transform=trans_train, anno_transform=None, path=path) -> None:
        super().__init__()
        self.file = read_from_txt(type="meta/" + types + ".txt")
        self.type = types
        self.path = path
        self.img_list = list(self.file.keys())
        self.label_onehot = label2onehot(self.file)
        self.transform = transform
        self.anno_transform = anno_transform
    
    def __len__(self):
        return len(self.file)
    
    def __getitem__(self, index):
        img_path = self.path + self.type + "/" + self.img_list[index]
        img = Image.open(img_path).convert("RGB")
            
        label = self.label_onehot[index]
        
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = img
            
        if self.anno_transform is not None:
            label = self.anno_transform(label)
        else:
            label = label
        return img, label

torch.cuda.set_device(0)
dist.init_process_group('nccl', init_method="env://")

train_dataset = MyImageNet(types='train', transform=trans_train)
test_dataset = MyImageNet(types="test", transform=trans_test)

train_sample = DistributedSampler(train_dataset)
test_sample = DistributedSampler(test_dataset)

train_dataloader = DataLoader(train_dataset, batch_size=64, pin_memory=True, num_workers=12, sampler=train_sample)
test_dataloader = DataLoader(test_dataset, batch_size=64, pin_memory=True, num_workers=12, sampler=test_sample)