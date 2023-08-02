#import cv2
import torchvision
import torch
#import torchvision.transforms as transforms

# Albumentations for augmentations

import albumentations as A
from albumentations.pytorch import ToTensorV2


train_transforms = A.Compose([

                                # A.Normalize(mean = (0.49, 0.48, 0.44),std = (0.24, 0.24, 0.26),p =1.0),
                                # A.HorizontalFlip(p = 0.2),
                                # A.PadIfNeeded(min_height = 40, min_width = 40),
                                # A.RandomResizedCrop(height=32,width=32), 
                                # #A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5),
                                # A.CoarseDropout(max_holes = 1, max_height=8, max_width=8, min_holes = 1, min_height=8, min_width=8,fill_value=0.4734, p = 0.2),
                                # #A.RandomRotate90(),
                                # #A.Blur(blur_limit=3),
                                # #A.OpticalDistortion(),
                                # #A.Rotate(15),
                                # ToTensorV2()

                                A.Normalize(mean = (0.49, 0.48, 0.44),std = (0.24, 0.24, 0.26),p =1.0),
                                A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),
                                A.RandomCrop(height=32, width=32, always_apply=True),
                                A.HorizontalFlip(),
                                A.CoarseDropout(max_holes=1, max_height=8, max_width=8, min_holes=1, min_height=8, min_width=8, fill_value=[0.4914, 0.4822, 0.4465], p =.50),
                                ToTensorV2(),

                            ]

                            

                            )

test_transform = A.Compose([
                                A.Normalize(mean = (0.49, 0.48, 0.44),std = (0.24, 0.24, 0.26),p =1.0),
                                ToTensorV2()

                            ]

                            

                            )

class Cifar10Dataset(torchvision.datasets.CIFAR10):

    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):

        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):

        image, label = self.data[index], self.targets[index]

        if self.transform is not None:

            transformed = self.transform(image=image)

            image = transformed["image"]

        return image, label



train_data = Cifar10Dataset(root='../data', train=True, download=False, transform=train_transforms)

test_data = Cifar10Dataset(root='../data', train=False,download=False, transform=test_transform)


dataloader_args = dict(shuffle=True, batch_size=512, num_workers=2, pin_memory=True)                                      

# train dataloader
train_loader = torch.utils.data.DataLoader(train_data, **dataloader_args)

test_loader = torch.utils.data.DataLoader(test_data, **dataloader_args)
