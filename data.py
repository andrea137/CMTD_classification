import torch
from torchvision import transforms
from imagefolderwithpaths import ImageFolderWithPaths

class Data():
    def __init__(self, image_dir, magnification, fold, batch_size=32,
                 resize_size=256, crop_size=224, num_workers=0, trns='base'):
        self.data_dir = image_dir/fold/magnification
        self.batch_size = batch_size
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.num_workers = min(batch_size, num_workers)

        train_transforms = { 
        'base' : transforms.Compose([
                # Transforms similar to
                #  Araújo et al. (2017) Classification of breast cancer histology images 
                # using Convolutional Neural Networks. 
                # PLoS ONE 12(6): e0177544. https://doi.org/10.1371/journal.pone.0177544
                transforms.Resize(self.resize_size),
                transforms.RandomCrop(self.crop_size),
                transforms.RandomChoice([
                    transforms.Lambda(lambda x: x),
                    transforms.Lambda(lambda x: x.rotate(90)),
                    transforms.Lambda(lambda x: x.rotate(180)),
                    transforms.Lambda(lambda x: x.rotate(270)),
                ]),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]),
                
        'imagenet' : transforms.Compose([
                #transforms.RandomCrop(crop_size),
                transforms.RandomResizedCrop(crop_size),
                transforms.RandomChoice([
                    transforms.Lambda(lambda x: x),
                    transforms.Lambda(lambda x: x.rotate(90)),
                    transforms.Lambda(lambda x: x.rotate(180)),
                    transforms.Lambda(lambda x: x.rotate(270)),
                ]),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
                #ImageNetPolicy(), 
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]),
        }



        # Data augmentation and normalization for training
        # Just normalization for validation
        self.data_transforms = {
            'train': train_transforms[trns],
            
            'val': transforms.Compose([
                transforms.Resize(self.resize_size),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            
            'test': transforms.Compose([    
                transforms.Resize(self.resize_size),
                transforms.TenCrop(self.crop_size), # this is a list of PIL Images
                transforms.Lambda(lambda crops: torch.stack(
                                  [transforms.Compose([transforms.ToTensor(), 
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                   ])(crop) for crop in crops])), # returns a 4D tensor
            ]),
        }

        self.image_datasets = {x: ImageFolderWithPaths(self.data_dir/x,
                                                self.data_transforms[x])
                        for x in ['train', 'val']}
        # In this case test is just val with 10 crop
        self.image_datasets['test'] = ImageFolderWithPaths(self.data_dir/'val',
                                                self.data_transforms['test'])
        
        bShuffle = {'train' : True, 'val' : False, 'test' : False}
        self.dataloaders = {x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size=batch_size,
                                                    shuffle=bShuffle[x], num_workers=num_workers)
                    for x in ['train', 'val', 'test']}
        self.dataset_sizes = {x: len(self.image_datasets[x]) for x in ['train', 'val', 'test']}
