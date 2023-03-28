import os
import torch
import torch.utils.data as data
import torch.nn as nn
import numpy as np
from torchvision.models import resnet34
from PIL.Image import Image
import random

# 数据预取
class data_prefetcher(): 
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input = next(self.loader)
        except StopIteration:
            self.next_input = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        self.preload()
        return input
    
class Inferencedataset(data.Dataset):
    def __init__(self, libraryfile='', transform=None, **kwargs):
        lib = torch.load(libraryfile, map_location='cpu')
        slides = []
        wsidirs = []
        for i,name in enumerate(lib['slides']):
            wsiname = os.path.basename(name) # return filename
            wsiname = wsiname.split('.')[0] # 去后缀
            # wsiname,_ = os.path.splitext(wsiname)
            wsidir = os.path.join(args.patch_dir,wsiname)
            wsidirs.append(wsidir)
        print('')

        grid = []
        slideIDX = []
        for i,g in enumerate(lib['grid']):
            grid.extend(g)
            slideIDX.extend(i*len(g))

        print('number of tiles: {}'.format(len(grid)))
        self.slidenames = lib['slides']
        self.slides = slides
        self.wsidirs = wsidirs
        self.targets = lib['targets']
        self.grid = grid
        self.slideIDX = slideIDX
        self.transform = transform
        self.level = 0
        self.size = 256

    def savetopndata(self, idxs, filename):
        slides = []
        grids = []
        targets = []
        topngrid = [self.grid[x] for x in idxs]
        topnid = [self.slideIDX[x] for x in idxs]
        topngrid = np.array(topngrid)
        topnid = np.array(topnid)
        for i in range(len(self.slidenames)):
            slides.append(self.slidenames[i])
            grid = topngrid[topnid==i]
            grid = grid.tolist()
            grids.append(grid)
            targets.append(self.targets[i])
        torch.save({
            'slides': slides,
            'grid': grids,
            'gridIDX': list(topnid),
            'targets': targets},
            os.path.join(args.output, f'{filename}.ckpt'))
            
    def maketraindata(self, idxs):
        self.t_data = [(self.slideIDX[x],self.grid[x],self.targets[self.slideIDX[x]]) for x in idxs]
        return self.t_data
        
    def __getitem__(self,index):
        slideIDX = self.slideIDX[index]
        coord = self.grid[index]
        wsidir = self.wsidirs[slideIDX]
        img_path = os.path.join(wsidir, f"{coord[0]}_{coord[1]}.jpg")
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img
    def __len__(self):
        return len(self.grid)

class Traindataset(data.Dataset):
    def __init__(self, data, libraryfile='', transform=None, shuffle=False):
        lib = torch.load(libraryfile, map_location='cpu')
        self.t_data = data
        wsidirs = []
        for i,name in enumerate(lib['slides']):
            wsiname = os.path.basename(name)
            wsiname = wsiname.split('.')[0]
            wsidir = os.path.join(args.patch_dir, wsiname)
            wsidirs.append(wsidir)
        #Flatten grid
        grid = []
        for i,g in enumerate(lib['grid']):
            grid.extend(g)
        self.wsidirs = wsidirs
        self.grid = grid
        self.transform = transform
        self.shuffle = shuffle

    def shuffletraindata(self):
        if self.shuffle:
            self.t_data = random.sample(self.t_data, len(self.t_data))

    def __getitem__(self,index):
        slideIDX, coord, target = self.t_data[index]
        wsidir = self.wsidirs[slideIDX]
        img_path = os.path.join(wsidir, f"{coord[0]}_{coord[1]}.jpg")
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target
    
    def __len__(self):
        return len(self.t_data)

class ResNetEncoder(nn.Module):
    def __init__(self):
        super(ResNetEncoder, self).__init__()
        temp = resnet34(pretrained=True)
        temp.fc = nn.Linear(temp.fc.in_features, 2)
        self.feature_extractor = nn.Sequential(*list(temp.children())[:-1])
        self.fc = temp.fc

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return self.fc(x), x

