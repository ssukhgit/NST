from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.utils.data as data
from os import listdir
from os.path import join, splitext
import numpy as np
import torch
import os
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

def is_image_file(filename):
    return any(filename.lower().endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def default_loader(path):
    return Image.open(path).convert('RGB')

class Dataset(data.Dataset):
    def __init__(self,contentPath,stylePath,fineSize):
        super(Dataset,self).__init__()
        self.contentPath = contentPath
        self.image_list_content = [x for x in listdir(contentPath) if is_image_file(x)]
        self.image_list_style = [x for x in listdir(stylePath) if is_image_file(x)]
        self.stylePath = stylePath
        self.fineSize = fineSize

        self.prep = transforms.Compose([
                    transforms.Resize(fineSize),
                    transforms.ToTensor(),
                    ])

    def __getitem__(self,index):
        contentImgPath = os.path.join(self.contentPath,self.image_list_content[index])
        styleImgPath = os.path.join(self.stylePath,self.image_list_style[index])
        contentImg = default_loader(contentImgPath)
        styleImg = default_loader(styleImgPath)

        # Preprocess Images
        contentImg = self.prep(contentImg)
        styleImg = self.prep(styleImg)

        return contentImg.squeeze(0),styleImg.squeeze(0),self.image_list_content[index]

    def __len__(self):
        return len(self.image_list_content)
