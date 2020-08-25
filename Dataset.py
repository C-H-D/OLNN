import torch
import torchvision.transforms as transforms
import glob
import numpy as np
import os
from PIL import Image


class Dataset(torch.utils.data.Dataset):
    def __init__(self, training=True):
        super(Dataset, self).__init__()
        self.training = training
        self.label_dir = '../dataset/detect/bbox/'
        if self.training:
            self.pos_dir = "../dataset/detect/pos/"
            self.neg_dir = "../dataset/detect/neg/"
        else:
            self.pos_dir = "../dataset/detect/postest/"
            self.neg_dir = "../dataset/detect/negtest/"
        self.transform_img = transforms.Compose([
            transforms.Resize((888, 1496)),
            transforms.ToTensor()
        ])

        index = glob.glob(self.pos_dir + '*.npy') + glob.glob(self.neg_dir + '*.npy')
        index = [int(i.split('_')[0].split('\\')[-1]) for i in index]
        index = list(set(index))
        index = sorted(index)

        self.imgs = []
        self.bboxes = []
        self.labels = []
        self.ids = []

        for i in index:
            label = np.load(self.label_dir + str(i) + '_label.npy')
            for l in label:
                if os.path.exists(self.pos_dir + str(i) + '_' + str(int(l[0])) + '.npy'):
                    self.imgs.append(self.pos_dir + str(i) + '_' + str(int(l[0])) + '.npy')
                    self.bboxes.append(l[1:5])
                    self.labels.append(l[5])
                    self.ids.append(str(i)+'_'+str(int(l[0])))
                elif os.path.exists(self.neg_dir + str(i) + '_' + str(int(l[0])) + '.npy'):
                    self.imgs.append(self.neg_dir + str(i) + '_' + str(int(l[0])) + '.npy')
                    self.bboxes.append(l[1:5])
                    self.labels.append(l[5])
                    self.ids.append(str(i) + '_' + str(int(l[0])))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = np.load(self.imgs[index])
        img = (img.astype(np.float32)) / img.max()
        img = Image.fromarray(img)
        img = self.transform_img(img)
        bbox = self.bboxes[index]
        d = {}
        d['boxes'] = torch.FloatTensor(bbox)
        d['labels'] = torch.LongTensor([self.labels[index]])
        return img, d
