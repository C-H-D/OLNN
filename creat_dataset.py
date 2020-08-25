import numpy as np
import glob
import os
from random import shuffle


label_dir = 'D:/Ear/dataset/detect/bbox/'
data_dir = "D:/Ear/dataset/allset/"
save_dir = 'D:/Ear/dataset/detect/'
imgs = glob.glob(data_dir+"*_clean.npy")

pos = []
neg = []

for img in imgs:
    name = img.split("\\")[-1].split("_")[0]
    label = np.load(label_dir+name+"_label.npy")
    if 2 in label[:, 5]:
        neg.append(img)
    else:
        pos.append(img)

shuffle(neg)
shuffle(pos)
neg_train = neg[:int(len(neg)*0.8)]
pos_train = pos[:int(len(pos)*0.8)]
neg_test = neg[int(len(neg)*0.8):]
pos_test = pos[int(len(pos)*0.8):]
count = 0
for img_name in neg_train:
    count += 1
    print(count/1665)
    name = img_name.split("\\")[-1].split("_")[0]
    label = np.load(label_dir + name + "_label.npy")
    img = np.load(img_name)
    for l in label:
        np.save(save_dir+"neg/"+name+'_'+str(int(l[0]))+".npy", img[int(l[0])])
    os.remove(img_name)
for img_name in pos_train:
    count += 1
    print(count / 1665)
    name = img_name.split("\\")[-1].split("_")[0]
    label = np.load(label_dir + name + "_label.npy")
    img = np.load(img_name)
    for l in label:
        np.save(save_dir+"pos/"+name+'_'+str(int(l[0]))+".npy", img[int(l[0])])
    os.remove(img_name)
for img_name in neg_test:
    count += 1
    print(count / 1665)
    name = img_name.split("\\")[-1].split("_")[0]
    label = np.load(label_dir + name + "_label.npy")
    img = np.load(img_name)
    for l in label:
        np.save(save_dir+"neg_test/"+name+'_'+str(int(l[0]))+".npy", img[int(l[0])])
    os.remove(img_name)
for img_name in pos_test:
    count += 1
    print(count / 1665)
    name = img_name.split("\\")[-1].split("_")[0]
    label = np.load(label_dir + name + "_label.npy")
    img = np.load(img_name)
    for l in label:
        np.save(save_dir+"pos_test/"+name+'_'+str(int(l[0]))+".npy", img[int(l[0])])
    os.remove(img_name)


