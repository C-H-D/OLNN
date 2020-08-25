import torchvision
import torch
import torch.optim as optim
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from Dataset import Dataset
from utils import *


BATCH_SIZE = 4
EPOCH = 300
THRESH = 0.9
IOU = 0.3
CLASSES = 3
DEVICE = torch.device("cuda")

anchor_generator = AnchorGenerator(sizes=((32, 64),),
                                   aspect_ratios=((0.6, 1.0, 1.6),))
backbone = torchvision.models.vgg19(pretrained=False).features
backbone.out_channels = 512
model = FasterRCNN(
    backbone,
    num_classes=CLASSES,
    rpn_anchor_generator=anchor_generator
)
model.to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-5)
train_dataset = Dataset()
test_dataset = Dataset(training=False)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    drop_last=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False
)
for epoch in range(EPOCH):
    for index, (data, target) in enumerate(train_loader):
        model.train()
        data = data.to(DEVICE)
        target['boxes'] = target['boxes'].to(DEVICE)
        target['labels'] = target['labels'].squeeze(0).to(DEVICE)
        optimizer.zero_grad()
        dic = []
        boxes = target['boxes']
        labels = target['labels']
        for i in range(len(labels)):
            dic.append({'boxes': boxes[i].unsqueeze(0), 'labels': labels[i]})
        output = model(data, dic)
        loss = sum(l for l in output.values())
        if index % 5 == 0:
            print("Epoch {}[{:.1f}%]\tLoss: {:.6f}".format(
                epoch + 1, 100 * index / len(train_loader), loss.item()
            ))
        loss.backward()
        optimizer.step()
    if epoch % 1 == 0:
        model.eval()
        tpp, fpp, fnp, tpn, fpn, fnn = [0, 0, 0, 0, 0, 0]
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(DEVICE)
                target['boxes'] = target['boxes'].to(DEVICE)
                target['labels'] = target['labels'].squeeze(0).to(DEVICE)
                dic = []
                boxes = target['boxes']
                labels = target['labels']
                for i in range(len(labels)):
                    dic.append({'boxes': boxes[i].unsqueeze(0), 'labels': labels[i]})
                output = model(data)
                for i in range(len(output)):
                    o = output[i]
                    t = dic[i]
                    t_box = t['boxes']
                    t_label = t['labels']
                    if len(o['labels']) == 0:
                        if t_label == 1:
                            fnp += 1
                        elif t_label == 2:
                            fnn += 1
                        continue
                    o_box = o['boxes'][0]
                    o_label = o['labels'][0]
                    o_score = o['scores'][0]
                    if o_score > THRESH:
                        if iou(t_box[0], o_box) > IOU:
                            if t_label == o_label:
                                if t_label == 1:
                                    tpp += 1
                                elif t_label == 2:
                                    tpn += 1
                            else:
                                if t_label == 1:
                                    fpn += 1
                                    fnp += 1
                                elif t_label == 2:
                                    fpp += 1
                                    fnn += 1
                        else:
                            if t_label == 1:
                                fnp += 1
                            elif t_label == 2:
                                fnn += 1
                    else:
                        if t_label == 1:
                            fnp += 1
                        elif t_label == 2:
                            fnn += 1
        print("Epoch {}\t Pos Precisison:{:.3f}\t Pos Recall:{:.3f}\t Neg Precisison:{:.3f}\t Neg Recall:{:.3f}".format(
            epoch+1, tpp/(tpp+fpp+1e-5), tpp/(tpp+fnp+1e-5), tpn/(tpn+fpn+1e-5), tpn/(tpn+fnn+1e-5)
        ))
        torch.save(model.state_dict(), './models/'+'model_'+str(epoch+1)+'.pth')
