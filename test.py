import torchvision
import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from Dataset import Dataset
from utils import *


BATCH_SIZE = 10
EPOCH = 250
THRESH = 0.99
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
model.load_state_dict(torch.load('models_new/'+'model_'+str(EPOCH)+'.pth'))
model.to(DEVICE)
model.eval()
test_dataset = Dataset(training=False)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False
)
total, tpp, fpp, fnp, tpn, fpn, fnn = [0, 0, 0, 0, 0, 0, 0]
IOUs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
count = 0
with torch.no_grad():
    for data, target in test_loader:
        count += 1
        print("{:.2f}".format(count/len(test_loader)*100))
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
            total += 1
            o = output[i]
            t = dic[i]
            t_box = t['boxes']
            t_label = t['labels']
            if len(o['labels']) == 0:
                print('no output')
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
                    IOUs[int(iou(t_box[0], o_box) * 10)] += 1
                    if t_label == o_label:
                        if t_label == 1:
                            tpp += 1
                        elif t_label == 2:
                if t_label == 1:
                    fnp += 1
                elif t_label == 2:
                    fnn += 1
print("Epoch {}\t Total:{}, TPP:{}, FPP:{}, FNP:{}, TPN:{}, FPN:{}, FNN:{}".format(
    EPOCH, total, tpp, fpp, fnp, tpn, fpn, fnn
))
print("Epoch {}\t Pos Precisison:{:.3f}\t Pos Recall:{:.3f}\t Neg Precisison:{:.3f}\t Neg Recall:{:.3f}".format(
    EPOCH, tpp/(tpp+fpp+1e-5), tpp/(tpp+fnp+1e-5), tpn/(tpn+fpn+1e-5), tpn/(tpn+fnn+1e-5)
))
print(IOUs)
