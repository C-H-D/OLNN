import torchvision
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import glob
import os


data_path = "D:/Ear/dataset/"
dataset_name = 'Test set of our otosclerosis-LNN model/'
result_name = 'eval_results/'
types = ["mixed(only left ear is normal)/", "otosclerosis/", "normal/part1/", "normal/part2/"]
ears = ["right ear/", "left ear/"]

transform_img = transforms.Compose([
    transforms.ToTensor()
])
EPOCH = 250
CLASSES = 3
DEVICE = torch.device("cuda")
BATCH_SIZE = 10

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
start_time = time.time()
ear_count = 0
for T in types:
    for E in ears:
        CTs = os.listdir(data_path+dataset_name+T+E)
        for CT in CTs:
            print('current path:{}'.format(data_path+dataset_name+T+E+CT))
            ear_count += 1
            img_names = glob.glob(data_path+dataset_name+T+E+CT+'/*.jpg')
            sorted(img_names, key=lambda x: x.split('\\')[-1])
            with torch.no_grad():
                start, end = 0, BATCH_SIZE
                path = data_path+result_name+T+E+CT
                if not os.path.exists(path):
                    os.makedirs(path)
                while True:
                    layers = []
                    images = []
                    for i in range(start, end):
                        layers.append(img_names[i].split("\\")[-1])
                        image = transform_img(Image.open(img_names[i]))
                        image = image / image.max()
                        image = image.to(DEVICE).unsqueeze(0)
                        images.append(image)
                    model_input = torch.cat(images, dim=0)
                    output = model(model_input)
                    for i in range(len(output)):
                        if len(output[i]['labels']) > 0:
                            log = open(path+'/'+'result.txt', 'a')
                            o_box = output[i]['boxes'][0]
                            o_label = output[i]['labels'][0]
                            o_score = output[i]['scores'][0]
                            log.write("layer:{}, output:{}, confidence:{:.3f}\n".format(layers[i], o_label, o_score))
                            x1, y1, x2, y2 = o_box
                            plt.figure()
                            rec_nd = [x1, y1]
                            d_nd = [x2 - x1, y2 - y1]
                            if o_label == 1:
                                plt.gca().add_patch(plt.Rectangle(
                                    rec_nd, d_nd[0], d_nd[1], fill=False, edgecolor='r', linewidth=1))
                            elif o_label == 2:
                                plt.gca().add_patch(plt.Rectangle(
                                    rec_nd, d_nd[0], d_nd[1], fill=False, edgecolor='g', linewidth=1))
                            plt.imshow(images[i].cpu().squeeze(0).squeeze(0), cmap='gray')
                            plt.savefig(path+'/'+'result_'+layers[i])
                            plt.close()
                    start += BATCH_SIZE
                    end += BATCH_SIZE
                    end = min(end, len(img_names))
                    if start >= len(img_names):
                        break
end_time = time.time()
print("Average inference time per ear:{} seconds".format((end_time - start_time)/ear_count))

