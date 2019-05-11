import argparse
import os
import time

import torch
import torchvision.models as models
from torch import nn
from torch.autograd import Variable

from models.yolov3.models import Darknet
from models.yolov3.utils.utils import weights_init_normal
from train_loader import get_train_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = Darknet("yolov3.cfg", img_size=640).to(device)
# model.apply(weights_init_normal)
# model.load_darknet_weights("/data/pretrained_weights/darknet53.conv.74")

model = models.alexnet(pretrained=True).to(device)
model.classifier[6] = nn.Linear(4096, 4)
loss = models.AlexNet(num_classes=3).to(device)
parser = argparse.ArgumentParser(description='Process images for new dataset')
parser.add_argument('-c', '--config', dest='config_path', required=True)
parser.add_argument('-gpu', type=int, default=[1], nargs='*', help='used gpu')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = ''.join(str(x) for x in args.gpu)

dataloader = get_train_loader(args.config_path, 16, 1)
for epoch in range(100):
    model.train()
    loss.train()

    start_time = time.time()
    print(epoch)
    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        print(batch_i)

        batches_done = len(dataloader) * epoch + batch_i
        imgs = Variable(imgs.to(device))
        targets = Variable(targets.to(device), requires_grad=False)

        outputs = model(imgs)
        loss_output = loss(outputs, targets)
        x=1