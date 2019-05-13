import os
import argparse
import datetime
import time

# must be set before torch import
from model import Darknet
from test import evaluate

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import torch
from torch.autograd import Variable

from terminaltables import AsciiTable

from utils.logger import Logger
from yolo_loader import get_data_loader
from utils.utils import weights_init_normal

# from terminaltables import AsciiTable


parser = argparse.ArgumentParser(description='Process images for new dataset')
parser.add_argument('-c', '--config', dest='config_path', required=True)
parser.add_argument('-gpu', type=int, default=[1], nargs='+', help='used gpu')
args = parser.parse_args()


logger = Logger("logs")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Darknet("/home/yuval/Documents/XNOR/sealnet/models/yolov3/yolov3.cfg", img_size=640).to(device)
model.apply(weights_init_normal)
model.load_darknet_weights("/data/pretrained_weights/darknet53.conv.74")

print("GPUS" + os.environ["CUDA_VISIBLE_DEVICES"])
train_data = get_data_loader(args.config_path, "train", 4, 8)
test_data = get_data_loader(args.config_path, "test", 1, 8)

optimizer = torch.optim.Adam(model.parameters())

metrics = [
    "grid_size",
    "loss",
    "x",
    "y",
    "w",
    "h",
    "conf",
    "cls",
    "cls_acc",
    "recall50",
    "recall75",
    "precision",
    "conf_obj",
    "conf_noobj",
]

EPOCHS = 100
IM_SIZE = 640
EVAL_INTERVAL = 1
CHKPT_INTERVAL = 10
class_names = ["Ringed", "Bearded", "UNK"]
saved = False
for epoch in range(EPOCHS):
    saved = False
    model.train()
    start_time = time.time()
    print(epoch)
    for batch_i, (_, imgs, targets) in enumerate(train_data):
        print(batch_i)

        batches_done = len(train_data) * epoch + batch_i
        imgs = Variable(imgs.to(device))
        targets = Variable(targets.to(device), requires_grad=False)

        loss, outputs = model(imgs, targets)
        if batches_done % 2:
            # Accumulates gradient before each step
            optimizer.step()
            optimizer.zero_grad()

        log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, EPOCHS, batch_i, len(train_data))

        metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

        # Log metrics at each YOLO layer
        for i, metric in enumerate(metrics):
            formats = {m: "%.6f" for m in metrics}
            formats["grid_size"] = "%2d"
            formats["cls_acc"] = "%.2f%%"
            row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
            metric_table += [[metric, *row_metrics]]

            # Tensorboard logging
            tensorboard_log = []
            for j, yolo in enumerate(model.yolo_layers):
                for name, metric in yolo.metrics.items():
                    if name != "grid_size":
                        tensorboard_log += [(f"{name}_{j+1}", metric)]
            tensorboard_log += [("loss", loss.item())]
            logger.list_of_scalars_summary(tensorboard_log, batches_done)

        log_str += AsciiTable(metric_table).table
        log_str += f"\nTotal loss {loss.item()}"

        # Determine approximate time left for epoch
        epoch_batches_left = len(train_data) - (batch_i + 1)
        time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
        log_str += f"\n---- ETA {time_left}"

        print(log_str)

        model.seen += imgs.size(0)

        if epoch % EVAL_INTERVAL == 0 and False:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=test_data,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=IM_SIZE,
                batch_size=8,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")

        if not saved and epoch % CHKPT_INTERVAL == 0:
            torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)
            saved = True