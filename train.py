import argparse
import logging
import json
import sys
import os
from collections import defaultdict


import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch import topk
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss
from albumentations import Compose, RandomBrightnessContrast, \
    HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur


from training import losses
from training.datasets.classifier_dataset import DeepFakeClassifierDataset
from training.losses import WeightedLosses
from training.tools.config import load_config
from training.tools.utils import create_optimizer, AverageMeter
from training.transforms.albu import IsotropicResize
from training.zoo import classifiers


stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
handlers = [stdout_handler]
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s; %(asctime)s; %(module)s:%(funcName)s:%(lineno)d; %(message)s",
    handlers=handlers)

logger = logging.getLogger(__name__)


def create_train_transforms(size=300):
    return Compose([
        ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
        GaussNoise(p=0.1),
        GaussianBlur(blur_limit=3, p=0.05),
        HorizontalFlip(),
        OneOf([
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA,
                            interpolation_up=cv2.INTER_CUBIC),
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA,
                            interpolation_up=cv2.INTER_LINEAR),
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR,
                            interpolation_up=cv2.INTER_LINEAR),
        ], p=1),
        PadIfNeeded(min_height=size, min_width=size,
                    border_mode=cv2.BORDER_CONSTANT),
        OneOf([RandomBrightnessContrast(), FancyPCA(),
              HueSaturationValue()], p=0.7),
        ToGray(p=0.2),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2,
                         rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
    ]
    )


def create_val_transforms(size=300):
    return Compose([
        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA,
                        interpolation_up=cv2.INTER_CUBIC),
        PadIfNeeded(min_height=size, min_width=size,
                    border_mode=cv2.BORDER_CONSTANT),
    ])


def val_epoch(data_val, model, current_epoch):
    print("Test phase")
    model = model.eval()

    bce, probs, targets = validate(model, data_loader=data_val)
    print("Epoch: {} bce: {}".format(
        current_epoch, bce))
    return bce


def validate(net, data_loader, prefix=""):
    probs = defaultdict(list)
    targets = defaultdict(list)

    with torch.no_grad():
        for sample in tqdm(data_loader):
            imgs = sample["image"].cuda()
            img_names = sample["img_name"]
            labels = sample["labels"].cuda().float()
            out = net(imgs)
            labels = labels.cpu().numpy()
            preds = torch.sigmoid(out).cpu().numpy()
            for i in range(out.shape[0]):
                video, img_id = img_names[i].split("/")
                probs[video].append(preds[i].tolist())
                targets[video].append(labels[i].tolist())
    data_x = []
    data_y = []
    for vid, score in probs.items():
        score = np.array(score)
        lbl = targets[vid]

        score = np.mean(score)
        lbl = np.mean(lbl)
        data_x.append(score)
        data_y.append(lbl)
    y = np.array(data_y)
    x = np.array(data_x)
    fake_idx = y > 0.1
    real_idx = y < 0.1
    fake_loss = log_loss(y[fake_idx], x[fake_idx], labels=[0, 1])
    real_loss = log_loss(y[real_idx], x[real_idx], labels=[0, 1])
    print("{}fake_loss".format(prefix), fake_loss)
    print("{}real_loss".format(prefix), real_loss)

    return (fake_loss + real_loss) / 2, probs, targets


def train_epoch(current_epoch, loss_functions, model, optimizer, scheduler, train_data_loader, conf):
    losses = AverageMeter()
    fake_losses = AverageMeter()
    real_losses = AverageMeter()
    max_iters = conf["batches_per_epoch"]
    print("training epoch: {}".format(current_epoch))

    pbar = tqdm(enumerate(train_data_loader), total=max_iters,
                desc="Epoch {}".format(current_epoch), ncols=0)
    if conf["optimizer"]["schedule"]["mode"] == "epoch":
        scheduler.step(current_epoch)
    for i, sample in pbar:
        imgs = sample["image"]
        labels = sample["labels"]

        if torch.cuda.is_available():
            imgs = imgs.cuda()
            labels = labels.cuda().float()
        else:
            labels = labels.float()

        out_labels = model(imgs)

        fake_loss = 0
        real_loss = 0
        fake_idx = labels > 0.5
        real_idx = labels <= 0.5

        if torch.sum(fake_idx * 1) > 0:
            fake_loss = loss_functions["classifier_loss"](
                out_labels[fake_idx], labels[fake_idx])
        if torch.sum(real_idx * 1) > 0:
            real_loss = loss_functions["classifier_loss"](
                out_labels[real_idx], labels[real_idx])

        loss = (fake_loss + real_loss) / 2
        losses.update(loss.item(), imgs.size(0))
        fake_losses.update(
            0 if fake_loss == 0 else fake_loss.item(), imgs.size(0))
        real_losses.update(
            0 if real_loss == 0 else real_loss.item(), imgs.size(0))

        optimizer.zero_grad()
        pbar.set_postfix({"lr": float(scheduler.get_lr()[-1]), "epoch": current_epoch, "loss": losses.avg,
                          "fake_loss": fake_losses.avg, "real_loss": real_losses.avg})

        loss.requires_grad = True
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        if conf["optimizer"]["schedule"]["mode"] in ("step", "poly"):
            scheduler.step(i + current_epoch * max_iters)
        if i == max_iters - 1:
            break
    pbar.close()


def cli():
    parser = argparse.ArgumentParser("PyTorch Xview Pipeline")
    arg = parser.add_argument
    arg('--config', metavar='CONFIG_FILE', help='path to configuration file')
    arg('--workers', type=int, default=0, help='number of cpu threads to use')
    arg('--output-dir', type=str, default='weights/')
    arg('--resume', type=str, default='')
    arg('--fold', type=int, default=0)
    arg('--prefix', type=str, default='classifier_')
    arg('--data-dir', type=str, default="/mnt/sota/datasets/deepfake")
    arg('--folds-csv', type=str, default='folds.csv')
    arg('--crops-dir', type=str, default='crops')
    arg('--label-smoothing', type=float, default=0.01)
    arg('--from-zero', action='store_true', default=False)
    arg("--seed", default=777, type=int)
    arg("--padding-part", default=3, type=int)
    arg("--opt-level", default='O1', type=str)
    arg("--test_every", type=int, default=1)
    arg("--no-oversample", action="store_true")
    arg("--no-hardcore", action="store_true")
    arg("--only-changed-frames", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = cli()
    os.makedirs(args.output_dir, exist_ok=True)
    conf = load_config(args.config)
    model = classifiers.__dict__[conf['network']](encoder=conf['encoder'])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    reduction = "mean"

    loss = losses.__dict__[conf["loss"]](
        reduction=reduction).to(device)

    loss_functions = {"classifier_loss": loss}
    optimizer, scheduler = create_optimizer(conf['optimizer'], model)
    bce_best = 100
    start_epoch = 0
    batch_size = conf['optimizer']['batch_size']

    data_train = DeepFakeClassifierDataset(mode="train",
                                           oversample_real=not args.no_oversample,
                                           fold=args.fold,
                                           padding_part=args.padding_part,
                                           hardcore=not args.no_hardcore,
                                           crops_dir=args.crops_dir,
                                           data_path=args.data_dir,
                                           label_smoothing=args.label_smoothing,
                                           folds_csv=args.folds_csv,
                                           transforms=create_train_transforms(
                                               conf["size"]),
                                           normalize=conf.get("normalize", None))
    data_val = DeepFakeClassifierDataset(mode="val",
                                         fold=args.fold,
                                         padding_part=args.padding_part,
                                         crops_dir=args.crops_dir,
                                         data_path=args.data_dir,
                                         folds_csv=args.folds_csv,
                                         transforms=create_val_transforms(
                                             conf["size"]),
                                         normalize=conf.get("normalize", None))

    val_data_loader = DataLoader(data_val, batch_size=batch_size * 2, num_workers=args.workers, shuffle=False,
                                 pin_memory=False)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            state_dict = checkpoint['state_dict']
            state_dict = {k[7:]: w for k, w in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            if not args.from_zero:
                start_epoch = checkpoint['epoch']
            print("=> loaded checkpoint '{}' (epoch {}, bce_best {})"
                  .format(args.resume, checkpoint['epoch'], checkpoint['bce_best']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    snapshot_name = "{}{}_{}_{}".format(
        conf.get("prefix", args.prefix), conf['network'], conf['encoder'], args.fold)

    data_val.reset(1, args.seed)
    max_epochs = conf['optimizer']['schedule']['epochs']
    current_epoch = start_epoch

    for epoch in range(start_epoch, max_epochs):
        data_train.reset(epoch, args.seed)
        train_sampler = None

        model.train()
        for p in model.parameters():
            p.requires_grad = True

        train_data_loader = DataLoader(data_train, batch_size=batch_size, num_workers=args.workers,
                                       shuffle=train_sampler is None, sampler=train_sampler, pin_memory=False,
                                       drop_last=True)

        train_epoch(current_epoch, loss_functions, model,
                    optimizer, scheduler, train_data_loader, conf)
        model = model.eval()

        if (epoch + 1) % args.test_every == 0:
            bce = val_epoch(val_data_loader, model,
                            current_epoch=current_epoch)
            torch.save({
                'epoch': current_epoch + 1,
                'state_dict': model.state_dict(),
                'bce': bce,
            }, args.output_dir + snapshot_name + "_{}".format(current_epoch))

        current_epoch += 1
