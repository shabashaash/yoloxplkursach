import time
import torch
import torch.nn as nn
import pytorch_lightning as pl

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import LightningModule

from models.OneStage import OneStageD
from models.darknet_csp import CSPDarkNet
from models.pafpn import PAFPN
from models.decoupled_head import DecoupledHead
from models.yolox_loss import YOLOXLoss
from models.yolox_decoder import YOLOXDecoder
from models.evaluators.coco import COCOEvaluator, convert_to_coco_format
from models.utils.ema import ModelEMA
from torch.optim import SGD, AdamW, Adam
from models.lr_scheduler import CosineWarmupScheduler

from torch.utils.data import DataLoader
from models.data.datasets.cocoDataset import COCODataset
from models.data.mosaic_detection import MosaicDetection
from models.data.augmentation.data_augments import TrainTransform, ValTransform
from torch.utils.data.sampler import BatchSampler, RandomSampler

import os
import yaml

class PRWDataModule(pl.LightningDataModule):
    def __init__(self, cfgs):
        super().__init__()
        self.dataset_train = None
        self.dataset_val = None
        self.cd = cfgs['dataset']
        self.ct = cfgs['transform']
        # dataloader parameters
        self.data_dir = self.cd['dir']
        self.train_dir = self.cd['train']
        self.val_dir = self.cd['val']
        self.img_size_train = tuple(self.cd['train_size'])
        self.img_size_val = tuple(self.cd['val_size'])
        self.train_batch_size = self.cd['train_batch_size']
        self.val_batch_size = self.cd['val_batch_size']
        # transform parameters
        self.hsv_prob = self.ct['hsv_prob']
        self.flip_prob = self.ct['flip_prob']
        # mosaic
        self.mosaic_prob = self.ct['mosaic_prob']
        self.mosaic_scale = self.ct['mosaic_scale']
        self.degrees = self.ct['degrees']
        self.translate = self.ct['translate']
        self.shear = self.ct['shear']
        self.perspective = self.ct['perspective']
        # copypaste
        self.copypaste_prob = self.ct['copypaste_prob']
        self.copypaste_scale = self.ct['copypaste_scale']
        # cutpaste
        self.cutpaste_prob = self.ct['cutpaste_prob']

    def train_dataloader(self):
        self.dataset_train = COCODataset(
            self.data_dir,
            name=self.train_dir,
            img_size=self.img_size_train,
            preprocess=TrainTransform(max_labels=50, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob),
            cache=False
        )
        self.dataset_train = MosaicDetection(
            self.dataset_train,
            mosaic_prob=self.mosaic_prob,
            mosaic_scale=self.mosaic_scale,
            img_size=self.img_size_train,
            preprocess=TrainTransform(
                max_labels=100,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,),
            degrees=self.degrees,
            translate=self.translate,
            shear=self.shear,
            perspective=self.perspective,
            copypaste_prob=self.copypaste_prob,
            copypaste_scale=self.copypaste_scale,
            cutpaste_prob=self.cutpaste_prob,
        )
        train_loader = DataLoader(self.dataset_train, batch_sampler=BatchSampler(RandomSampler(self.dataset_train), batch_size=self.train_batch_size, drop_last=False),
                                  num_workers=12, pin_memory=True)
        return train_loader

    def val_dataloader(self):
        self.dataset_val = COCODataset(
            self.data_dir,
            name=self.val_dir,
            img_size=self.img_size_val,
            preprocess=ValTransform(legacy=False),
            cache=False,
        )
        val_loader = DataLoader(self.dataset_val, batch_size=self.val_batch_size, sampler=torch.utils.data.SequentialSampler(self.dataset_val),
                                num_workers=6, pin_memory=True, shuffle=False)
        return val_loader


class YOLOXLit(LightningModule):
    def __init__(self, cfgs):
        super().__init__()
        self.cb = cfgs['backbone']
        self.cn = cfgs['neck']
        self.ch = cfgs['head']
        self.cd = cfgs['dataset']
        self.co = cfgs['optimizer']
        b_depth = self.cb['depth']
        b_norm = self.cb['normalization']
        b_act = self.cb['activation']
        b_channels = self.cb['input_channels']
        out_features = self.cb['output_features']
        n_depth = self.cn['depth']
        n_channels = self.cn['input_channels']
        n_norm = self.cn['normalization']
        n_act = self.cn['activation']
        n_anchors = 1
        strides = [8, 16, 32]
        self.use_l1 = False
        self.nms_threshold = 0.65
        self.confidence_threshold = 0.01
        self.num_classes = self.cd['num_classes']
        self.train_batch_size = self.cd['train_batch_size']
        self.val_batch_size = self.cd['val_batch_size']
        self.img_size_train = tuple(self.cd['train_size'])
        self.img_size_val = tuple(self.cd['val_size'])
        self.ema = self.co['ema']
        self.warmup = self.co['warmup']
        self.iter_times = []
        self.backbone = CSPDarkNet(b_depth, b_channels, out_features, b_norm, b_act)
        self.neck = PAFPN(n_depth, n_channels, n_norm, n_act)
        self.head = DecoupledHead(self.num_classes, n_anchors, n_channels, n_norm, n_act)
        self.loss = YOLOXLoss(self.num_classes, strides)
        self.decoder = YOLOXDecoder(self.num_classes, strides)
        self.model = OneStageD(self.backbone, self.neck, self.head)
        self.ema_model = None
        self.head.initialize_biases(1e-2)
        self.model.apply(initializer)
        self.automatic_optimization = False
        self.ap50_95 = 0
        self.ap50 = 0

    def on_train_start(self) -> None:
        if self.ema is True:
            self.ema_model = ModelEMA(self.model, 0.9998)

    def training_step(self, batch, batch_idx):
        imgs, labels, _, _, _ = batch
        output = self.model(imgs)
        loss, loss_iou, loss_obj, loss_cls, loss_l1, proportion = self.loss(output, labels)
        self.log("loss/loss", loss, prog_bar=True)
        self.log("loss/iou", loss_iou, prog_bar=False)
        self.log("loss/obj", loss_obj, prog_bar=False)
        self.log("loss/cls", loss_cls, prog_bar=False)
        self.log("loss/l1", loss_l1, prog_bar=False)
        self.log("loss/proportion", proportion, prog_bar=False)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        if self.ema is True:
            self.ema_model.update(self.model)
        self.lr_schedulers().step()

    def validation_step(self, batch, batch_idx):
        imgs, labels, img_hw, image_id, img_name = batch
        if self.ema_model is not None:
            model = self.ema_model.ema
        else:
            model = self.model
        start_time = time.time()
        output = model(imgs)
        detections = self.decoder(output, self.confidence_threshold, self.nms_threshold)
        self.iter_times.append(time.time() - start_time)
        detections = convert_to_coco_format(detections, image_id, img_hw, self.img_size_val,
                                            self.trainer.datamodule.dataset_val.class_ids)
        return detections

    def validation_epoch_end(self, val_step_outputs):
        detect_list = []
        for i in range(len(val_step_outputs)):
            detect_list += val_step_outputs[i]
        ap50_95, ap50, summary = COCOEvaluator(
            detect_list, self.trainer.datamodule.dataset_val)
        print("Batch {:d}, mAP = {:.3f}, mAP50 = {:.3f}".format(self.current_epoch, ap50_95, ap50))
        print(summary)
        self.log("val/mAP", ap50_95, prog_bar=False)
        self.log("val/mAP50", ap50, prog_bar=False)
        if ap50_95 > self.ap50_95:
            self.ap50_95 = ap50_95
        if ap50 > self.ap50:
            self.ap50 = ap50

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=self.co["learning_rate"], momentum=self.co["momentum"])
        total_steps = self.trainer.estimated_stepping_batches
        lr_scheduler = CosineWarmupScheduler(
            optimizer, warmup=self.warmup * total_steps, max_iters=total_steps
        )
        return [optimizer], [lr_scheduler]

    def on_train_end(self) -> None:
        average_ifer_time = torch.tensor(self.iter_times, dtype=torch.float32).mean()
        print("The average iference time is ", average_ifer_time, " ms")
        print("Best mAP = {:.3f}, best mAP50 = {:.3f}".format(self.ap50_95, self.ap50))


def initializer(M):
    for m in M.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03






CFG_FN = 'configs/yolox_m.yaml' 


def main():
    assert os.path.isfile(CFG_FN), f"Config file '{CFG_FN}' does not exist!"
    with open(CFG_FN, encoding='ascii', errors='ignore') as f:
        config = yaml.safe_load(f)
    config['dataset']['dir'] = '/kaggle/input/prwv16/coco'
    config['dataset']['train'] = 'minitrain'
    config['dataset']['val'] = 'minival'
    config['dataset']['num_classes'] = 1
    print(config)
    model = YOLOXLit(config)
    data = PRWDataModule(config)
    #print(data.train_dataloader())
    #print(data.val_dataloader())
    logger = CSVLogger("logs", name="csvlogger")
    
    seed_everything(42, workers=True)
    
    trainer = Trainer(
        gpus=1,
        max_epochs=300,
        check_val_every_n_epoch=5,
        log_every_n_steps=10,
        enable_progress_bar=True,
        logger=logger,
        # precision=16,
        # amp_backend="apex",
        # amp_level=01,
        # auto_lr_find=True,
        # benchmark=False,
        # callbacks=[device_stats],
        # default_root_dir="lightning_logs",
        # detect_anomaly=True,
        # limit_train_batches=3,
        # limit_val_batches=2,
        # reload_dataloaders_every_n_epochs=10,
    )

    trainer.fit(model, datamodule=data)
    # trainer.tune(model, datamodule=data)
    # trainer.validate(model, datamodule=data)
    # trainer.test(model, datamodule=data)
    