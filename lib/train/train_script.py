import os
import copy
# loss function related
from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss
# train pipeline related
from lib.train.trainers import LTRTrainer
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from .base_functions import *
# network related
from lib.models.ortrack import build_ortrack
# forward propagation related
from lib.train.actors import ORTrackActor
# for import modules
import importlib

from ..utils.focal_loss import FocalLoss


def run(settings):
    settings.description = 'Training script for STARK-S, STARK-ST stage1, and STARK-ST stage2'

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

    # update settings based on cfg
    update_settings(settings, cfg)

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    # Build dataloaders
    loader_train, loader_val = build_dataloaders(cfg, settings)

    if "RepVGG" in cfg.MODEL.BACKBONE.TYPE or "swin" in cfg.MODEL.BACKBONE.TYPE or "LightTrack" in cfg.MODEL.BACKBONE.TYPE:
        cfg.ckpt_dir = settings.save_dir

    # Create network
    if settings.script_name == "ortrack":
        # =====distillation====
        is_distill_training = cfg.MODEL['IS_DISTILL']
        net_teacher = None
        if is_distill_training:
            from lib.train.utils.teacher_loader import (
                resolve_teacher_path,
                read_checkpoint,
                load_teacher_weights,
                freeze_teacher,
                print_teacher_report,
            )

            # Dung deepcopy de tach biet hoan toan config cua teacher, tranh lam thay doi config cua student
            cfg_teacher = copy.deepcopy(cfg)
            student_backbone = cfg.MODEL['BACKBONE']['TYPE']
            teacher_backbone = student_backbone.replace('_distilled', '')
            cfg_teacher.MODEL['BACKBONE']['TYPE'] = teacher_backbone

            # Teacher khong su dung refiner module cua student (RepBLGR)
            if hasattr(cfg_teacher.MODEL, "REFINER"):
                cfg_teacher.MODEL.REFINER.TYPE = "NONE"

            net_teacher = build_ortrack(cfg_teacher)

            # Xac dinh duong dan va load weights cua teacher (ho tro ca .zip lan .pth/.tar)
            teacher_ckpt_path = resolve_teacher_path(getattr(settings, "teacher_ckpt", None), cfg)
            checkpoint, meta_info = read_checkpoint(teacher_ckpt_path)
            report = load_teacher_weights(net_teacher, checkpoint)
            frozen_params = freeze_teacher(net_teacher)

            if settings.local_rank in [-1, 0]:
                print_teacher_report(report, meta_info)
                print(f"[TeacherLoader] Da dong bang {frozen_params:,} tham so cua teacher.")

            net_teacher.cuda()

        # Build model student voi day du refiner va branch distillation
        net = build_ortrack(cfg)
    else:
        raise ValueError("illegal script name")

    # wrap networks to distributed one
    net.cuda()
    if settings.local_rank != -1:
        # net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)  # add syncBN converter
        # Chu y: net_teacher khong co learnable parameters nen KHONG wrap DDP de tranh overhead/sync loi
        if net_teacher is not None:
            net_teacher.cuda(settings.local_rank)
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")
    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)
    settings.distill = getattr(cfg.TRAIN, "DISTILL", False)
    settings.distill_loss_type = getattr(cfg.TRAIN, "DISTILL_LOSS_TYPE", "KL")
    # Loss functions and Actors
    if settings.script_name == "ortrack":
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': 1.0,  'sim_loss': 0.0002, 'distill_loss': 0.00002}
        actor = ORTrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
        if is_distill_training:
            actor.net_teacher = net_teacher
    else:
        raise ValueError("illegal script name")

    # if cfg.TRAIN.DEEP_SUPERVISION:
    #     raise ValueError("Deep supervision is not supported now.")

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    use_amp = getattr(cfg.TRAIN, "AMP", False)
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler, use_amp=use_amp)

    if is_distill_training:
        trainer.actor.net.is_distill_training=True
    else:
        trainer.actor.net.is_distill_training = False

    # train process
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)
