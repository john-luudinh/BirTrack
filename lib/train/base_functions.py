import torch
from torch.utils.data.distributed import DistributedSampler
from lib.train.dataset import Got10k
from lib.train.dataset import UAV123, ANTI300, UAV_Anti_UAV
try:
    from lib.train.dataset import MSCOCOSeq
except ImportError:
    MSCOCOSeq = None
try:
    from lib.train.dataset import Lasot, ImagenetVID, TrackingNet
    from lib.train.dataset import Lasot_lmdb, Got10k_lmdb, MSCOCOSeq_lmdb, ImagenetVID_lmdb, TrackingNet_lmdb
except ImportError:
    pass
from lib.train.data import sampler, opencv_loader, processing, LTRLoader
import lib.train.data.transforms as tfm
from lib.utils.misc import is_main_process


def update_settings(settings, cfg):
    settings.print_interval = cfg.TRAIN.PRINT_INTERVAL
    settings.search_area_factor = {'template': cfg.DATA.TEMPLATE.FACTOR,
                                   'search': cfg.DATA.SEARCH.FACTOR}
    settings.output_sz = {'template': cfg.DATA.TEMPLATE.SIZE,
                          'search': cfg.DATA.SEARCH.SIZE}
    settings.center_jitter_factor = {'template': cfg.DATA.TEMPLATE.CENTER_JITTER,
                                     'search': cfg.DATA.SEARCH.CENTER_JITTER}
    settings.scale_jitter_factor = {'template': cfg.DATA.TEMPLATE.SCALE_JITTER,
                                    'search': cfg.DATA.SEARCH.SCALE_JITTER}
    settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM
    settings.print_stats = None
    settings.batchsize = cfg.TRAIN.BATCH_SIZE
    settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE


def names2datasets(name_list: list, settings, image_loader):
    assert isinstance(name_list, list)
    datasets = []
    for name in name_list:
        assert name in [
            "LASOT", "GOT10K_vottrain", "GOT10K_votval", "GOT10K_train_full", "GOT10K_official_val",
            "COCO17", "COCO17_val", "COCO_val", "VID", "TRACKINGNET", "UAV123", "UAV_Anti_UAV", "UAV_Anti_UAV_val",
            "ANTI300", "ANTI300_val", "ANTI300_valid", "ANTI300_IR", "ANTI300_IR_val", "ANTI300_IR_valid"
        ]
        if name == "LASOT":
            if settings.use_lmdb:
                print("Building lasot dataset from lmdb")
                datasets.append(Lasot_lmdb(settings.env.lasot_lmdb_dir, split='train', image_loader=image_loader))
            else:
                datasets.append(Lasot(settings.env.lasot_dir, split='train', image_loader=image_loader))
        if name == "GOT10K_vottrain":
            if settings.use_lmdb:
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='vottrain', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='vottrain', image_loader=image_loader))
        if name == "GOT10K_train_full":
            if settings.use_lmdb:
                print("Building got10k_train_full from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='train_full', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='train_full', image_loader=image_loader))
        if name == "GOT10K_votval":
            if settings.use_lmdb:
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='votval', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='votval', image_loader=image_loader))
        if name == "GOT10K_official_val":
            if settings.use_lmdb:
                raise ValueError("Not implement")
            else:
                datasets.append(Got10k(settings.env.got10k_val_dir, split=None, image_loader=image_loader))
        if name == "COCO17":
            if settings.use_lmdb:
                print("Building COCO2017 from lmdb")
                datasets.append(MSCOCOSeq_lmdb(settings.env.coco_lmdb_dir, version="2017", image_loader=image_loader))
            else:
                datasets.append(MSCOCOSeq(settings.env.coco_dir, version="2017", image_loader=image_loader))
        if name in ["COCO17_val", "COCO_val"]:
            datasets.append(MSCOCOSeq(settings.env.coco_dir, version="2017", split="val", image_loader=image_loader))
        if name == "VID":
            if settings.use_lmdb:
                print("Building VID from lmdb")
                datasets.append(ImagenetVID_lmdb(settings.env.imagenet_lmdb_dir, image_loader=image_loader))
            else:
                datasets.append(ImagenetVID(settings.env.imagenet_dir, image_loader=image_loader))
        if name == "TRACKINGNET":
            if settings.use_lmdb:
                print("Building TrackingNet from lmdb")
                datasets.append(TrackingNet_lmdb(settings.env.trackingnet_lmdb_dir, image_loader=image_loader))
            else:
                # raise ValueError("NOW WE CAN ONLY USE TRACKINGNET FROM LMDB")
                datasets.append(TrackingNet(settings.env.trackingnet_dir, image_loader=image_loader))
        if name == "UAV123":
            datasets.append(UAV123(settings.env.uav123_dir, image_loader=image_loader))
        if name in ["UAV_Anti_UAV", "UAV_ANTI_UAV"]:
            datasets.append(UAV_Anti_UAV(settings.env.uav_anti_uav, split="train", image_loader=image_loader))
        if name in ["UAV_Anti_UAV_val", "UAV_ANTI_UAV_val"]:
            datasets.append(UAV_Anti_UAV(settings.env.uav_anti_uav, split="test", image_loader=image_loader))
        if name == "ANTI300":
            datasets.append(ANTI300(settings.env.anti300_dir, split="train", image_loader=image_loader))
        if name in ["ANTI300_val", "ANTI300_valid"]:
            datasets.append(ANTI300(settings.env.anti300_dir, split="valid", image_loader=image_loader))
        if name == "ANTI300_IR":
            datasets.append(ANTI300(settings.env.anti300_ir_dir, split="train", image_loader=image_loader))
        if name in ["ANTI300_IR_val", "ANTI300_IR_valid"]:
            datasets.append(ANTI300(settings.env.anti300_ir_dir, split="valid", image_loader=image_loader))
    return datasets


def summarize_datasets(name_list, datasets, ratios=None, phase="TRAIN"):
    """
    In bang tong ket chi tiet cac dataset da duoc khoi tao:
    Bao gom ten trong config, class Python tuong ung, root path, so luong sequence, va ty le sampling.
    Canh bao nghiem trong neu co dataset rong (0 sequence).

    Complexity: Time O(D) voi D la so dataset (~6), Space O(1).
    """
    total_ratio = sum(ratios) if ratios is not None and sum(ratios) > 0 else len(datasets)
    print("\n" + "=" * 105)
    print(f"                               DATASET INVENTORY REPORT ({phase.upper()})")
    print("=" * 105)
    header = f" {'No':<3} | {'Config Name':<18} | {'Class Name':<15} | {'Root Directory':<42} | {'# Seq':>8} | {'Ratio':>7} "
    print(header)
    print("-" * 105)

    total_seqs = 0
    empty_datasets = []

    for i, (name, ds) in enumerate(zip(name_list, datasets)):
        ds_class = ds.__class__.__name__
        root_dir = getattr(ds, "root", getattr(ds, "dataset_dir", "N/A"))
        if len(root_dir) > 42:
            display_root = "..." + root_dir[-39:]
        else:
            display_root = root_dir

        num_seq = ds.get_num_sequences() if hasattr(ds, "get_num_sequences") else len(ds)
        total_seqs += num_seq

        ratio_val = (ratios[i] / total_ratio) if ratios is not None and i < len(ratios) else (1.0 / len(datasets))
        row = f" {i + 1:<3} | {name:<18} | {ds_class:<15} | {display_root:<42} | {num_seq:>8,} | {ratio_val:>7.4f} "
        print(row)

        if num_seq == 0:
            empty_datasets.append(name)

    print("-" * 105)
    print(f" TOTAL SEQUENCES: {total_seqs:,} across {len(datasets)} dataset(s)")
    if empty_datasets:
        print(f" [CANH BAO NGUY HIEM] Co {len(empty_datasets)} dataset rong (0 sequences): {empty_datasets}!")
        print(" Vui long kiem tra lai duong dan va nhan du lieu.")
    print("=" * 105 + "\n")


def build_dataloaders(cfg, settings):
    # Data transform
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                    tfm.RandomHorizontalFlip(probability=0.5))

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                    tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    # The tracking pairs processing module
    output_sz = settings.output_sz
    search_area_factor = settings.search_area_factor

    data_processing_train = processing.STARKProcessing(search_area_factor=search_area_factor,
                                                       output_sz=output_sz,
                                                       center_jitter_factor=settings.center_jitter_factor,
                                                       scale_jitter_factor=settings.scale_jitter_factor,
                                                       mode='sequence',
                                                       transform=transform_train,
                                                       joint_transform=transform_joint,
                                                       settings=settings)

    data_processing_val = processing.STARKProcessing(search_area_factor=search_area_factor,
                                                     output_sz=output_sz,
                                                     center_jitter_factor=settings.center_jitter_factor,
                                                     scale_jitter_factor=settings.scale_jitter_factor,
                                                     mode='sequence',
                                                     transform=transform_val,
                                                     joint_transform=transform_joint,
                                                     settings=settings)

    # Train sampler and loader
    settings.num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 1)
    settings.num_search = getattr(cfg.DATA.SEARCH, "NUMBER", 1)
    sampler_mode = getattr(cfg.DATA, "SAMPLER_MODE", "causal")
    train_cls = getattr(cfg.TRAIN, "TRAIN_CLS", False)
    print("sampler_mode", sampler_mode)

    datasets_train = names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, opencv_loader)
    if is_main_process():
        summarize_datasets(cfg.DATA.TRAIN.DATASETS_NAME, datasets_train, cfg.DATA.TRAIN.DATASETS_RATIO, phase="TRAIN")

    dataset_train = sampler.TrackingSampler(datasets=datasets_train,
                                            p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
                                            samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
                                            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                            num_template_frames=settings.num_template, processing=data_processing_train,
                                            frame_sample_mode=sampler_mode, train_cls=train_cls)

    train_sampler = DistributedSampler(dataset_train) if settings.local_rank != -1 else None
    shuffle = False if settings.local_rank != -1 else True

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=shuffle,
                             num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=train_sampler)

    # Validation samplers and loaders
    datasets_val = names2datasets(cfg.DATA.VAL.DATASETS_NAME, settings, opencv_loader)
    if is_main_process():
        summarize_datasets(cfg.DATA.VAL.DATASETS_NAME, datasets_val, cfg.DATA.VAL.DATASETS_RATIO, phase="VAL")

    dataset_val = sampler.TrackingSampler(datasets=datasets_val,
                                          p_datasets=cfg.DATA.VAL.DATASETS_RATIO,
                                          samples_per_epoch=cfg.DATA.VAL.SAMPLE_PER_EPOCH,
                                          max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                          num_template_frames=settings.num_template, processing=data_processing_val,
                                          frame_sample_mode=sampler_mode, train_cls=train_cls)
    val_sampler = DistributedSampler(dataset_val) if settings.local_rank != -1 else None
    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=cfg.TRAIN.BATCH_SIZE,
                           num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=val_sampler,
                           epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL)

    return loader_train, loader_val


def get_optimizer_scheduler(net, cfg):
    train_cls = getattr(cfg.TRAIN, "TRAIN_CLS", False)
    if train_cls:
        print("Only training classification head. Learnable parameters are shown below.")
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "cls" in n and p.requires_grad]}
        ]

        for n, p in net.named_parameters():
            if "cls" not in n:
                p.requires_grad = False
            else:
                print(n)
    else:
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in net.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
            },
        ]
        if is_main_process():
            print("Learnable parameters are shown below.")
            for n, p in net.named_parameters():
                if p.requires_grad:
                    print(n)

    if cfg.TRAIN.OPTIMIZER == "ADAMW":
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR,
                                      weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        raise ValueError("Unsupported Optimizer")
    if cfg.TRAIN.SCHEDULER.TYPE == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP_EPOCH)
    elif cfg.TRAIN.SCHEDULER.TYPE == "Mstep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
                                                            gamma=cfg.TRAIN.SCHEDULER.GAMMA)
    else:
        raise ValueError("Unsupported scheduler")
    return optimizer, lr_scheduler
