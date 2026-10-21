import os

class EnvironmentSettings:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(current_dir, "../../.."))

        self.workspace_dir = os.environ.get('WORKSPACE_DIR', repo_root)
        self.tensorboard_dir = os.environ.get('TENSORBOARD_DIR', os.path.join(self.workspace_dir, 'tensorboard'))
        self.pretrained_networks = os.environ.get('PRETRAINED_DIR', os.path.join(self.workspace_dir, 'pretrained_models'))

        # Base data root: container /data or /workspace/data/Raw_datasets or local Data
        default_data_base = '/workspace/data/Raw_datasets'
        if not os.path.isdir(default_data_base):
            if os.path.isdir('/data'):
                default_data_base = '/data'
            else:
                default_data_base = os.path.join(self.workspace_dir, 'data')

        data_base = os.environ.get('DATA_ROOT', default_data_base)

        # 1. GOT-10k
        self.got10k_dir = os.environ.get('GOT10K_DIR', os.path.join(data_base, 'got-10k/train'))
        self.got10k_val_dir = os.environ.get('GOT10K_VAL_DIR', os.path.join(data_base, 'got-10k/val'))
        self.got10k_lmdb_dir = ''

        # 2. MS COCO 2017
        self.coco_dir = os.environ.get('COCO_DIR', os.path.join(data_base, 'coco2017'))
        self.coco_lmdb_dir = ''

        # 3. UAV123
        self.uav123_dir = os.environ.get('UAV123_DIR', os.path.join(data_base, 'UAV123'))
        self.uav123_lmdb_dir = ''

        # 4. UAV-Anti-UAV
        self.uav_anti_uav = os.environ.get('UAV_ANTI_UAV_DIR', os.path.join(data_base, 'UAV_Anti_UAV_processed'))

        # 5. ANTI300 (RGB)
        self.anti300_dir = os.environ.get('ANTI300_DIR', os.path.join(data_base, 'anti300_processed/anti300_rgb'))

        # 6. ANTI300_IR (Thermal Infrared)
        self.anti300_ir_dir = os.environ.get('ANTI300_IR_DIR', os.path.join(data_base, 'anti300_processed/anti300_ir'))

        # Other standard tracking datasets (optional)
        self.lasot_dir = os.environ.get('LASOT_DIR', os.path.join(data_base, 'lasot'))
        self.lasot_lmdb_dir = ''
        self.trackingnet_dir = os.environ.get('TRACKINGNET_DIR', os.path.join(data_base, 'trackingnet'))
        self.trackingnet_lmdb_dir = ''
        self.imagenet_dir = os.environ.get('IMAGENET_DIR', os.path.join(data_base, 'vid'))
        self.imagenet_lmdb_dir = ''

        # Checkpoints for Teacher Distillation (Phase 2)
        self.teacher_ckpt = os.environ.get(
            'TEACHER_CKPT',
            os.path.join(self.workspace_dir, 'output/teacher_uav/checkpoints/train/ortrack/deit_tiny_teacher_saorr_uav/ORTrack_ep0300.pth.tar')
        )
