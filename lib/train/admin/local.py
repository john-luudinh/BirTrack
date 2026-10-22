import os

class EnvironmentSettings:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(current_dir, "../../.."))

        self.workspace_dir = os.environ.get('WORKSPACE_DIR', repo_root)
        self.tensorboard_dir = os.environ.get('TENSORBOARD_DIR', os.path.join(self.workspace_dir, 'tensorboard'))
        self.pretrained_networks = os.environ.get('PRETRAINED_DIR', os.path.join(self.workspace_dir, 'pretrained_models'))

        # Base data root: uu tien DATA_ROOT -> /mnt/DATA/tuld3/Datasets/tracking -> /workspace -> /data
        candidates = [
            os.environ.get('DATA_ROOT', ''),
            '/mnt/DATA/tuld3/Datasets/tracking',
            '/workspace/data/Raw_datasets',
            '/data',
            os.path.join(self.workspace_dir, 'data'),
        ]
        data_base = ''
        for c in candidates:
            if c and os.path.isdir(c):
                data_base = c
                break
        if not data_base:
            data_base = os.environ.get('DATA_ROOT', '/mnt/DATA/tuld3/Datasets/tracking')

        # 1. GOT-10k (Ho tro ca got10k lan got-10k)
        got10k_base = 'got10k' if os.path.isdir(os.path.join(data_base, 'got10k')) else 'got-10k'
        self.got10k_dir = os.environ.get('GOT10K_DIR', os.path.join(data_base, f'{got10k_base}/train'))
        self.got10k_val_dir = os.environ.get('GOT10K_VAL_DIR', os.path.join(data_base, f'{got10k_base}/val'))
        self.got10k_lmdb_dir = ''

        # 2. MS COCO 2017 (Ho tro ca Coco2017 lan coco2017)
        coco_folder = 'Coco2017' if os.path.isdir(os.path.join(data_base, 'Coco2017')) else 'coco2017'
        self.coco_dir = os.environ.get('COCO_DIR', os.path.join(data_base, coco_folder))
        self.coco_lmdb_dir = ''

        # 3. UAV123 (Ho tro ca uav123 lan UAV123)
        uav123_folder = 'uav123' if os.path.isdir(os.path.join(data_base, 'uav123')) else 'UAV123'
        self.uav123_dir = os.environ.get('UAV123_DIR', os.path.join(data_base, uav123_folder))
        self.uav123_lmdb_dir = ''

        # 4. UAV-Anti-UAV (Ho tro uav-anti-uav-processed lan UAV_Anti_UAV_processed)
        uav_anti_folder = 'uav-anti-uav-processed' if os.path.isdir(os.path.join(data_base, 'uav-anti-uav-processed')) else 'UAV_Anti_UAV_processed'
        self.uav_anti_uav = os.environ.get('UAV_ANTI_UAV_DIR', os.path.join(data_base, uav_anti_folder))

        # 5. ANTI300 (RGB) & ANTI300_IR (Thermal)
        if os.path.isdir(os.path.join(data_base, 'Data_processed/anti300_rgb')):
            anti300_base = os.path.join(data_base, 'Data_processed')
        else:
            anti300_base = os.path.join(data_base, 'anti300_processed')
        self.anti300_dir = os.environ.get('ANTI300_DIR', os.path.join(anti300_base, 'anti300_rgb'))
        self.anti300_ir_dir = os.environ.get('ANTI300_IR_DIR', os.path.join(anti300_base, 'anti300_ir'))

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
