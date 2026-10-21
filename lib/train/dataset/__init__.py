from .got10k import Got10k
from .anti_300 import ANTI300
from .uav_anti_uav import UAV_Anti_UAV
from .uav123 import UAV123

# Optional datasets that require additional dependencies like pycocotools/pandas/lmdb
try:
    from .coco_seq import MSCOCOSeq
    from .coco import MSCOCO
    from .coco_seq_lmdb import MSCOCOSeq_lmdb
except ImportError:
    pass

try:
    from .lasot import Lasot
    from .tracking_net import TrackingNet
    from .imagenetvid import ImagenetVID
    from .got10k_lmdb import Got10k_lmdb
    from .lasot_lmdb import Lasot_lmdb
    from .imagenetvid_lmdb import ImagenetVID_lmdb
    from .tracking_net_lmdb import TrackingNet_lmdb
except ImportError:
    pass
