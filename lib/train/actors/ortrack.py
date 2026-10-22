from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate


class ORTrackActor(BaseActor):
    """ Actor for training ORTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

        # Initialize SA-ORR if enabled
        net_module = self.net.module if hasattr(self.net, "module") else self.net
        if cfg is not None and getattr(cfg.TRAIN, "SA_ORR", None) and getattr(cfg.TRAIN.SA_ORR, "ENABLE", False):
            from lib.train.utils.sa_orr import SearchAwareCoxMasking
            self.sa_orr = SearchAwareCoxMasking(
                patch_size=cfg.MODEL.BACKBONE.STRIDE,
                mask_ratio_z=getattr(cfg.TRAIN.SA_ORR, "MASK_Z", 0.30),
                mask_ratio_x_in_box=getattr(cfg.TRAIN.SA_ORR, "MASK_X_IN_BOX", 0.30),
                sigma_scale=getattr(cfg.TRAIN.SA_ORR, "SIGMA_SCALE", 0.5),
                bg_weight=getattr(cfg.TRAIN.SA_ORR, "BG_WEIGHT", 0.2)
            )
            if hasattr(net_module, "use_internal_orr"):
                net_module.use_internal_orr = False
        else:
            self.sa_orr = None

        # Initialize MI-AFKD if enabled
        if cfg is not None and getattr(cfg.TRAIN, "MI_AFKD", None) and getattr(cfg.TRAIN.MI_AFKD, "ENABLE", False):
            from lib.train.losses.mi_afkd import DifficultyAdaptiveMIDistillationLoss
            self.mi_afkd = DifficultyAdaptiveMIDistillationLoss(
                tau_0=10.0,
                rho=10.0,
                temperature=getattr(cfg.TRAIN.MI_AFKD, "TAU", 2.0),
                mi_weight=getattr(cfg.TRAIN.MI_AFKD, "MI_WEIGHT", 0.5),
                use_ema=getattr(cfg.TRAIN.MI_AFKD, "USE_EMA", False)
            )
        else:
            self.mi_afkd = None

    def train(self, mode=True):
        """
        Giu nguyen student o mode duoc chi dinh (train/eval), nhung LUON LUON
        bat buoc teacher o mode eval() de dong bang BatchNorm running stats va Dropout.
        """
        super().train(mode)
        if hasattr(self, "net_teacher") and self.net_teacher is not None:
            self.net_teacher.eval()

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        # currently only support 1 template and 1 search region
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1

        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            template_list.append(template_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 256, 256)

        if len(template_list) == 1:
            template_list = template_list[0]

        # ====distillation====
        out_dict_teacher = None
        if self.net.is_distill_training and hasattr(self, "net_teacher") and self.net_teacher is not None:
            with torch.no_grad():
                out_dict_teacher = self.net_teacher(template=template_list,
                                                    search=search_img, is_distill=True)
        # ==========================

        out_dict = self.net(template=template_list,
                            search=search_img, is_distill=False)

        # Store teacher outputs for distillation loss computation
        if out_dict_teacher is not None:
            out_dict['teacher_out'] = out_dict_teacher

        # SA-ORR: Search-Aware Cox Masking consistency
        if self.sa_orr is not None and self.net.training:
            gt_bbox = data['search_anno'][-1]
            z_m, x_m, w_x = self.sa_orr(template_list, search_img, gt_bbox)
            with torch.no_grad():
                out_dict_masked = self.net(template=z_m, search=x_m, is_distill=True)

            feat_clean = out_dict['backbone_feat']
            feat_masked = out_dict_masked['backbone_feat']

            net_module = self.net.module if hasattr(self.net, "module") else self.net
            feat_len_t = getattr(net_module, "feat_len_t", 64)

            # 1. Template consistency
            loss_z = torch.nn.functional.mse_loss(
                feat_clean[:, :feat_len_t], feat_masked[:, :feat_len_t].detach()
            )
            # 2. Search consistency weighted by target spatial mask
            lambda_s = getattr(self.cfg.TRAIN.SA_ORR, "LAMBDA_S", 0.5)
            diff_x = (feat_clean[:, feat_len_t:] - feat_masked[:, feat_len_t:].detach()) ** 2
            loss_x = (diff_x * w_x).mean()

            out_dict['sa_orr_loss'] = loss_z + lambda_s * loss_x

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou, giou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou, giou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)

        # Baseline detection loss
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss
        status = {}

        # ORR / SA-ORR loss
        if self.sa_orr is not None and 'sa_orr_loss' in pred_dict:
            sa_orr_loss = pred_dict['sa_orr_loss']
            w_sa_orr = getattr(self.cfg.TRAIN.SA_ORR, "WEIGHT", 0.0002)
            loss += w_sa_orr * sa_orr_loss
            status["Loss/sa_orr"] = sa_orr_loss.item()
        elif 'sim_loss' in pred_dict and self.loss_weight.get('sim_loss', 0) > 0:
            sim_loss = pred_dict['sim_loss']
            loss += self.loss_weight['sim_loss'] * sim_loss

        # Distillation loss
        if self.net.is_distill_training and 'teacher_out' in pred_dict:
            out_teacher = pred_dict['teacher_out']
            feat_teacher = out_teacher['backbone_feat']
            feat_student = pred_dict['backbone_feat']

            if self.mi_afkd is not None:
                # MI-AFKD with JSD and adaptive difficulty
                refined_s = pred_dict.get('search_feat_2d', None)
                refined_t = out_teacher.get('search_feat_2d', None)
                if refined_s is None or refined_t is None:
                    # fallback to search tokens reshaped to 2D
                    net_module = self.net.module if hasattr(self.net, "module") else self.net
                    feat_sz_s = getattr(net_module, "feat_sz_s", 16)
                    C = feat_student.shape[-1]
                    refined_s = feat_student[:, -feat_sz_s**2:].transpose(1, 2).view(-1, C, feat_sz_s, feat_sz_s)
                    refined_t = feat_teacher[:, -feat_sz_s**2:].transpose(1, 2).view(-1, C, feat_sz_s, feat_sz_s)

                distill_loss, loss_mse, loss_mi = self.mi_afkd(
                    tokens_student=feat_student,
                    tokens_teacher=feat_teacher,
                    refined_2d_student=refined_s,
                    refined_2d_teacher=refined_t,
                    giou_scores=giou
                )
                w_distill = self.loss_weight.get('distill_loss', 0.00002)
                loss += w_distill * distill_loss
                status["Loss/distill_total"] = distill_loss.item()
                status["Loss/distill_mse"] = loss_mse.item()
                status["Loss/distill_mi"] = loss_mi.item()
            else:
                # Fixed standard AFKD with proper gradient flow (Bug F3 fix)
                distill_mse = ((feat_student - feat_teacher.detach()) ** 2).mean(dim=(1, 2))  # [B]
                tau_0 = 10.0
                rho = 10.0
                coef = self.loss_weight['distill_loss'] * (tau_0 + rho * ((1.0 - giou) - (1.0 - giou).mean()))
                distill_loss = (coef * distill_mse).mean()
                loss += distill_loss
                status["Loss/distill"] = distill_loss.item()

        if return_status:
            mean_iou = iou.detach().mean()
            status.update({
                "Loss/total": loss.item(),
                "Loss/giou": giou_loss.item(),
                "Loss/l1": l1_loss.item(),
                "Loss/location": location_loss.item(),
                "IoU": mean_iou.item()
            })
            return loss, status
        else:
            return loss
