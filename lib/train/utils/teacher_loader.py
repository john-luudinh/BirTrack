"""
Teacher Model Loader & Inspector for Knowledge Distillation (BiRTrack).

Module nay dam nhan:
1. Resolve duong dan teacher weights (CLI > ENV > YAML > default).
2. Doc truc tiep tu file `.zip` (giai nen in-memory qua BytesIO) hoac file `.pth/.tar`.
3. Load vao teacher model voi co che kiem tra chat che:
   - Bat buoc cac key cot loi (backbone, box_head) phai ton tai va khop shape.
   - Phat hien va bao cao missing keys, unexpected keys, shape mismatches.
   - Sanity check gia tri tensor sau khi load de dam bao du lieu that su duoc copy.
4. Freeze toan bo parameters cua teacher va chuyen sang eval mode.
"""

import os
import io
import zipfile
from typing import Dict, Any, Tuple, List, Optional
import torch
import torch.nn as nn


def resolve_teacher_path(cli_path: Optional[str] = None, cfg: Optional[Any] = None) -> str:
    """
    Xac dinh duong dan tuyet doi den checkpoint cua teacher model theo thu tu uu tien:
      1. Tham so CLI (--teacher_ckpt)
      2. Bien moi truong (TEACHER_CKPT)
      3. Cau hinh YAML (cfg.MODEL.TEACHER_CKPT)
      4. Thu muc mac dinh theo ten backbone (teacher_model/<backbone>/ORTrack_ep0300.pth.tar)

    Complexity: Time O(1), Space O(1)
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    candidates: List[Tuple[str, str]] = []

    if cli_path and str(cli_path).strip():
        candidates.append(("CLI (--teacher_ckpt)", str(cli_path).strip()))

    env_path = os.environ.get("TEACHER_CKPT", "").strip()
    if env_path:
        candidates.append(("Environment (TEACHER_CKPT)", env_path))

    if cfg is not None:
        yaml_path = getattr(cfg.MODEL, "TEACHER_CKPT", "")
        if yaml_path and str(yaml_path).strip():
            candidates.append(("Config YAML (MODEL.TEACHER_CKPT)", str(yaml_path).strip()))

    # Thu tung nguon theo thu tu uu tien
    for source_name, p in candidates:
        # Neu la duong dan tuong doi, resolve theo repo_root chu KHONG theo CWD hien tai
        abs_p = p if os.path.isabs(p) else os.path.normpath(os.path.join(repo_root, p))
        if os.path.isfile(abs_p):
            print(f"[TeacherLoader] Tim thay weights tu [{source_name}]: {abs_p}")
            return abs_p
        else:
            print(f"[TeacherLoader] Canh bao: [{source_name}] chi dinh '{p}' nhung file khong ton tai tai '{abs_p}'")

    # Neu tat ca deu khong tim thay, thu tim trong thu muc teacher_model/
    if cfg is not None and hasattr(cfg.MODEL, "BACKBONE"):
        backbone_name = str(cfg.MODEL.BACKBONE.TYPE).replace("_distilled", "")
        default_p = os.path.join(repo_root, f"teacher_model/{backbone_name}/ORTrack_ep0300.pth.tar")
        if os.path.isfile(default_p):
            print(f"[TeacherLoader] Su dung checkpoint mac dinh: {default_p}")
            return default_p

    raise FileNotFoundError(
        "[TeacherLoader] KHONG TIM THAY teacher checkpoint hop le!\n"
        "Cac nguon da kiem tra:\n" +
        "\n".join([f"  - {src}: {p}" for src, p in candidates]) +
        "\nVui long kiem tra lai duong dan trong file YAML hoac truyen qua --teacher_ckpt."
    )


def read_checkpoint(path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Doc checkpoint tu file .zip hoac .pth/.tar.

    Idiom: Su dung zipfile.ZipFile ket hop voi io.BytesIO de doc truc tiep member
    checkpoint vao RAM ma khong can giai nen ra o dia.
    - Time Complexity: O(M) voi M la dung luong file checkpoint (~100MB, mat ~0.5s).
    - Space Complexity: O(M) bo nho RAM dem BytesIO tam thoi.
    - Cam bay: Can truyen weights_only=False vi checkpoint cua PyTracking thuong chua dict
      gom cac kieu du lieu bo sung nhu epoch (int), net_type (str).
    """
    meta_info: Dict[str, Any] = {
        "file_path": path,
        "file_size_mb": round(os.path.getsize(path) / (1024 * 1024), 2),
        "is_zip": False,
        "zip_member": None,
    }

    if path.lower().endswith(".zip"):
        meta_info["is_zip"] = True
        with zipfile.ZipFile(path, "r") as z:
            # Tim file chua trong so (.pth.tar hoac .pth hoac .pt)
            ckpt_members = [
                m for m in z.namelist()
                if m.endswith((".pth.tar", ".pth", ".pt", ".bin")) and not m.startswith("__MACOSX")
            ]
            if not ckpt_members:
                raise ValueError(f"[TeacherLoader] File zip '{path}' khong chua file trong so hop le (.pth/.pth.tar). Cac file ben trong: {z.namelist()}")

            target_member = ckpt_members[0]
            meta_info["zip_member"] = target_member
            print(f"[TeacherLoader] Dang doc member '{target_member}' tu archive '{path}'...")
            raw_bytes = z.read(target_member)
            ckpt = torch.load(io.BytesIO(raw_bytes), map_location="cpu", weights_only=False)
    else:
        print(f"[TeacherLoader] Dang doc file checkpoint truc tiep: {path}...")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

    return ckpt, meta_info


def load_teacher_weights(
    net: nn.Module,
    checkpoint: Dict[str, Any],
    strict_prefixes: Tuple[str, ...] = ("backbone.", "box_head.")
) -> Dict[str, Any]:
    """
    Nap trong so vao model teacher va kiem tra toan ven.

    Thuat toan:
    1. Trinh trich state_dict (thuong nam o key 'net' hoac truc tiep o root).
    2. Doi chieu tap key giua model va checkpoint:
       - missing_keys = model_keys - ckpt_keys
       - unexpected_keys = ckpt_keys - model_keys
       - shape_mismatches = {k: (ckpt_shape, model_shape)}
    3. Rang buoc chat che (Strict Check):
       Neu bat ky key nao bat dau bang `strict_prefixes` bi missing hoac sai shape,
       lap tuc raise loi de tranh truong hop distill voi teacher chua weights ngau nhien!
    4. Sanity check: So sanh gia tri tensor thuc te sau khi load voi checkpoint ban dau qua torch.equal.

    Complexity:
      - Time: O(P) voi P la tong so tham so (~5-10M floats, ~0.1s).
      - Space: O(P) de luu giu state_dict.
    """
    model_sd = net.state_dict()
    ckpt_sd = checkpoint["net"] if "net" in checkpoint else checkpoint

    model_keys = set(model_sd.keys())
    ckpt_keys = set(ckpt_sd.keys())

    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys

    shape_mismatches: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]] = {}
    for k in model_keys.intersection(ckpt_keys):
        if model_sd[k].shape != ckpt_sd[k].shape:
            shape_mismatches[k] = (tuple(ckpt_sd[k].shape), tuple(model_sd[k].shape))

    # Kiem tra xem co missing key nao thuoc nhom cot loi khong
    critical_missing = [k for k in missing if any(k.startswith(pfx) for pfx in strict_prefixes)]

    if critical_missing:
        sample_missing = critical_missing[:10]
        raise RuntimeError(
            f"[TeacherLoader] LOI NGHIEP TRONG: Co {len(critical_missing)} keys cot loi bi thieu khi load teacher!\n"
            f"Vi du missing keys: {sample_missing}\n"
            "Dung qua trinh training de tranh distill voi teacher chua weights ngau nhien."
        )

    if shape_mismatches:
        raise RuntimeError(
            f"[TeacherLoader] LOI NGHIEP TRONG: Phat hien {len(shape_mismatches)} keys bi lech shape giua checkpoint va teacher model!\n"
            f"Chi tiet: {shape_mismatches}"
        )

    # Thuc hien nap trong so (strict=False de cho phep bo qua cac key phu khong thuoc model neu co)
    load_res = net.load_state_dict(ckpt_sd, strict=False)

    # Sanity check: Chon ngau nhien 3 tensor de xac nhan du lieu da duoc copy chinh xac vao module
    verify_keys = [k for k in model_keys.intersection(ckpt_keys) if "weight" in k][:3]
    verified_count = 0
    for vk in verify_keys:
        if torch.equal(net.state_dict()[vk], ckpt_sd[vk]):
            verified_count += 1
        else:
            raise RuntimeError(f"[TeacherLoader] Sanity check that bai tai key: {vk}!")

    report: Dict[str, Any] = {
        "total_model_keys": len(model_keys),
        "total_ckpt_keys": len(ckpt_keys),
        "matched_keys": len(model_keys.intersection(ckpt_keys)),
        "missing_keys": sorted(list(missing)),
        "unexpected_keys": sorted(list(unexpected)),
        "shape_mismatches": shape_mismatches,
        "verified_sample_tensors": f"{verified_count}/{len(verify_keys)} OK",
        "epoch": checkpoint.get("epoch", "N/A"),
        "net_type": checkpoint.get("net_type", "N/A"),
    }
    return report


def freeze_teacher(net: nn.Module) -> int:
    """
    Dong bang toan bo parameter va chuyen model sang che do eval.
    - eval(): vo hieu hoa dropout va co dinh running mean/var cua BatchNorm.
    - requires_grad = False: tranh ton VRAM luu computation graph va gradient cua teacher.

    Returns:
        total_frozen_params: Tong so luong tham so da duoc dong bang.
    """
    net.eval()
    total_params = 0
    for param in net.parameters():
        param.requires_grad = False
        total_params += param.numel()
    return total_params


def print_teacher_report(report: Dict[str, Any], meta_info: Dict[str, Any]):
    """
    In bang thong tin xac nhan teacher weights load thanh cong ra console.
    """
    print("\n" + "=" * 65)
    print("           TEACHER MODEL WEIGHTS VERIFICATION REPORT")
    print("=" * 65)
    print(f" Source File     : {meta_info.get('file_path')}")
    if meta_info.get("is_zip"):
        print(f" Archive Member  : {meta_info.get('zip_member')}")
    print(f" File Size       : {meta_info.get('file_size_mb')} MB")
    print(f" Checkpoint Info : Epoch {report.get('epoch')}, Type: {report.get('net_type')}")
    print("-" * 65)
    print(f" Model Keys Total: {report.get('total_model_keys')}")
    print(f" Matched Keys    : {report.get('matched_keys')} (All critical keys loaded)")
    print(f" Missing Keys    : {len(report.get('missing_keys'))}")
    if report.get('missing_keys'):
        print(f"   -> Details    : {report.get('missing_keys')[:5]}")
    print(f" Unexpected Keys : {len(report.get('unexpected_keys'))}")
    if report.get('unexpected_keys'):
        print(f"   -> Details    : {report.get('unexpected_keys')[:5]}")
    print(f" Shape Mismatches: {len(report.get('shape_mismatches'))}")
    print(f" Tensor Sanity   : {report.get('verified_sample_tensors')}")
    print(" Status          : [PASSED] Teacher loaded & verified successfully!")
    print("=" * 65 + "\n")
