"""
Verify Reparameterization Equivalence and Benchmark Latency/FPS for BiRTrack
Usage:
    python tools/verify_reparam.py [--checkpoint path/to/model.pth.tar]
"""
import sys
import os
import argparse
import time
import torch
import torch.nn as nn

# Ensure project root is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from lib.models.layers.rep_blgr import RepBLGR, RepDWConv
from lib.models.ortrack import build_ortrack
from lib.config.ortrack.config import cfg, update_config_from_file


def get_device(requested_device: str = "auto") -> torch.device:
    if requested_device == "cpu":
        return torch.device("cpu")
    if requested_device == "cuda" or (requested_device == "auto" and torch.cuda.is_available()):
        try:
            d = torch.device("cuda")
            _ = torch.zeros(1, device=d)
            return d
        except Exception as e:
            print(f"-> CUDA khong kha dung ({e}), chuyen sang CPU.")
            return torch.device("cpu")
    return torch.device("cpu")


def verify_module_reparam(device: torch.device):
    print("=" * 65)
    print("1. KIEM TRA TINH TUONG DUONG SO HOC CUA REP-BLGR (STANDALONE)")
    print("=" * 65)
    print(f"-> Su dung thiet bi: {device}")

    channels = 192
    H, W = 16, 16
    batch_size = 2

    for global_type in ["fasa", "se"]:
        for bidirectional in [False, True]:
            refiner = RepBLGR(
                dim=channels,
                num_heads=3,
                window_size=2,
                bidirectional=bidirectional,
                global_type=global_type,
                deploy=False
            ).to(device)

            # Assign non-zero gamma to test active residual path
            with torch.no_grad():
                refiner.gamma.fill_(1.0)

            refiner.train()
            # Feed some random batches to update BN statistics
            for _ in range(5):
                dummy = torch.randn(4, channels, H, W, device=device)
                _ = refiner(dummy)

            refiner.eval()
            x_test = torch.randn(batch_size, channels, H, W, device=device)

            with torch.no_grad():
                out_train = refiner(x_test)

            refiner.switch_to_deploy()
            assert refiner.deploy and refiner.local.deploy, "switch_to_deploy that bai!"

            with torch.no_grad():
                out_deploy = refiner(x_test)

            max_err = torch.max(torch.abs(out_train - out_deploy)).item()
            mean_err = torch.mean(torch.abs(out_train - out_deploy)).item()
            print(f"[{global_type.upper():4s} | Bi={str(bidirectional):5s}] "
                  f"Max error: {max_err:.8e} | Mean error: {mean_err:.8e}")
            assert max_err < 1e-4, f"Sai so vuot nguong: {max_err}"

    print("-> KET QUA: TAT CA CAC CHE DO REP-BLGR DAT CHUAN TUONG DUONG TUYET DOI (< 1e-4)!\n")


def benchmark_refiner_fps(device: torch.device):
    print("=" * 65)
    print("2. DO KIEM DO TRE (LATENCY) VA TOC DO (FPS)")
    print("=" * 65)

    if device.type != 'cuda':
        print("-> Khong su dung CUDA GPU (dang chay CPU)! Bo qua do FPS chuan GPU.")
        return

    channels = 192
    H, W = 16, 16
    batch_size = 1  # Standard UAV streaming inference
    num_warmup = 100
    num_runs = 500

    x = torch.randn(batch_size, channels, H, W, device=device)

    # 1. Measure Training-time structure (3 parallel branches)
    m_train = RepBLGR(dim=channels, deploy=False).to(device).eval()
    for _ in range(num_warmup):
        _ = m_train(x)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(num_runs):
        _ = m_train(x)
    torch.cuda.synchronize()
    lat_train = ((time.time() - t0) / num_runs) * 1000.0

    # 2. Measure Deploy structure (1 fused DWConv)
    m_deploy = RepBLGR(dim=channels, deploy=False).to(device).eval()
    m_deploy.switch_to_deploy()
    for _ in range(num_warmup):
        _ = m_deploy(x)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(num_runs):
        _ = m_deploy(x)
    torch.cuda.synchronize()
    lat_deploy = ((time.time() - t0) / num_runs) * 1000.0

    speedup = ((lat_train - lat_deploy) / lat_train) * 100.0
    print(f"-> Do tre khoi Rep-BLGR truoc khi gop (Train mode): {lat_train:.4f} ms ({1000.0 / lat_train:.1f} FPS)")
    print(f"-> Do tre khoi Rep-BLGR sau khi gop (Deploy mode): {lat_deploy:.4f} ms ({1000.0 / lat_deploy:.1f} FPS)")
    print(f"-> Ty le tang toc cua rieng khoi Rep-BLGR: {speedup:.2f}%\n")


def main():
    parser = argparse.ArgumentParser(description="Verify reparameterization equivalence and speedup")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to trained checkpoint (optional)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Compute device")
    args = parser.parse_args()

    dev = get_device(args.device)
    verify_module_reparam(dev)
    benchmark_refiner_fps(dev)


if __name__ == "__main__":
    main()
