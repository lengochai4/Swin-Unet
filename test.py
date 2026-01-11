import argparse
import logging
import os
import random
import sys
from os.path import split as path_split

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_config
from datasets.dataset_synapse import Synapse_dataset
from networks.vision_transformer import SwinUnet as ViT_seg

from scipy.ndimage import distance_transform_edt, binary_erosion


# =========================
# Metrics: Dice + HD95 (NumPy + SciPy)
# =========================
def dice_np(pred, gt):
    pred = pred.astype(np.bool_)
    gt = gt.astype(np.bool_)
    inter = np.logical_and(pred, gt).sum()
    den = pred.sum() + gt.sum()
    if den == 0:
        return 1.0  # both empty
    return float(2.0 * inter / den)


def hd95_np(pred, gt, spacing=(1.0, 1.0, 1.0)):
    pred = pred.astype(np.bool_)
    gt = gt.astype(np.bool_)

    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return np.nan

    dt_pred = distance_transform_edt(~pred, sampling=spacing)
    dt_gt = distance_transform_edt(~gt, sampling=spacing)

    pred_surf = np.logical_xor(pred, binary_erosion(pred))
    gt_surf = np.logical_xor(gt, binary_erosion(gt))

    d_pred_to_gt = dt_gt[pred_surf]
    d_gt_to_pred = dt_pred[gt_surf]

    if d_pred_to_gt.size == 0 or d_gt_to_pred.size == 0:
        return np.nan

    all_d = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return float(np.percentile(all_d, 95))


# =========================
# Inference on 1 volume 
# =========================
@torch.no_grad()
def test_single_volume(image, label, net, classes, patch_size=(224, 224),
                       test_save_path=None, case=None, z_spacing=1.0):
    """
    image, label: tensors from Synapse_dataset(test_vol)
      - commonly image: (1, 1, D, H, W) or (1, D, H, W)
      - label: same without channel
    Returns: metric_list = [(dice_c, hd95_c) for c=1..classes-1]
    Also saves prediction npy: {case}_pred.npy as label volume (D,H,W)
    """
    # sanitize case
    if isinstance(case, (list, tuple)):
        case = case[0]
    case = str(case).strip() if case is not None else "unknown"

    try:
        net.eval()
        device = next(net.parameters()).device

        # ---------- normalize shapes to (D,H,W) ----------
        img = image
        lab = label

        # image tensor can be on cpu from dataloader
        if isinstance(img, torch.Tensor):
            pass
        else:
            raise TypeError(f"image must be torch.Tensor, got {type(img)}")

        if isinstance(lab, torch.Tensor):
            pass
        else:
            raise TypeError(f"label must be torch.Tensor, got {type(lab)}")

        # remove batch dim
        if img.dim() == 5:          # (B,C,D,H,W)
            img = img[0]
        elif img.dim() == 4:        # (B,D,H,W) or (B,C,H,W)
            img = img[0]
        else:
            raise RuntimeError(f"Unexpected image dim: {img.shape}")

        if lab.dim() == 4:          # (B,D,H,W)
            lab = lab[0]
        elif lab.dim() == 5:        # unlikely
            lab = lab[0]
        else:
            raise RuntimeError(f"Unexpected label dim: {lab.shape}")

        # if image is (C,D,H,W) -> make sure C exists
        if img.dim() == 4:
            # could be (C,D,H,W) or (D,H,W,?) not in torch; assume (C,D,H,W)
            if img.shape[0] in (1, 3):   # channel-first
                c = img.shape[0]
                D, H, W = img.shape[1], img.shape[2], img.shape[3]
            else:
                # fallback: treat as (D,H,W,?) not expected
                raise RuntimeError(f"Ambiguous 4D image shape: {img.shape}")
        elif img.dim() == 3:
            # (D,H,W) -> add channel=1
            img = img.unsqueeze(0)  # (1,D,H,W)
            c = 1
            D, H, W = img.shape[1], img.shape[2], img.shape[3]
        else:
            raise RuntimeError(f"Unexpected image shape after squeeze: {img.shape}")

        # label should be (D,H,W)
        if lab.dim() == 3:
            pass
        elif lab.dim() == 4:
            # sometimes label becomes (1,D,H,W)
            if lab.shape[0] == 1:
                lab = lab[0]
            else:
                raise RuntimeError(f"Unexpected label 4D shape: {lab.shape}")
        else:
            raise RuntimeError(f"Unexpected label shape: {lab.shape}")

        # ---------- slice-by-slice inference ----------
        pred_vol = np.zeros((D, H, W), dtype=np.int16)

        for z in range(D):
            # slice: (C,H,W)
            slice_img = img[:, z, :, :]  # (C,H,W)
            # make (1,C,H,W)
            slice_img = slice_img.unsqueeze(0).to(device)

            # resize to patch_size for model
            if slice_img.shape[-2:] != patch_size:
                slice_in = F.interpolate(slice_img, size=patch_size, mode="bilinear", align_corners=False)
            else:
                slice_in = slice_img

            logits = net(slice_in)  # (1, classes, ph, pw)
            if logits.dim() != 4 or logits.shape[1] != classes:
                raise RuntimeError(f"Unexpected logits shape: {logits.shape}")

            # argmax on patch_size
            pred_patch = torch.argmax(logits, dim=1, keepdim=True).float()  # (1,1,ph,pw)

            # resize back to original H,W using nearest
            if pred_patch.shape[-2:] != (H, W):
                pred_hw = F.interpolate(pred_patch, size=(H, W), mode="nearest")
            else:
                pred_hw = pred_patch

            pred_slice = pred_hw[0, 0].to("cpu").numpy().astype(np.int16)  # (H,W)
            pred_vol[z] = pred_slice

        gt_vol = lab.to("cpu").numpy().astype(np.int16)  # (D,H,W)

        # ---------- compute metrics per class (ignore background=0) ----------
        metric_list = []
        spacing = (1.0, 1.0, float(z_spacing))  # (x,y,z) — voxel units; OK even if approx
        for c in range(1, classes):
            d = dice_np(pred_vol == c, gt_vol == c)
            h = hd95_np(pred_vol == c, gt_vol == c, spacing=spacing)
            metric_list.append((float(d), float(h) if np.isfinite(h) else float("nan")))

        # ---------- save prediction ----------
        if test_save_path is not None:
            os.makedirs(test_save_path, exist_ok=True)
            np.save(os.path.join(test_save_path, f"{case}_pred.npy"), pred_vol)

        return metric_list

    except Exception:
        import traceback
        print(f"\n[ERROR] test_single_volume failed for case={repr(case)}")
        print(traceback.format_exc())
        return [(float("nan"), float("nan")) for _ in range(classes - 1)]


# =========================
# Args compatibility for config.py
# =========================
def ensure_config_args(args):
    defaults = {
        "opts": None,
        "zip": False,
        "cache_mode": "part",
        "resume": None,
        "accumulation_steps": None,
        "use_checkpoint": False,
        "amp_opt_level": "O1",
        "tag": None,
        "eval": False,
        "throughput": False,
    }
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)
    return args


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_path", type=str, default="/content/Synapse",
                        help="root dir for dataset (contains train_npz/ and test_vol_h5/)")
    parser.add_argument("--dataset", type=str, default="Synapse")
    parser.add_argument("--n_class", type=int, default=9, help="num classes incl background")
    parser.add_argument("--num_classes", type=int, default=9)
    parser.add_argument("--list_dir", type=str, default="./lists/Synapse",
                        help="dir containing train.txt / val.txt / test_vol.txt")
    parser.add_argument("--output_dir", type=str, required=True, help="output dir (where best_model.pth is)")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--deterministic", type=int, default=1)
    parser.add_argument("--is_savenii", action="store_true",
                        help="save predictions as npy into output_dir/predictions")

    parser.add_argument("--cfg", type=str, required=True, metavar="FILE", help="path to config yaml")
    parser.add_argument("--opts", default=None, nargs="+",
                        help="Modify config options by adding 'KEY VALUE' pairs.")
    parser.add_argument("--zip", default=False, type=lambda x: str(x).lower() in ["1", "true", "yes"],
                        help="dummy for config.py compatibility")

    parser.add_argument("--resume", type=str, default=None,
                        help="checkpoint path. default: output_dir/best_model.pth or last_model.pth")
    parser.add_argument("--split_name", type=str, default="test_vol",
                        help="txt name without extension. default reads test_vol.txt")
    parser.add_argument("--z_spacing", type=float, default=1.0)

    # dummy args for config compatibility
    parser.add_argument("--cache-mode", dest="cache_mode", type=str, default="part",
                        choices=["no", "full", "part"])
    parser.add_argument("--accumulation-steps", dest="accumulation_steps", type=int, default=None)
    parser.add_argument("--use-checkpoint", dest="use_checkpoint", action="store_true")
    parser.add_argument("--amp-opt-level", dest="amp_opt_level", type=str, default="O1",
                        choices=["O0", "O1", "O2"])
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--throughput", action="store_true")

    args = parser.parse_args()
    args = ensure_config_args(args)

    # deterministic
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.num_classes = int(args.n_class)

    # build config & model
    config = get_config(args)
    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()

    # checkpoint
    ckpt_path = args.resume
    if ckpt_path is None:
        ckpt_path = os.path.join(args.output_dir, "best_model.pth")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(args.output_dir, "last_model.pth")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cuda")
    msg = net.load_state_dict(state, strict=True)

    print("Loaded checkpoint:", ckpt_path)
    print("load_state_dict:", msg)

    # logging (file + console)
    log_folder = os.path.join(args.output_dir, "test_log")
    os.makedirs(log_folder, exist_ok=True)
    log_file = os.path.join(log_folder, os.path.basename(ckpt_path) + ".txt")

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info("Checkpoint: %s", ckpt_path)

    # dataset for volume testing
    volume_root = os.path.join(args.root_path, "test_vol_h5")
    if not os.path.isdir(volume_root):
        raise FileNotFoundError(f"Missing test_vol_h5 folder: {volume_root}")

    db_test = Synapse_dataset(base_dir=volume_root, list_dir=args.list_dir,
                              split=args.split_name, transform=None)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=args.num_workers)

    logging.info("Test volumes: %d", len(testloader))
    print("Test volumes:", len(testloader))

    # save preds
    test_save_path = None
    if args.is_savenii:
        test_save_path = os.path.join(args.output_dir, "predictions")
        os.makedirs(test_save_path, exist_ok=True)

    # collect metrics
    all_case_metrics = []   # list of list[(dice,hd95)] length=(C-1)
    case_names = []

    for i_batch, sampled_batch in tqdm(enumerate(testloader), total=len(testloader), desc="Test"):
        image = sampled_batch["image"]
        label = sampled_batch["label"]
        case_name = sampled_batch["case_name"][0]

        # sanitize case name
        case_name = str(case_name).strip()
        if "," in case_name:
            case_name = path_split(case_name.split(",")[0])[-1]
        case_name = case_name.strip()

        metric_list = test_single_volume(
            image, label, net,
            classes=args.num_classes,
            patch_size=(args.img_size, args.img_size),
            test_save_path=test_save_path,
            case=case_name,
            z_spacing=args.z_spacing
        )

        # ensure list[(dice,hd95)]
        if metric_list is None:
            metric_list = [(float("nan"), float("nan")) for _ in range(args.num_classes - 1)]

        # if a single tuple accidentally returned
        if isinstance(metric_list, tuple) and len(metric_list) == 2 and not isinstance(metric_list[0], (tuple, list, np.ndarray)):
            metric_list = [metric_list]

        # force array (C-1,2)
        metric_arr = np.asarray(metric_list, dtype=np.float32)
        if metric_arr.ndim == 1 and metric_arr.size == 2:
            metric_arr = metric_arr.reshape(1, 2)
        if metric_arr.ndim != 2 or metric_arr.shape[1] != 2:
            raise RuntimeError(f"Bad metric_arr shape for case={case_name}: {metric_arr.shape}, metric_list={metric_list}")

        all_case_metrics.append(metric_arr.tolist())
        case_names.append(case_name)

        mean_dice = float(np.nanmean(metric_arr[:, 0]))
        mean_hd95 = float(np.nanmean(metric_arr[:, 1]))
        logging.info("Case %s: mean_dice=%.6f mean_hd95=%.6f", case_name, mean_dice, mean_hd95)

    # -------- summary --------
    arr = np.asarray(all_case_metrics, dtype=np.float32)  # (N, C-1, 2)

    case_mean_dice = np.nanmean(arr[:, :, 0], axis=1)  # (N,)
    case_mean_hd95 = np.nanmean(arr[:, :, 1], axis=1)  # (N,)

    logging.info("\n=== Per-case (mean over classes, ignore bg) ===")
    for name, d, h in zip(case_names, case_mean_dice, case_mean_hd95):
        logging.info("%s: Dice=%.6f | HD95=%.6f", name, float(d), float(h))

    logging.info("\n=== Per-class (mean over cases) ===")
    for c in range(1, args.num_classes):
        d = float(np.nanmean(arr[:, c - 1, 0]))
        h = float(np.nanmean(arr[:, c - 1, 1]))
        logging.info("Class %d: dice=%.6f hd95=%.6f", c, d, h)

    performance = float(np.nanmean(case_mean_dice))
    mean_hd95 = float(np.nanmean(case_mean_hd95))
    logging.info("\nFINAL: mean_dice=%.6f mean_hd95=%.6f", performance, mean_hd95)

    print("\n✅ Testing Finished!")
    print("FINAL mean_dice:", performance)
    print("FINAL mean_hd95:", mean_hd95)
    print("Log saved to:", log_file)
    if test_save_path is not None:
        print("Predictions saved to:", test_save_path)


if __name__ == "__main__":
    main()
