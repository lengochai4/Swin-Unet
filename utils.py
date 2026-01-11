import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import zoom, distance_transform_edt, binary_erosion

# =========================
# DiceLoss 
# =========================
class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = (input_tensor == i)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1.0 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)

        if weight is None:
            weight = [1.0] * self.n_classes

        assert inputs.size() == target.size(), (
            f"predict {inputs.size()} & target {target.size()} shape do not match"
        )

        loss = 0.0
        for i in range(self.n_classes):
            dice_i = self._dice_loss(inputs[:, i], target[:, i])
            loss += dice_i * weight[i]
        return loss / self.n_classes


# =========================
# Metrics (NumPy + SciPy) 
# =========================
def dice_numpy(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-5) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    return float((2.0 * inter + eps) / (pred.sum() + gt.sum() + eps))


def _surface_distances(a: np.ndarray, b: np.ndarray, spacing=None) -> np.ndarray:
    """
    Distance from surface of a to nearest surface of b using distance transform.
    a, b are boolean.
    spacing: tuple like (sx, sy, sz) for 3D or (sx, sy) for 2D.
    """
    a = a.astype(bool)
    b = b.astype(bool)

    if not np.any(a) or not np.any(b):
        return np.array([], dtype=np.float32)

    # surface/boundary voxels
    a_border = a ^ binary_erosion(a)

    # distance to nearest True voxel in b => DT computed on ~b
    dt = distance_transform_edt(~b, sampling=spacing) if spacing is not None else distance_transform_edt(~b)
    return dt[a_border].astype(np.float32)


def hd95_numpy(pred: np.ndarray, gt: np.ndarray, spacing=None) -> float:
    """
    Symmetric HD95 between binary masks pred & gt.
    Returns:
      - 0.0 if both empty
      - inf if one empty, other non-empty
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if (not np.any(pred)) and (not np.any(gt)):
        return 0.0
    if np.any(pred) != np.any(gt):
        return float("inf")

    d1 = _surface_distances(pred, gt, spacing=spacing)
    d2 = _surface_distances(gt, pred, spacing=spacing)
    all_d = np.concatenate([d1, d2])

    # 95th percentile
    return float(np.percentile(all_d, 95))


def calculate_metric_percase(pred: np.ndarray, gt: np.ndarray, spacing=None):
    """
    pred, gt: binary or label masks (will be binarized >0).
    Return (dice, hd95)
    """
    pred = pred.copy()
    gt = gt.copy()
    pred[pred > 0] = 1
    gt[gt > 0] = 1

    if pred.sum() > 0 and gt.sum() > 0:
        dice = dice_numpy(pred, gt)
        hd95 = hd95_numpy(pred, gt, spacing=spacing)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        # giữ behavior giống repo cũ
        return 1.0, 0.0
    else:
        return 0.0, 0.0


# =========================
# Inference for one volume
# =========================
def test_single_volume(
    image,
    label,
    net,
    classes,
    patch_size=[256, 256],
    test_save_path=None,
    case=None,
    z_spacing=1,
):
    """
    image: torch tensor shape [1,1,D,H,W] or [1,1,H,W]
    label: torch tensor same spatial shape
    classes: number of classes
    If test_save_path is not None: save prediction.
      - If SimpleITK available -> save .nii.gz
      - Else -> save .npy/.npz fallback
    """
    image = image.squeeze(0).cpu().detach().numpy().squeeze(0)  # -> (D,H,W) or (H,W)
    label = label.squeeze(0).cpu().detach().numpy().squeeze(0)

    net.eval()

    if len(image.shape) == 3:
        prediction = np.zeros_like(label, dtype=np.int16)

        for ind in range(image.shape[0]):
            slc = image[ind, :, :]
            x, y = slc.shape[0], slc.shape[1]

            if x != patch_size[0] or y != patch_size[1]:
                slc_rs = zoom(slc, (patch_size[0] / x, patch_size[1] / y), order=3)
            else:
                slc_rs = slc

            inp = torch.from_numpy(slc_rs).unsqueeze(0).unsqueeze(0).float().cuda()

            with torch.no_grad():
                outputs = net(inp)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().numpy().astype(np.int16)

            if x != patch_size[0] or y != patch_size[1]:
                pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0).astype(np.int16)
            else:
                pred = out

            prediction[ind] = pred

        spacing = (1, 1, z_spacing)  # keep same style as old sitk spacing set
    else:
        inp = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(inp), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().numpy().astype(np.int16)
        spacing = (1, 1)  # 2D case

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i, spacing=spacing))

    # -------------------------
    # Save output 
    # -------------------------
    if test_save_path is not None:
        os_mkdir(test_save_path)

        # Try SimpleITK if installed; otherwise fallback to npy/npz
        try:
            import SimpleITK as sitk  # optional dependency

            img_itk = sitk.GetImageFromArray(image.astype(np.float32))
            prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
            lab_itk = sitk.GetImageFromArray(label.astype(np.float32))

            # SimpleITK spacing expects (x,y,z) for 3D
            if len(image.shape) == 3:
                img_itk.SetSpacing((1, 1, float(z_spacing)))
                prd_itk.SetSpacing((1, 1, float(z_spacing)))
                lab_itk.SetSpacing((1, 1, float(z_spacing)))
            else:
                img_itk.SetSpacing((1, 1))
                prd_itk.SetSpacing((1, 1))
                lab_itk.SetSpacing((1, 1))

            sitk.WriteImage(prd_itk, f"{test_save_path}/{case}_pred.nii.gz")
            sitk.WriteImage(img_itk, f"{test_save_path}/{case}_img.nii.gz")
            sitk.WriteImage(lab_itk, f"{test_save_path}/{case}_gt.nii.gz")

        except Exception:
            # Fallback: save numpy files
            np.save(f"{test_save_path}/{case}_pred.npy", prediction)
            np.save(f"{test_save_path}/{case}_img.npy", image.astype(np.float32))
            np.save(f"{test_save_path}/{case}_gt.npy", label.astype(np.int16))

    return metric_list


def os_mkdir(path: str):
    import os
    os.makedirs(path, exist_ok=True)
