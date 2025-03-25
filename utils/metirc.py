import torch
from torchmetrics import Metric
import numpy as np
from scipy.spatial.distance import directed_hausdorff
import SimpleITK as sitk
import monai
from monai.metrics import compute_average_surface_distance, SurfaceDistanceMetric

class MultDice(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("intersection", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("union", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        # preds = preds.argmax(dim=1)  # Assuming preds is a multi-class prediction, getting class index with highest probability
        assert prediction.shape == target.shape
        # Calculate intersection and union
        # intersection = torch.sum(preds * target)
        # union = torch.sum(preds) + torch.sum(target)

        batchsize = target.size(0)
        num_classes = target.size(1)
        prediction = prediction.view(batchsize, num_classes, -1)
        target = target.view(batchsize, num_classes, -1)
        intersection = (prediction * target).sum(2).sum(0)[1]
        union = (prediction.sum(2) + target.sum(2)).sum(0)[1]

        # Update metric states
        self.intersection += intersection
        self.union += union

    def compute(self) -> torch.Tensor:
        epsilon = 1e-5
        # Compute final result
        return (2.0 * self.intersection + epsilon) / (self.union + epsilon)

# [B,C,H,W]
def compute_multi_md95(target, y_pred, num_classes=2):
    total_hd = [0] * num_classes
    hd_length = [0] * num_classes

    # 遍历每个样本和每个类别
    for i in range(target.shape[0]):
        for c in range(num_classes):
            # 检查类别 c 的目标和预测掩码是否非空
            if target[i, c].sum() != 0 and y_pred[i, c].sum() != 0:
                # 转换为numpy数组，因为目前没有现成的PyTorch实现可直接用
                target_np = target[i, c].cpu().numpy()
                pred_np = y_pred[i, c].cpu().numpy()

                # 提取非零点坐标
                u = np.array(np.nonzero(target_np)).T
                v = np.array(np.nonzero(pred_np)).T

                # 计算两个方向的 Hausdorff 距离并取最大值
                forward_hausdorff = directed_hausdorff(u, v)[0]
                backward_hausdorff = directed_hausdorff(v, u)[0]
                hausdorff_dist = max(forward_hausdorff, backward_hausdorff)

                total_hd[c] += hausdorff_dist
                hd_length[c] += 1

    # 计算每个类别的平均 Hausdorff 距离
    hd95 = [-1 if length == 0 else total / length for total, length in zip(total_hd, hd_length)]
    return hd95

# gpt4给的代码  [B,H,W]
def compute_hd(target, y_pred):
    total_hd = 0
    hd_length = 0

    # 遍历每个样本
    for i in range(target.shape[0]):
        # 确保预测和目标都不全为空
        if target[i].sum() != 0 and y_pred[i].sum() != 0:
            # 转换为numpy数组，因为目前没有现成的PyTorch实现可直接用
            target_np = target[i].cpu().numpy()
            pred_np = y_pred[i].cpu().numpy()

            # 使用 scipy 库来计算 Hausdorff 距离
            u = np.array(np.nonzero(target_np)).T
            v = np.array(np.nonzero(pred_np)).T

            # 计算两个方向的 Hausdorff 距离并取最大值
            forward_hausdorff = directed_hausdorff(u, v)[0]
            backward_hausdorff = directed_hausdorff(v, u)[0]
            hausdorff_dist = max(forward_hausdorff, backward_hausdorff)

            total_hd += hausdorff_dist
            hd_length += 1

    # 计算平均 Hausdorff 距离
    hd95 = -1 if hd_length == 0 else total_hd / hd_length
    return hd95

# TMI的计算代码  [B,H,W]
def com_hd(target, y_pred):
    total_hd = 0.
    hd_length = 0
    for i in range(target.shape[0]):
        if target[i].sum() != 0 and y_pred[i].sum() != 0:
            gt = sitk.GetImageFromArray(target[i].cpu(), isVector=False)
            my_mask = sitk.GetImageFromArray(y_pred[i].cpu(), isVector=False)
            hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
            hausdorff_distance_filter.Execute(gt, my_mask)
            total_hd += hausdorff_distance_filter.GetHausdorffDistance()
            hd_length += 1

    hd95 = 40 if hd_length == 0 else total_hd / hd_length  # todo 预测结果全为0
    return hd95

# 应用95%,实际验证值会变大  [B,H,W]
def com_hd95(target, y_pred):
    distances = []
    for i in range(target.shape[0]):
        if target[i].sum() != 0 and y_pred[i].sum() != 0:
            gt = sitk.GetImageFromArray(target[i].cpu().numpy(), isVector=False)
            my_mask = sitk.GetImageFromArray(y_pred[i].cpu().numpy(), isVector=False)
            hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
            hausdorff_distance_filter.Execute(gt, my_mask)
            distances.append(hausdorff_distance_filter.GetHausdorffDistance())

    # 计算95% Hausdorff Distance
    hd95 = np.percentile(distances, 95) if distances else -1
    return hd95

class ASSD1(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("sum_assd", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds.detach().cpu().numpy()
        target = target.detach().cpu().numpy()

        batch_size = preds.shape[0]
        for i in range(batch_size):
            pred_surface = self.get_surface_points(preds[i])
            target_surface = self.get_surface_points(target[i])
            assd = self.compute_assd(pred_surface, target_surface)
            self.sum_assd += assd
            self.num_samples += 1

    def compute(self):
        return self.sum_assd / self.num_samples

    def get_surface_points(self, mask):
        surface_points = np.argwhere(mask)
        return surface_points

    def compute_assd(self, pred_surface, target_surface):
        forward_distance = self.compute_average_surface_distance(pred_surface, target_surface)
        backward_distance = self.compute_average_surface_distance(target_surface, pred_surface)
        return (forward_distance + backward_distance) / 2

    def compute_average_surface_distance(self, source_points, target_points):
        distances = []
        for point in source_points:
            distances.append(np.min(np.linalg.norm(target_points - point, axis=1)))
        return np.mean(distances)

class ASSD(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("sum_assd", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, epsilon=20) -> None:
        # Ensure preds and target are detached from the computation graph and moved to CPU
        preds = preds.detach()
        target = target.detach()

        # Compute the average surface distance for the batch
        assd = compute_average_surface_distance(preds, target, symmetric=True)
        assd = torch.where(torch.isnan(assd) | torch.isinf(assd), torch.tensor(epsilon), assd)  # todo 预测结果全为0

        # Update the sum of ASSD and the number of samples
        self.sum_assd += assd.sum()
        self.num_samples += preds.shape[0]

    def compute(self) -> torch.Tensor:
        # Return the average ASSD
        return self.sum_assd / self.num_samples

def com_assd(preds, target):
    return monai.metrics.compute_average_surface_distance(preds, target, symmetric=True)

