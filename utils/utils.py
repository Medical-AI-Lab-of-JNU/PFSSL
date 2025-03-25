import os
import numpy as np
from skimage import measure
import csv
import cv2
import torch


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def dice_score(prediction, target):
    smooth = 1e-5
    num_classes = target.size(0)
    prediction = prediction.view(num_classes, -1)
    target = target.view(num_classes, -1)

    intersection = (prediction * target)

    dice = (2. * intersection.sum(1) + smooth) / (prediction.sum(1) + target.sum(1) + smooth)

    return dice



def dice_score_batch(prediction, target):
    smooth = 1e-5
    batchsize = target.size(0)
    num_classes = target.size(1)
    prediction = prediction.view(batchsize, num_classes, -1)
    target = target.view(batchsize, num_classes, -1)
    intersection = (prediction * target)
    dice = (2. * intersection.sum(2) + smooth) / (prediction.sum(2) + target.sum(2) + smooth)
    return dice


def measure_img(o_img, t_num=1):
    p_img=np.zeros_like(o_img)
    testa1 = measure.label(o_img.astype("bool"))
    props = measure.regionprops(testa1)
    numPix = []
    for ia in range(len(props)):
        numPix += [props[ia].area]
    for i in range(0, t_num):
        index = numPix.index(max(numPix)) + 1
        p_img[testa1 == index]=o_img[testa1 == index]
        numPix[index-1]=0
    return p_img


def write_csv(file_path, epoch, data):
    # 获取文件目录
    directory = os.path.dirname(file_path)

    # 如果目录不存在，则创建
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 创建或打开csv文件
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # 写入表头（第一轮时创建）
        # if epoch == 0:
        #     writer.writerow(['Epoch', 'Train ACC', 'Train AUC', 'Train Loss'])
        # 写入结果
        data_list = [epoch]
        data_list.extend(data)
        writer.writerow(data_list)

def log_metrics(writer, metrics, epoch, mode):
    """
    使用TensorBoard的SummaryWriter记录训练指标。

    参数:
    writer -- TensorBoard的SummaryWriter对象
    epoch -- 当前的训练周期
    metrics -- 一个包含指标名称和它们的值的字典
    """
    train_metric_list = ["train_ce", "train_miou", "train_dice", "res_miou", "res_dice", "cmt_loss", "cps_loss", "dice0", "dice1"]
    test_metric_list = ["test_ce", "test_miou", "test_dice", "res_miou", "res_dice", "cmt_loss", "cps_loss", "dice0", "dice1"]
    for i, metric in enumerate(metrics):
        if mode == "train":
            writer.add_scalar(train_metric_list[i], metric, epoch)
        else:
            writer.add_scalar(test_metric_list[i], metric, epoch)

def save_image(batch_size, binary_predictions, filename, predict_path, type="pseudo"):
    for i in range(batch_size):
        # 取出当前 batch 中的图片
        current_image = binary_predictions[i, 0].cpu().numpy()  # 假设只有一个通道（单通道图像）

        # 设置保存路径和文件名
        img_name = filename[i]
        patient_name = img_name.rsplit('_', 1)[0]
        pseudo_path = os.path.join(predict_path, patient_name)
        os.makedirs(pseudo_path, exist_ok=True)
        img_path = f'{pseudo_path}/{img_name}_{type}.png'
        cv2.imwrite(img_path, current_image * 255)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 0.1 * sigmoid_rampup(epoch, 200.0)

def detect_edges_and_preserve_mask(tensor_batch):
    # tensor_batch: [B, H, W], assuming binary masks with values 0 and 1
    batch_size = tensor_batch.size(0)
    edge_maps = []

    for i in range(batch_size):
        img_tensor = tensor_batch[i]  # [H, W], single channel mask
        img_np = img_tensor.detach().cpu().numpy().astype(np.float32)  # Convert to numpy array and ensure type is float32

        # Sobel edge detection
        edges = cv2.Sobel(img_np, cv2.CV_64F, 1, 1, ksize=5)
        edges = np.abs(edges)  # Get absolute value of edges
        edges[edges > 0] = 1  # Binarize detected edges

        # Preserve original mask values by combining edges with original mask
        combined_mask = np.maximum(edges, img_np)  # Use np.maximum to retain mask center values
        edge_tensor = torch.from_numpy(combined_mask).to(tensor_batch.device)
        edge_maps.append(edge_tensor.unsqueeze(0))  # Add back the batch dimension [1, H, W]

    # Concatenate all edge maps to form a batch of edge maps
    return torch.cat(edge_maps, dim=0)  # [B, H, W]

def detect_edges(tensor_batch):
    # tensor_batch: [B, H, W], assuming binary masks with values 0 and 1
    batch_size = tensor_batch.size(0)
    edge_maps = []

    for i in range(batch_size):
        img_tensor = tensor_batch[i]  # [H, W], single channel mask
        img_np = img_tensor.detach().cpu().numpy().astype(np.float32)  # Convert to numpy array and ensure type is float32

        # Sobel edge detection
        edges = cv2.Sobel(img_np, cv2.CV_64F, 1, 1, ksize=5)
        edge_tensor = torch.from_numpy(np.abs(edges)).to(tensor_batch.device)
        edge_maps.append(edge_tensor.unsqueeze(0))  # Add back the batch dimension [1, H, W]

    # Concatenate all edge maps to form a batch of edge maps
    return torch.cat(edge_maps, dim=0)  # [B, H, W]

def enhance_confidence_on_edges(tensor_batch, high_confidence=10):
    """
    Enhance the confidence values along the edges of binary masks.
    tensor_batch: [B, H, W] - Binary masks
    high_confidence: float - The confidence value to set on edges
    """
    batch_size = tensor_batch.size(0)
    enhanced_masks = []

    for i in range(batch_size):
        img_tensor = tensor_batch[i]  # [H, W], single channel mask
        img_np = img_tensor.detach().cpu().numpy().astype(np.float32)  # Ensure type is float32

        # Detect edges using Sobel operator
        edges = cv2.Sobel(img_np, cv2.CV_64F, 1, 1, ksize=5)
        edges = np.abs(edges) > 0  # Binary edge map

        # Create an enhanced mask
        enhanced_mask = np.where(edges, high_confidence, img_np)
        enhanced_tensor = torch.from_numpy(enhanced_mask).to(tensor_batch.device)
        enhanced_masks.append(enhanced_tensor.unsqueeze(0))  # Add back the batch dimension [1, H, W]

    return torch.cat(enhanced_masks, dim=0)  # [B, H, W]

def enhance_confidence_on_edges_with_transition(tensor_batch, transition_width=5, high_confidence=10):
    """
    Enhance the confidence values along the edges of binary masks, adding a smooth transition.
    tensor_batch: [B, H, W] - Binary masks
    transition_width: int - Width of the transition zone around edges
    high_confidence: float - The maximum confidence value to set on edges
    """
    batch_size = tensor_batch.size(0)
    enhanced_masks = []

    for i in range(batch_size):
        img_tensor = tensor_batch[i]  # [H, W], single channel mask
        img_np = img_tensor.detach().cpu().numpy().astype(np.float32)  # Convert to numpy array

        # Sobel edge detection to get the gradient magnitude
        grad_x = cv2.Sobel(img_np, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(img_np, cv2.CV_64F, 0, 1, ksize=5)
        edges = np.sqrt(grad_x**2 + grad_y**2)

        # Normalize and scale edge values to create a transition effect
        edges_normalized = np.clip(edges / np.max(edges), 0, 1)
        transition = edges_normalized * high_confidence

        # Ensure that the interior mask values are maintained
        enhanced_mask = np.where(img_np > 0, np.maximum(transition, img_np * high_confidence), transition)
        enhanced_mask = np.clip(enhanced_mask, 0, high_confidence)  # Ensure values are still within the confidence limits

        enhanced_tensor = torch.from_numpy(enhanced_mask).to(tensor_batch.device)
        enhanced_masks.append(enhanced_tensor.unsqueeze(0))  # Add back the batch dimension [1, H, W]

    return torch.cat(enhanced_masks, dim=0)  # [B, H, W]


def detect_and_enhance_edges(tensor_batch, high_confidence=10):
    """
    Detect edges in binary masks and enhance the confidence values along the edges.
    Args:
    - tensor_batch (torch.Tensor): [B, H, W], binary masks with values 0 and 1.
    - high_confidence (float): The confidence value to set on edges.

    Returns:
    - torch.Tensor: [B, H, W], masks with enhanced edge confidence.
    """
    batch_size = tensor_batch.size(0)
    enhanced_masks = []

    for i in range(batch_size):
        img_tensor = tensor_batch[i]  # [H, W], single channel mask
        img_np = img_tensor.detach().cpu().numpy().astype(np.float32)  # Ensure type is float32

        # Sobel edge detection
        edges = cv2.Sobel(img_np, cv2.CV_64F, 1, 1, ksize=5)
        edges = np.abs(edges)

        # Smooth the edges for a transition effect
        kernel_size = 5
        smoothed_edges = cv2.GaussianBlur(edges, (kernel_size, kernel_size), 0)

        # Create an enhanced mask
        enhanced_mask = np.where(smoothed_edges > 0, high_confidence, img_np)
        enhanced_tensor = torch.from_numpy(enhanced_mask).to(tensor_batch.device)
        enhanced_masks.append(enhanced_tensor.unsqueeze(0))  # Add back the batch dimension [1, H, W]

    return torch.cat(enhanced_masks, dim=0)  # [B, H, W]

def sort_confidence(x1_mean):
    # 假设 x1_mean 是你的输入张量，B x L 的尺寸
    B, L = x1_mean.shape

    # 设定 L 为每个 batch 结果的固定长度，这里假设取 3 个最小的索引
    L = 64
    # 初始化一个张量来存储每个batch的结果
    min_indices_tensor = torch.zeros((B, L), dtype=torch.long)

    for i in range(B):
        # 获取当前batch的数据
        batch_data = x1_mean[i]

        # 获取大于0的元素及其索引
        positive_indices = torch.where(batch_data > 0)[0]
        positive_values = batch_data[positive_indices]

        if positive_values.numel() == 0:
            continue  # 如果没有大于0的元素，跳过

        # 对这些值进行排序
        sorted_indices = torch.argsort(positive_values)

        # 获取排序后的索引对应的原始索引
        min_indices = positive_indices[sorted_indices]

        # 可以选择取最小的几个索引，例如最小的3个
        min_indices = min_indices[:L]  # 取最小的L个索引

        # 填充结果到张量中
        min_indices_tensor[i, :min_indices.size(0)] = min_indices
    return min_indices_tensor

def sort_mean(mean_val):
    # 获取batch size和length
    B, L = mean_val.shape

    # 创建一个用于存储排序后的索引的tensor
    sorted_indices = torch.zeros_like(mean_val, dtype=torch.long, device=mean_val.device)

    # 对每个batch进行处理
    for i in range(B):
        tensor_1d = mean_val[i]

        # 获取非0值的索引和0值的索引
        non_zero_indices = (tensor_1d != 0).nonzero(as_tuple=True)[0]
        zero_indices = (tensor_1d == 0).nonzero(as_tuple=True)[0]

        # 获取非0值及其索引，并按值排序
        non_zero_values = tensor_1d[non_zero_indices]
        sorted_non_zero_indices = non_zero_indices[non_zero_values.argsort()]

        # 将非0值的索引和0值的索引拼接
        sorted_indices[i, :len(sorted_non_zero_indices)] = sorted_non_zero_indices
        sorted_indices[i, len(sorted_non_zero_indices):] = zero_indices
    return sorted_indices

def cal_mask_ratio(label):
    # 计算1的像素数
    ones_count = torch.sum(label == 1).item()
    # 计算1的占比
    ones_ratio = ones_count / (label.numel())
    return ones_ratio

def calculate_patch_num(x):
    if x <= 0.02:
        return 1
    elif 0.02 < x <= 0.06:
        return 2
    elif 0.06 < x <= 0.1:
        return 3
    elif x > 0.1:
        return 4
    else:
        return None  # 如果x不在任何范围内，返回None

def normalize(tensor):
    min_val = tensor.min(1, keepdim=True)[0]
    max_val = tensor.max(1, keepdim=True)[0]
    result = tensor - min_val
    result = result / max_val
    return result

def get_edge_by_sobel(mask_tensor, ksize=3):
    batch = mask_tensor.shape[0]
    sobel_edges = []
    # 遍历每个batch元素
    for i in range(batch):
        # 将每个batch元素从GPU复制到CPU，并转换为NumPy数组
        mask_numpy = mask_tensor[i].cpu().numpy()

        # 确保NumPy数组的类型是uint8并将值扩展到0和255
        mask_numpy = (mask_numpy * 255).astype(np.uint8)

        # 使用Sobel算子检测边缘
        # ksize = 3  # 可以调整此值，必须是奇数
        sobelx = cv2.Sobel(mask_numpy, cv2.CV_64F, 1, 0, ksize=ksize)  # 水平边缘
        sobely = cv2.Sobel(mask_numpy, cv2.CV_64F, 0, 1, ksize=ksize)  # 垂直边缘

        # 计算边缘的幅值
        sobel_edge = np.sqrt(sobelx ** 2 + sobely ** 2)

        # 将边缘图像转换为8位无符号整数类型，并进行归一化以便于显示
        # sobel_edge_normalized = cv2.normalize(sobel_edge, None, 0, 255, cv2.NORM_MINMAX)
        # sobel_edge_normalized = np.uint8(sobel_edge_normalized)
        # sobel_edge = np.uint8(np.clip(sobel_edge, 0, 255))
        sobel_edge = np.uint8(np.clip(sobel_edge, 0, 1))

        # 将处理后的结果添加到列表中
        sobel_edges.append(sobel_edge)

    # 将列表中的结果转换回Tensor
    sobel_edges_tensor = torch.tensor(np.array(sobel_edges), dtype=torch.int64).cuda()
    return sobel_edges_tensor
