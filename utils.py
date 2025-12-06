import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
import matplotlib.pyplot as plt
import os
import math

def label_to_onehot(target, num_classes=10):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))

def plot_accuracy_curve(save_dir, accuracy_list, loss_list):
    save_path = os.path.join(save_dir, 'accuracy_loss_curve.png')
    plt.figure(figsize=(10, 6))

    rounds = range(1, len(accuracy_list) + 1)

    # 主轴画 Accuracy
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(rounds, accuracy_list, 'b-o', label='Accuracy')
    ax1.set_xlabel('Communication Round')
    ax1.set_ylabel('Test Accuracy', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # 副轴画 Loss
    ax2 = ax1.twinx()
    ax2.plot(rounds, loss_list, 'r-s', label='Loss')
    ax2.set_ylabel('Test Loss', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # 添加标题和网格
    plt.title('Global Model Test Accuracy and Loss Across Rounds')
    ax1.grid(True)

    # 图例（分别添加）
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')

    # 保存图像
    plt.savefig(save_path)
    plt.close()
    print(f"Saved accuracy and loss curve to {save_path}")
def dirichlet_partition(dataset, num_clients, alpha=0.5, seed=42):
    """
    Dirichlet非IID划分：按类别分布差异划分数据（alpha越小，Non-IID程度越强）
    返回：每个客户端的数据集（WrapperDataset格式）
    保证每个客户端至少有一个样本
    """
    np.random.seed(seed)

    num_samples = len(dataset)
    labels = []
    for i in range(num_samples):
        _, label = dataset[i]
        labels.append(int(label) if isinstance(label, torch.Tensor) else label)
    labels = np.array(labels)

    num_classes = len(np.unique(labels))
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_indices = np.where(labels == c)[0]
        if len(class_indices) == 0:
            continue
        np.random.shuffle(class_indices)

        # 只有当该类别样本数大于等于客户端数时，才强制每个客户端分到至少一个
        if len(class_indices) >= num_clients:
            while True:
                proportions = np.random.dirichlet(alpha=alpha * np.ones(num_clients))
                proportions = proportions / proportions.sum()
                client_counts = (proportions * len(class_indices)).astype(int)
                # 每个客户端至少一个
                for i in range(num_clients):
                    if client_counts[i] == 0:
                        client_counts[i] = 1
                # 修正总数
                diff = len(class_indices) - np.sum(client_counts)
                for i in range(abs(diff)):
                    client_counts[i % num_clients] += np.sign(diff)
                if np.all(client_counts > 0) and np.sum(client_counts) == len(class_indices):
                    break
        else:
            # 样本数小于客户端数，直接分配
            client_counts = np.zeros(num_clients, dtype=int)
            client_counts[:len(class_indices)] = 1

        start = 0
        for client_id in range(num_clients):
            end = start + client_counts[client_id]
            client_indices[client_id].extend(class_indices[start:end])
            start = end

    # 检查是否有客户端没有样本
    for i, indices in enumerate(client_indices):
        if len(indices) == 0:
            # 从样本最多的客户端拿一个
            max_client = np.argmax([len(idx) for idx in client_indices])
            client_indices[i].append(client_indices[max_client].pop())

    client_datasets = []
    for indices in client_indices:
        subset = Subset(dataset, indices)
        client_datasets.append(WrapperDataset(subset))

    return client_datasets
class WrapperDataset(Dataset):
    """ 确保数据格式始终为 (image, label) """
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label

    def __len__(self):
        return len(self.dataset)
    
class Quantizer:
    """
    Per-dimension symmetric quantizer for 1D coefficient vectors.
    quantize(x) -> (q_tensor_cpu (int32), scales_cpu (float32 tensor length r), 0.0, "pdq")
    dequantize(quant_tuple, device="cpu", dtype=torch.float32) -> float tensor
    """
    def __init__(self, num_bits: int = 8, symmetric: bool = True):
        # 允许任意 >=2 的位宽；把 >=16 当作 FP16 快捷路径处理
        if int(num_bits) < 2:
            raise ValueError("num_bits must be >= 2")
        self.num_bits = int(num_bits)
        self.symmetric = bool(symmetric)
        self.qmax = 2 ** (self.num_bits - 1) - 1

    def quantize(self, x: torch.Tensor):
        if x is None:
            raise ValueError("Quantize got None")
        x = x.detach().cpu().to(torch.float32)
        if x.dim() != 1:
            raise ValueError("Quantizer.quantize expects 1D tensor of coefficients")

        r = x.numel()

        # FP16 shortcut
        if self.num_bits >= 16:
            return (x.half().cpu(), torch.ones(r, dtype=torch.float32).cpu(), 0.0, "fp16")

        eps = 1e-12
        abs_max = x.abs().clamp_min(eps)

        scale = (abs_max / float(self.qmax)).to(torch.float32)  # shape [r]

        q = torch.round(x / scale).clamp(min=-self.qmax, max=self.qmax).to(torch.int32)
        return (q.cpu(), scale.cpu(), 0.0, "pdq")

    def dequantize(self, quant_tuple, device="cpu", dtype=torch.float32):
        if not isinstance(quant_tuple, tuple):
            raise ValueError("quant_tuple must be a tuple")

        # PDQ format
        if len(quant_tuple) == 4 and (quant_tuple[3] == "pdq" or quant_tuple[3] == "fp16"):
            q_tensor, scales = quant_tuple[0], quant_tuple[1]

            # fp16 sentinel
            if quant_tuple[3] == "fp16":
                return q_tensor.to(device=device, dtype=dtype)

            qf = q_tensor.to(torch.float32).to(device=device)
            scales_t = scales.to(torch.float32).to(device=device)
            if qf.numel() != scales_t.numel():
                raise ValueError("Quantizer.dequantize: q and scales length mismatch")
            return (qf * scales_t).to(dtype)

        # legacy format
        if len(quant_tuple) >= 2 and torch.is_tensor(quant_tuple[0]) and isinstance(quant_tuple[1], (float, int)):
            q_tensor = quant_tuple[0]
            scale_val = float(quant_tuple[1])
            return q_tensor.to(torch.float32).to(device) * scale_val

        raise ValueError("Unsupported quant_tuple format")