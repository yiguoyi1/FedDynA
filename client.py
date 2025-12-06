import copy
from utils import label_to_onehot, cross_entropy_for_onehot as criterion,Quantizer
import matplotlib
import torch
import torch.optim as optim
import numpy as np
import os
import csv
from datetime import datetime
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F
import server
matplotlib.use('Agg')

class Client:
    def __init__(self, args, id, save_dir, model, train_loader, device, num_classes=10):
        self.id = id
        self.args = args
        self.device = device
        self.model = copy.deepcopy(model).to(device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.eta, momentum=0.9)
        self.num_classes = num_classes
        self.train_loader = train_loader
        self.local_epochs = self.args.E
        self.dataset_name = self.args.dataset_name
        # 梯度统计收集
        self.prev_params = [p.clone().detach() for p in self.model.parameters()]
    def calculate_accuracy(self, outputs, targets):
        """计算准确率"""
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == targets).sum().item()
        total = targets.size(0)
        return correct / total
    def copy_gradients(self, model):
        """复制模型梯度"""
        gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.detach().clone()
        return gradients
    def deep_copy_state_dict(self, model):
        state_dict = model.state_dict()
        return copy.deepcopy(state_dict)
    def set_model_params(self, params):
        """设置模型参数"""
        self.model.load_state_dict(params)

    def set_subspaces(self, subspaces):
        self.subspaces = {name: U.to(self.device) for name, U in subspaces.items()}
    def train(self, round_num, do_local_update: bool = True):
        """
        do_local_update: True -> 在本地执行 optimizer.step()（用于 warm-up / FedAvg）
                         False -> 不执行本地 step，仅计算并上传平均梯度的投影系数（用于投影模式）
        返回: (coeffs_dict, num_samples, avg_loss, avg_acc, local_state)
        local_state 在投影模式可以忽略（main 仅在 warm-up 使用）
        """
        self.model.train()
        total_loss, total_acc = 0.0, 0.0

        # ===== 累积平均梯度 =====
        sum_grads = {}
        total_seen = 0

        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.98)
        for epoch in range(self.local_epochs):
            steps_made = 0
            for batch_x, batch_y in self.train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                bs = batch_x.size(0)

                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, label_to_onehot(batch_y, num_classes=self.num_classes))
                loss.backward()

                # clip gradients to stabilize
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

                # 先复制 grads（用于投影/上报）
                for name, param in self.model.named_parameters():
                    if param.grad is None:
                        continue
                    g = param.grad.detach().clone().flatten()
                    if name not in sum_grads:
                        sum_grads[name] = g * float(bs)
                    else:
                        sum_grads[name] += g * float(bs)

                if do_local_update:
                    # 本地参数更新（warm-up / FedAvg 模式）
                    self.optimizer.step()
                    steps_made += 1
                else:
                    # 投影模式：不执行 local update（全局由 server 更新）
                    pass

                total_seen += bs
                total_loss += loss.item() * bs
                total_acc += self.calculate_accuracy(outputs, batch_y) * bs

            # 只在做本地更新时调用 scheduler.step()
            if do_local_update and steps_made > 0:
                scheduler.step()

        # ===== 计算平均梯度并投影（如果收到子空间）=====
        coeffs_dict = {}
        if total_seen > 0 and hasattr(self, "subspaces"):
            for name, g_sum in sum_grads.items():
                if name not in self.subspaces:
                    continue
                g_avg = (g_sum / float(total_seen)).view(-1)
                U = self.subspaces[name]
                if U is None or U.dim() != 2:
                    continue
                if g_avg.numel() != U.shape[0]:
                    print(f"[Client {self.id}] mismatch for {name}: grad {g_avg.numel()} vs U {U.shape[0]}")
                    continue

                g_avg = g_avg.to(U.device, dtype=U.dtype)
                c = torch.matmul(U.t(), g_avg)
                coeffs_dict[name] = c.detach().cpu().clone()
        quant_bits = getattr(self.args, "quant_bits", 8)
        use_verbose = getattr(self.args, "verbose", False)

        # Case A: quant_bits >= 32 -> bypass quantization, upload float32
        if int(quant_bits) >= 32:
            for name, c in list(coeffs_dict.items()):
                try:
                    orig_energy = float(c.pow(2).sum().cpu().item())
                except Exception:
                    orig_energy = float(torch.norm(c).cpu().item())**2

                # send (float32 tensor, orig_energy)
                coeffs_dict[name] = (c.detach().cpu().float(), orig_energy)

                if use_verbose:
                    try:
                        print(f"[diag_client] bypass-quant layer={name} "
                            f"c_norm={float(torch.norm(c).cpu().item()):.6e}")
                    except Exception:
                        pass

        # Case B: quant_bits < 32 -> per-dim PDQ quantize
        else:
            quantizer = Quantizer(
                num_bits=int(quant_bits),
                symmetric=getattr(self.args, "quant_symmetric", True)
            )

            for name, c in list(coeffs_dict.items()):
                try:
                    orig_energy = float(c.pow(2).sum().cpu().item())
                except Exception:
                    orig_energy = float(torch.norm(c).cpu().item())**2

                q_tuple = quantizer.quantize(c)
                coeffs_dict[name] = (q_tuple, orig_energy)

                if use_verbose:
                    try:
                        deq = quantizer.dequantize(q_tuple, device="cpu", dtype=torch.float32)
                        print(f"[diag_client] layer={name} "
                            f"c_norm={float(torch.norm(c).cpu().item()):.6e} "
                            f"deq_norm={float(torch.norm(deq).cpu().item()):.6e} "
                            f"scale_len={q_tuple[1].numel() if torch.is_tensor(q_tuple[1]) else 'scalar'}")
                    except Exception as e:
                        print(f"[diag_client] quant debug failed for {name}: {e}")
        avg_loss = total_loss / max(total_seen, 1)
        avg_acc = total_acc / max(total_seen, 1)
        num_samples = len(self.train_loader.dataset)
        with torch.no_grad():
            diff_norm = 0.0
            for p1, p2 in zip(self.model.parameters(), self.prev_params):
                diff_norm += (p1 - p2.to(self.device)).norm().item()
            """
            print(f"[Client {self.id}] local update L2 diff = {diff_norm:.4f}")
            """
            # 更新缓存
            self.prev_params = [p.clone().detach() for p in self.model.parameters()]
        # 返回本地 state（warm-up 时使用；投影模式 main 可忽略）
        local_state = self.deep_copy_state_dict(self.model)
        return coeffs_dict, num_samples, avg_loss, avg_acc, local_state