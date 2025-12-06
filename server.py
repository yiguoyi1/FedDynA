import csv
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import os
import math
from utils import cross_entropy_for_onehot as criterion, label_to_onehot,Quantizer
from typing import List, Dict, Optional, Union

class FedDynA_SubspaceUpdater:
    """
    子空间更新器（优化显存）：
    - U / S 常驻 CPU（减少 GPU 常驻显存）
    - update() 在 CPU 上完成（eigh/qr 在 CPU）
    - reconstruct() 在需要时把 U 临时拷贝到 coeff 的 device 并在 no_grad 下计算
    """
    def __init__(self, d: int, r: int, alpha: float = 0.9, device="cpu"):
        self.d = d
        self.r = r
        self.alpha = alpha            # 仅用于 S 的 EMA
        self.device = device          # 训练主设备（如 "cuda:0"）
      

        # 在 CPU 上保存 U 与 S
        rand_mat = torch.randn((d, r), device="cpu")
        Q, _ = torch.linalg.qr(rand_mat)
        self.U = Q[:, :r].contiguous()       # CPU tensor
        self.S = torch.zeros((r, r), device="cpu")
        self.S_epsilon = 1e-6
        
    @torch.no_grad()
    def reconstruct(self, coeff: torch.Tensor, chunk_size: int = 4096):
    
        coeff = coeff.to(torch.float32, non_blocking=True)
        device = coeff.device
        d, r = self.U.shape

        
        eps = 1e-12
        try:
            tr_S = float(torch.trace(self.S).cpu().item())
            G = (self.U.t() @ self.U)              # CPU
            tr_USU = float(torch.trace(self.S @ G).cpu().item())
        except Exception:
            tr_S = 0.0
            tr_USU = 0.0

        if tr_USU < eps or tr_S <= 0.0:
            k = 1.0
        else:
            k = math.sqrt(max(tr_S, eps) / (tr_USU + eps))

        out = torch.empty((d,), dtype=torch.float32, device=device)
        for start in range(0, d, chunk_size):
            end = min(start + chunk_size, d)

            subU_cpu = self.U[start:end]  
            if device.type == "cuda":
                subU = subU_cpu.to(device, non_blocking=True)
            else:
                subU = subU_cpu

            if subU.dtype != coeff.dtype:
                subU = subU.to(coeff.dtype)

            out[start:end] = subU @ coeff
            del subU

        # ---- 3. 应用理论比例 k ----
        out *= float(k)
          # --- 诊断打印（按需开启 self.verbose） ---
        if getattr(self, "verbose", True):
            try:
                recon_norm = float(torch.norm(out).cpu().item())
                print(f"[reconstruct][diag] d={d} r={r} k={k:.4e} tr_S={tr_S:.4e} tr_USU={tr_USU:.4e} recon_norm={recon_norm:.4e}")
            except Exception:
                pass
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        return out


    @torch.no_grad()
    def update(self, coeff_list: List[torch.Tensor], weights=None, scales: List[Optional[Union[float, torch.Tensor, list, np.ndarray]]] = None):
        """
        coeff_list: list of 1D tensors (r,) (will be moved to CPU float64)
        weights: optional list of floats (same length as coeff_list)
        scales: optional list aligned with coeff_list where each entry is:
                - None
                - scalar (float/int)
                - 1D torch.Tensor of length r (per-dim scales)
                - list/np.ndarray of length r
        """
        if len(coeff_list) == 0:
            return

        # collect to CPU float64
        C = torch.stack([c.detach().to("cpu").to(torch.float64) for c in coeff_list], dim=0)  # [n, r]
        n, r = C.shape

        # handle weights
        if weights is None:
            w = torch.ones(n, dtype=C.dtype, device="cpu") / float(n)
        else:
            w = torch.tensor(weights, dtype=C.dtype, device="cpu")
            s = float(w.sum())
            if s <= 0:
                w = torch.ones_like(w) / float(w.numel())
            else:
                w = w / s

        # compute mean and centered
        c_mean = (w.unsqueeze(1) * C).sum(dim=0, keepdim=True)   # [1, r]
        Cc = C - c_mean  # [n, r]

        # covariance (float64)
        cov = (Cc.t() * w) @ Cc  # [r, r]

        # estimate & remove quantization noise variance (diagonal approx)
         
        mean_noise_var = 0.0
        if scales is not None and len(scales) == n:
            # compute per-client noise variance (scalar) robustly
            noise_vars = []
            for s_idx, s_val in enumerate(scales):
                try:
                    if s_val is None:
                        noise_vars.append(0.0)
                    elif isinstance(s_val, torch.Tensor):
                        sv = s_val.detach().cpu().to(torch.float64)
                        # if per-dim scales provided, approximate noise variance as mean(scale^2/12)
                        noise_vars.append(float((sv.pow(2).mean().item()) / 12.0))
                    elif isinstance(s_val, (list, tuple, np.ndarray)):
                        arr = np.asarray(s_val, dtype=float)
                        noise_vars.append(float((np.mean(arr ** 2)) / 12.0))
                    elif isinstance(s_val, (float, int)):
                        noise_vars.append(float((float(s_val) ** 2) / 12.0))
                    else:
                        noise_vars.append(0.0)
                except Exception:
                    noise_vars.append(0.0)

            noise_vars_t = torch.tensor(noise_vars, dtype=cov.dtype, device=cov.device)  # [n]
            # weighted mean over clients (respecting w)
            try:
                denom = float((w * (noise_vars_t != 0.0).to(w.dtype)).sum().cpu().item())
                if denom <= 0:
                    # fallback to standard weighted mean including zeros
                    mean_noise_var = float((w * noise_vars_t).sum().cpu().item())
                else:
                    # normalize over contributing clients
                    mean_noise_var = float((w * noise_vars_t).sum().cpu().item() / max(denom, 1e-12))
            except Exception:
                mean_noise_var = float((w * noise_vars_t).sum().cpu().item())
            # ensure non-negative
            mean_noise_var = max(mean_noise_var, 0.0)

            if mean_noise_var > 0.0:
                cov = cov - (mean_noise_var * torch.eye(r, dtype=cov.dtype, device=cov.device))

        # symmetrize and numerical stability
        cov = 0.5 * (cov + cov.t())

        # eig decomp and clamp to ensure PSD
        try:
            eigvals, eigvecs = torch.linalg.eigh(cov)
            eigvals_clamped = torch.clamp(eigvals, min=1e-12)
            cov_psd = (eigvecs * eigvals_clamped.unsqueeze(0)) @ eigvecs.t()
        except Exception:
            cov_psd = cov + 1e-12 * torch.eye(r, dtype=cov.dtype, device=cov.device)

        # EMA update self.S in float64
        if not hasattr(self, "S") or self.S is None:
            self.S = cov_psd.to(torch.float64)
        else:
            self.S = (float(self.alpha) * self.S) + ((1.0 - float(self.alpha)) * cov_psd)

        # small regularization
        self.S = self.S + (self.S_epsilon * torch.eye(self.S.size(0), dtype=self.S.dtype, device="cpu"))

        # update U using top eigenvectors of self.S
        try:
            eigvals_S, eigvecs_S = torch.linalg.eigh(self.S)
            top_k = min(self.r, eigvals_S.numel())
            _, top_idx = torch.topk(eigvals_S, k=top_k, largest=True)
            V_new = eigvecs_S[:, top_idx]
            U_new = (self.U @ V_new).contiguous()
            self.U, _ = torch.linalg.qr(U_new)
        except Exception:
            pass

        # cleanup
        del C, c_mean, Cc, cov, cov_psd
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        return

    def project(self, grad):
        """
        将梯度向量投影到当前子空间中。
        grad: Tensor, shape [d]
        返回: 投影系数 c, shape [r]
        """
        return self.U.T @ grad


class FedDynAServer:
   

    def __init__(self, model, layer_dims: Dict[str, int], rank: int = 16, alpha: float = 0.9,
             lr: float = 1.0, device="cpu", k_update=5, save_path="result",
             warmup_rounds: int = 10, warmup_lr_scale: float = 0.1, warmup_extra: int = 10, verbose: bool = True, server_momentum: float = 0, server_lr_scale: float = 1.0):
        self.global_model = model
        self.layer_dims = layer_dims
        self.rank = rank
        self.alpha = alpha
        self.lr = lr  # base server lr
        self.device = device
        self.k_update = k_update
        self.warmup_rounds = warmup_rounds
        self.warmup_lr_scale = warmup_lr_scale
        self.warmup_extra = warmup_extra
        self.verbose = verbose
        self.server_momentum = float(server_momentum)
        self.server_lr_scale = float(server_lr_scale)
        os.makedirs(save_path, exist_ok=True)
        self.server_velocity = {}   # lazy init
        self.prev_global = None
        self.subspace_lr_scale = 1
        # 初始化子空间更新器
        self.subspaces = {
            name: FedDynA_SubspaceUpdater(d=dim, r=rank, alpha=alpha, device=device)
            for name, dim in layer_dims.items()
        }

    @torch.no_grad()
    def aggregate(self, client_coeffs, client_sizes, round_num):
        total = sum(client_sizes)
        if total == 0:
            return

        # 学习率：预热阶段小步更新
        if round_num < self.warmup_rounds + self.warmup_extra:
            effective_lr = 1.0
        else:
            effective_lr = self.lr

        # 保持全局模型参数在 CPU，按层临时搬到 GPU 进行更新
        global_state = {k: v.detach().cpu() for k, v in self.global_model.state_dict().items()}
        new_state = dict(global_state)

        for name, updater in self.subspaces.items():
            coeffs_cpu, weights = [], []

            # 初始化量化器
            quantizer = Quantizer(
                num_bits=getattr(self, "quant_bits", 8),
                symmetric=getattr(self, "quant_symmetric", True)
            )

            # 收集客户端的系数（保持在 CPU）
            coeffs_cpu = []
            weights = []
            scales_list = []  
            orig_energies = []

            for c_dict, n in zip(client_coeffs, client_sizes):
                if name not in c_dict:
                    continue

                entry = c_dict[name]
                coeff_entry = None
                orig_energy = None
                client_scales = None 
            

                if isinstance(entry, tuple) and len(entry) == 2 \
                    and isinstance(entry[0], tuple) \
                    and torch.is_tensor(entry[0][0]):

                    quant_tuple, orig_energy = entry

                    # 1) decode
                    try:
                        coeff_entry = quantizer.dequantize(quant_tuple, device="cpu", dtype=torch.float32)
                    except Exception as e:
                        if self.verbose:
                            print(f"[aggregate][decode-fail] layer={name}: {e}")
                        continue

                    # 2) parse scale
                    raw_scale = quant_tuple[1]
                    if isinstance(raw_scale, torch.Tensor):
                        client_scales = raw_scale.detach().cpu().float()
                    elif isinstance(raw_scale, (list, tuple, np.ndarray)):
                        client_scales = torch.as_tensor(raw_scale, dtype=torch.float32)
                    else:
                        client_scales = None

                    scales_list.append(client_scales)
                    orig_energy = float(orig_energy)

                # ------------------------------
                # Case ②  direct PDQ tuple (rare)
                # ------------------------------
                elif isinstance(entry, tuple) and len(entry) == 4 \
                    and torch.is_tensor(entry[0]) and isinstance(entry[3], str):

                    try:
                        coeff_entry = quantizer.dequantize(entry, device="cpu", dtype=torch.float32)
                    except Exception as e:
                        if self.verbose:
                            print(f"[aggregate][decode-fail] layer={name}: {e}")
                        continue

                    raw_scale = entry[1]
                    if isinstance(raw_scale, torch.Tensor):
                        client_scales = raw_scale.detach().cpu().float()
                    elif isinstance(raw_scale, (list, tuple, np.ndarray)):
                        client_scales = torch.as_tensor(raw_scale, dtype=torch.float32)
                    else:
                        client_scales = None

                    scales_list.append(client_scales)

                # ------------------------------
                # Case ③  (tensor, orig_energy) no quant
                # ------------------------------
                elif isinstance(entry, tuple) and len(entry) == 2 \
                    and torch.is_tensor(entry[0]):

                    coeff_entry = entry[0]
                    orig_energy = float(entry[1])
                    scales_list.append(None)

                # ------------------------------
                # Case ④  plain tensor
                # ------------------------------
                elif torch.is_tensor(entry):
                    coeff_entry = entry
                    scales_list.append(None)

                # ------------------------------
                # invalid
                # ------------------------------
                else:
                    if self.verbose:
                        print(f"[aggregate] invalid entry for layer={name}, type={type(entry)}")
                    continue

                # materialize 到 CPU 并做基本校验
                try:
                    c_cpu = coeff_entry.detach().to("cpu", dtype=torch.float32)
                except Exception as e:
                    if self.verbose:
                        print(f"[aggregate] coeff convert fail layer={name}: {e}")
                    continue

                if torch.isnan(c_cpu).any() or not np.isfinite(float(torch.norm(c_cpu).cpu().item())):
                    if self.verbose:
                        print(f"[aggregate] invalid coeff (NaN/Inf) for layer={name}, skip client.")
                    continue

                coeffs_cpu.append(c_cpu)
                weights.append(float(n))
                if orig_energy is not None:
                    orig_energies.append(orig_energy)

            # 若该层无客户端参与，则跳过
            if len(coeffs_cpu) == 0:
                new_state[name] = global_state[name]
                continue

            # 在 CPU 上计算加权平均的低维系数
            w_cpu = torch.tensor(weights, device="cpu", dtype=coeffs_cpu[0].dtype)
            w_cpu = w_cpu / w_cpu.sum()
            C_cpu = torch.stack(coeffs_cpu, dim=0)  # [num_clients, r] on CPU
            c_bar_cpu = (w_cpu[:, None] * C_cpu).sum(dim=0)

            # 将平均系数搬到目标设备（GPU）进行重构
            c_bar_dev = c_bar_cpu.to(updater.device, non_blocking=True)
            grad_bar_dev = updater.reconstruct(c_bar_dev, chunk_size=getattr(self, "chunk_size", 4096))

            # 重构结果搬回更新设备
            grad_bar = grad_bar_dev.to(self.device) if grad_bar_dev.device != self.device else grad_bar_dev

            # 提前计算诊断量，避免在删除后再访问
            if self.verbose and (round_num % max(1, self.k_update) == 0):
                mean_coeff_norm = float(torch.norm(C_cpu, dim=1).mean().cpu().item())
                cbar_norm = float(torch.norm(c_bar_cpu).cpu().item())
                grad_norm = float(torch.norm(grad_bar).cpu().item())
                print(f"[aggregate][round {round_num}] layer={name} clients={len(weights)} mean_coeff_norm={mean_coeff_norm:.4e} cbar_norm={cbar_norm:.4e} grad_norm={grad_norm:.4e} eff_lr={effective_lr:.3e}")

            # 提前释放中间变量降低峰值显存
            del c_bar_dev, grad_bar_dev

            # 当前层参数搬上 GPU 做更新
            param_cpu = global_state[name]
            param_dev = param_cpu.to(self.device)
           
            if round_num < self.warmup_rounds + self.warmup_extra:
                 subspace_scale = getattr(self, "subspace_lr_scale", 1.0)
            else:
                 subspace_scale = getattr(self, "subspace_lr_scale", 10.0)
            new_param_dev = param_dev - effective_lr * subspace_scale * grad_bar.reshape_as(param_dev)

            # 更新结果搬回 CPU，存入 new_state
            new_state[name] = new_param_dev.detach().cpu()

            # 释放 GPU 临时变量
            del param_cpu, param_dev, new_param_dev, grad_bar

            # 子空间更新（在 CPU 上执行）
            if (round_num + 1) % self.k_update == 0:
                try:
                    if len(scales_list) == len(coeffs_cpu) and len(scales_list) > 0:
                        updater.update(coeffs_cpu, weights, scales=scales_list)
                    else:
                        updater.update(coeffs_cpu, weights, scales=None)
                except Exception as e:
                    if self.verbose:
                        print(f"[Subspace Update] layer={name} update failed: {e}")

            # 清理临时 CPU 变量
            del coeffs_cpu, C_cpu, c_bar_cpu

        # 所有层处理完后，载入新参数
        self.global_model.load_state_dict(new_state)

        # 循环结束后统一释放显存缓存（而非每层调用）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            if round_num >= int(getattr(self, "warmup_rounds", 0)):
                print_every = int(getattr(self, "print_comm_every", 5))
                if print_every > 0 and (round_num % print_every == 0):
                    m = len(client_sizes)

                    R_subspace = int(getattr(self, "est_subspace_rounds", 1))
                    bits = int(getattr(self, "quant_bits", 8))

                    info = self.compute_subspace_comm(
                        m=m, R_subspace=R_subspace, bits=bits
                    )

                    def human(x):
                        if x >= 1024 * 1024:
                            return f"{x/1024/1024:.2f} MB"
                        if x >= 1024:
                            return f"{x/1024:.2f} KB"
                        return f"{x} B"

                    print(f"[SubspaceComm][round {round_num}] "
                          f"rank={info['rank']} layers={info['num_layers']} "
                          f"clients={info['clients_per_round']} R_subspace={info['R_subspace']} bits={info['bits']}")

                    print(f"  per-client: baseline={human(info['baseline_per_client'])}   pdq8={human(info['pdq_per_client'])}")
                    print(f"  per-round : baseline={human(info['baseline_per_round'])}   pdq8_per_round={human(info['pdq_per_round'])}")
                    print(f"  total R   : baseline={human(info['baseline_total'])}   pdq8_total={human(info['pdq_total'])}")

        except Exception:
            pass

    def aggregate_full_models(self, client_states: List[Dict[str, torch.Tensor]], client_sizes: List[int]):
        """
        稳健 warm-up 聚合（替换原实现）：
        - 跳过非浮点 state（例如 BatchNorm.num_batches_tracked）
        - 对每层按客户端方向归一化后加权累加，再用均值尺度恢复
        """
        total = float(sum(client_sizes))
        if total == 0:
            return

        device = next(self.global_model.parameters()).device
        # 当前全局状态（保留 dtype/device）
        global_state = {k: v.detach().to(device) for k, v in self.global_model.state_dict().items()}

        # 初始化累加器：与 global_state 保持相同 dtype/device
        delta_accum = {k: torch.zeros_like(v, dtype=v.dtype, device=v.device) for k, v in global_state.items()}
        per_layer_orig_norms = {k: [] for k in global_state.keys()}
        per_layer_abs_means = {k: [] for k in global_state.keys()}

        eps = 1e-8
        # 遍历每个客户端的 state_dict，累积方向（跳过非浮点项）
        for st, n in zip(client_states, client_sizes):
            for name, local_tensor in st.items():
                if name not in global_state:
                    continue

                g_global = global_state[name]
                # 跳过非浮点（如 long/int/bool buffer）
                if not torch.is_floating_point(g_global):
                    if getattr(self, "verbose", False):
                        print(f"[aggregate_full_models] skip non-float tensor '{name}'")
                    continue

                # 将 local_tensor 转为与 global 一致的 device/dtype，避免隐式迁移
                local_tensor = local_tensor.detach().to(device=g_global.device, dtype=g_global.dtype)
                delta = local_tensor - g_global  # same dtype/device as g_global

                # 计算范数（保证为浮点）
                orig_norm = float(delta.view(-1).norm().cpu().item())
                norm = orig_norm + eps
                direction = delta / norm  # 层内方向归一化，dtype matches g_global.dtype

                # 累加，按样本数加权
                delta_accum[name] = delta_accum[name] + (direction * float(n))

                per_layer_orig_norms[name].append(orig_norm)
                per_layer_abs_means[name].append(float(direction.abs().mean().cpu().item()))

        # 诊断打印（原始 norm 统计）
        if getattr(self, "verbose", False):
            for name, norms in per_layer_orig_norms.items():
                if len(norms) == 0:
                    continue
                mean_norm = float(np.mean(norms))
                max_norm = float(np.max(norms))
                print(f"[aggregate_full_models] layer={name} provided_clients={len(norms)} mean_delta_norm={mean_norm:.4e} max_delta_norm={max_norm:.4e}")

        # 构造新的 state：mean_direction * mean_orig_norm * server_lr_scale
        new_state = {}
        for name, g in global_state.items():
            # 若该层没有被任何客户端提供（或是非浮点项），保持原样
            if name not in delta_accum or (len(per_layer_orig_norms.get(name, [])) == 0):
                new_state[name] = g
                continue

            mean_direction = delta_accum[name] / total  # 平均方向（dtype 与 g 相同）
            norms = per_layer_orig_norms.get(name, [])
            mean_orig_norm = float(np.mean(norms)) if len(norms) > 0 else 1.0

            applied = mean_direction * mean_orig_norm * float(getattr(self, "server_lr_scale", 1.0))
            new_param = g + applied

            # 保证 new_state 的 dtype/device 与原 global 一致
            new_state[name] = new_param.to(device=g.device, dtype=g.dtype)

        # 应用更新并诊断 Δglobal（跳过非浮点项）
        self.global_model.load_state_dict(new_state)

        prev_global = getattr(self, "prev_global", None)
        current_global = {k: v.detach().cpu() for k, v in self.global_model.state_dict().items()}
        if prev_global is not None and getattr(self, "verbose", False):
            total_delta = 0.0
            for k, cur in current_global.items():
                if not torch.is_floating_point(cur):
                    continue
                total_delta += float((cur - prev_global[k]).to(torch.float32).view(-1).norm().item())
            print(f"[Server] Δglobal (normalized warm-up): {total_delta:.6e}")
        self.prev_global = {k: v.clone().detach() for k, v in current_global.items()}
    def get_global_model_params(self):
        return self.global_model.state_dict()
    
    def get_subspace(self, layer_name):
        return self.subspaces[layer_name].U

    def project_client_update(self, layer_name, grad):
        """供客户端调用：将梯度投影到共享子空间"""
        return self.subspaces[layer_name].project(grad)

    def broadcast_model(self):
        """向客户端发送全局模型参数"""
        return {k: v.detach().clone() for k, v in self.global_model.state_dict().items()}
  
    def broadcast_subspaces(self):
        """向客户端发送全局子空间"""
        return {name: updater.U.detach().cpu().clone() for name, updater in self.subspaces.items()}
    
    def test(self, test_loader, device, round_num=None):
        self.global_model.to(device)
        self.global_model.eval()
        correct = 0
        total = 0
        total_test_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                outputs = self.global_model(x)
                loss = F.cross_entropy(outputs, y)
                total_test_loss += loss.item() * y.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        avg_loss = total_test_loss / total
        accuracy = correct / total

        return avg_loss, accuracy
    def compute_subspace_comm(self, m: int, R_subspace: int = 1, bits: int = 8):
        """
        仅计算子空间阶段（每层发送 r 个系数）的通信量估算。
        - m: 每轮参与客户端数
        - R_subspace: 子空间阶段轮数（用于计算总量），默认 1 表示只本轮
        - bits: 量化位宽（例如 8 表示 int8 PDQ）
        返回 summary dict（字节）
        """
        r = int(getattr(self, "rank", self.rank))

        # number of layers
        if isinstance(self.layer_dims, dict):
            num_layers = len(self.layer_dims)
        else:
            num_layers = len(list(self.layer_dims))

        # baseline 32bit: r*4 + 4 bytes
        baseline_per_layer = 4 * r + 4

        # PDQ8 per-dim: quant(r * bits/8) + scales(r * 4) + 4
        q_bytes = int(r * (bits // 8)) if bits % 8 == 0 else int(r * (bits / 8.0))
        pdq_per_layer = q_bytes + 4 * r + 4

        baseline_per_client = baseline_per_layer * num_layers
        pdq_per_client = pdq_per_layer * num_layers

        baseline_per_round = baseline_per_client * m
        pdq_per_round = pdq_per_client * m

        baseline_total = baseline_per_round * R_subspace
        pdq_total = pdq_per_round * R_subspace

        return {
            "rank": r,
            "num_layers": num_layers,
            "baseline_per_layer": int(baseline_per_layer),
            "pdq_per_layer": int(pdq_per_layer),
            "baseline_per_client": int(baseline_per_client),
            "pdq_per_client": int(pdq_per_client),
            "baseline_per_round": int(baseline_per_round),
            "pdq_per_round": int(pdq_per_round),
            "baseline_total": int(baseline_total),
            "pdq_total": int(pdq_total),
            "bits": bits,
            "clients_per_round": m,
            "R_subspace": R_subspace
        }


    

def make_layer_dims(model, include_bias=False, min_size=1, select_fn=None):
        """
        model: nn.Module
        include_bias: 是否包含 bias 参数
        min_size: 忽略小于该阈值的参数（降低子空间数量）
        select_fn: 可选函数 (name, param) -> bool 进一步筛选
        返回: dict name->numel
        """
        layer_dims = {}
        for name, p in model.named_parameters():
            if not include_bias and name.endswith(".bias"):
                continue
            if p.numel() < min_size:
                continue
            if select_fn is not None and not select_fn(name, p):
                continue
            layer_dims[name] = p.numel()
        return layer_dims