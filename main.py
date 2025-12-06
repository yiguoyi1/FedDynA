import torch
import random
import numpy as np
from ast import arg
import argparse
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Subset
from dataset import get_dataset_loaders
import models
from client import Client
from server import make_layer_dims,FedDynAServer
from utils import plot_accuracy_curve, dirichlet_partition, WrapperDataset
import matplotlib
import os
import time
from datetime import datetime
matplotlib.use('Agg')

def get_num_classes(dataset):
    labels = [dataset[i][1] for i in range(len(dataset))]
    return len(set(labels))

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
def main(args):

    set_seed(args.seed)
    # 获取当前时间，并创建保存路径
    current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    save_dir = os.path.join(args.save_path, current_time)
    # 创建文件夹
    os.makedirs(save_dir, exist_ok=True)
    hyperparams_file = os.path.join(save_dir, 'hyperparams.txt')
    with open(hyperparams_file, 'w') as f:
        for k, v in vars(args).items():
            f.write(f"{k}={v}\n")
            print (f"{k}={v}")
    if torch.cuda.is_available():
        # 如果有CUDA可用，使用指定的GPU
        device = torch.device(f"cuda:{args.gpu}")
    else:
        # 回退到CPU
        device = torch.device("cpu")   
    print(f"Using device: {device}")
    train_dataset, test_dataset = get_dataset_loaders(args.dataset_name)
    client_data = dirichlet_partition(train_dataset, args.clients_num, alpha=args.alpha)
    print(f"Data partitioned among {args.clients_num} clients with Dirichlet alpha={args.alpha}")
    num_classes = get_num_classes(train_dataset)
    if args.dataset_name == 'cifar10':
        global_model = models.ResNet18(num_classes=num_classes)
    elif args.dataset_name == 'FashionMNIST':
        global_model = models.LeNet(num_classes=num_classes)
    elif args.dataset_name == 'cifar100':
        global_model = models.ResNet34(num_classes=num_classes)
    global_model.to(device)
    
    clients = [
        Client(args, i, model=global_model, 
            train_loader=DataLoader(client_data[i], batch_size=args.batch_size,shuffle=True), device=device, 
            num_classes=num_classes,
            save_dir=save_dir)
        for i in range(args.clients_num)
    ]
    layer_dims = make_layer_dims(global_model, include_bias=False, min_size=64, select_fn=lambda n,p: "weight" in n)
    rank = 32
    alpha = args.subspace_updater_alpha
    if args.server_lr is None:
       args.server_lr = 1
    server_lr = args.server_lr
    server = FedDynAServer(model=global_model, layer_dims=layer_dims, rank=rank, alpha=alpha, lr=server_lr, device=device, warmup_rounds=args.warmup_rounds, warmup_lr_scale=1, warmup_extra=10, verbose=True)
    server.quant_bits = args.quant_bits
    server.quant_symmetric = args.quant_symmetric
    # 记录每轮准确率
    accuracy_list = []
    loss_list = []

     # 添加指标记录列表
     
    metrics_history = {
        'round': [],
        'avg_task_loss': [],
        'test_accuracy': [],
        'test_loss': []
    }

    log_file = os.path.join(save_dir, 'global_objective_log.csv')
    with open(log_file, 'w') as f:
        f.write('round,avg_task_loss,test_accuracy,test_loss\n')
    
    for round in range(args.rounds):
        print(f'Round {round + 1}/{args.rounds}')
        # 随机选择部分客户端
        m = max(int(args.C * args.clients_num), 1)
        selected_clients = random.sample(clients, m)
        # 下发全局模型
        global_params = server.get_global_model_params()
        # 决定是否启用子空间投影（warm-up 期间不使用子空间）
        do_subspace = (round >= args.warmup_rounds)
        if do_subspace:
            subspaces = server.broadcast_subspaces()
            subspaces_gpu = {name: U.to(device, non_blocking=True) for name, U in subspaces.items()}
            subspaces = subspaces_gpu

        client_updates = []   # list of coeffs_dict (for subspace aggregation)
        client_states = []    # list of full model states (for warm-up FedAvg)
        client_sizes = []     # list of n_samples
        client_stats = []     # optional: metrics
        client_loss_list = []

        for client in selected_clients:
            client.set_model_params(global_params)
            if do_subspace:
                client.set_subspaces(subspaces)

            # 如果是子空间模式 (do_subspace)，客户端不做本地 step（由 server 统一更新）
            coeffs_dict, n_samples, loss, acc, local_state = client.train(round, do_local_update=(not do_subspace))

            client_sizes.append(n_samples)
            client_loss_list.append(loss)
            client_stats.append({"loss": loss, "acc": acc, "id": client.id})

            if do_subspace:
                coeffs_cpu = {}
                for name, c in coeffs_dict.items():
                   # 支持两种上传格式：量化后的 tuple 或 原始 tensor
                    if isinstance(c, tuple):
                        # c == (tensor_q_cpu, scale, min_val, dtype_name)
                        coeffs_cpu[name] = c
                    else:
                        coeffs_cpu[name] = c.detach().cpu()
                client_updates.append(coeffs_cpu)
            else:
                client_states.append(local_state)

        # 聚合：warm-up 期间做 FedAvg 参数平均，之后使用低维系数聚合
        if round < args.warmup_rounds:
            server.aggregate_full_models(client_states, client_sizes)
        else:
            server.aggregate(client_updates, client_sizes, round)
        avg_task_loss = np.mean(client_loss_list)
         # 测试
        test_loss, test_acc = server.test(torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size), device=device,
                                        round_num=round)
        metrics_history['round'].append(round)
        metrics_history['avg_task_loss'].append(avg_task_loss)
        metrics_history['test_accuracy'].append(test_acc)
        metrics_history['test_loss'].append(test_loss)
        with open(log_file, 'a') as f:
            f.write(f'{round},{avg_task_loss},{test_acc},{test_loss}\n')
        print(f"[Round {round}] Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        accuracy_list.append(test_acc)
        loss_list.append(test_loss)
        # 绘制准确率曲线
        plot_accuracy_curve(save_dir, accuracy_list, loss_list)

        print("Training Complete.")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning with FedAvg")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--clients_num', type=int, default=20, help="Number of clients")
    parser.add_argument('--C', type=float, default=0.25, help="Fraction of clients participating per round")
    parser.add_argument('--batch_size', type=int, default=4, help="Local batch size")
    parser.add_argument('--E', type=int, default=2, help="Local epochs")
    parser.add_argument('--eta', type=float, default=0.1, help="Learning rate")
    parser.add_argument('--rounds', type=int, default=200, help="Number of communication rounds")
    parser.add_argument('--alpha', type=float, default=1, help="Dirichlet parameter for data partitioning")
    parser.add_argument('--warmup_rounds', type=int, default=5, help="Number of warm-up rounds to run standard FedAvg before enabling subspace projection")
    parser.add_argument('--dataset_name', type=str, choices=['cifar10', 'cifar100', 'FashionMNIST', 'SVHN'], default='FashionMNIST',
                        help="Name of the dataset")
    parser.add_argument('--save_path', type=str, default='result', help="Directory to save results and logs")
    parser.add_argument('--IID', action='store_true', help="使用IID数据分布（默认不使用）")
    parser.add_argument('--non-IID', action='store_false', dest='IID', help="使用非IID数据分布")
    parser.add_argument( "--gpu", type=int, default=0, help="Specify which GPU to use" )
    parser.add_argument('--subspace_updater_alpha', type=float, default=0.9, help="Alpha parameter for the subspace updater")
    parser.add_argument('--server_lr', type=float, default=None, help="Server learning rate ")
    parser.add_argument('--quant_bits', type=int, default=4, help="量化位宽")
    parser.add_argument('--quant_symmetric', action='store_true', help="默认使用非对称量化")
    # Parse arguments
    args = parser.parse_args()
    if args.dataset_name == 'cifar10':
        args.batch_size=64
        args.E = 5
        args.eta=0.1
    elif args.dataset_name == 'FashionMNIST':
        args.batch_size = 64
        args.E = 4
        args.eta=0.1
        args.server_lr=20
    elif args.dataset_name == 'SVHN':
        args.batch_size = 64
        args.E = 3
    elif args.dataset_name == 'cifar100':
        args.batch_size = 64
        args.E = 5
    # Run the federated learning training process
    main(args)


