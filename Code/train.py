import argparse
import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from model_arch import MNIST_TENet
from model_arch_legacy import MNIST_TENet as MNIST_TENet_Legacy
# 导入我们之前定义的 MNIST_TENet 和相关层
# (假设上面的类定义已经在当前环境中)

def build_args():
    parser = argparse.ArgumentParser(description='Train MNIST_TENet on MNIST')
    default_save_path = 'checkpoints/tenet_best.pth'
    parser.add_argument('--data-dir', type=str, default='./data', help='MNIST data directory')
    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='train batch size')
    parser.add_argument('--test-batch-size', type=int, default=1000, help='test batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lr-scheduler', type=str, default='warmup-cosine', choices=['none', 'warmup-cosine'], help='learning rate scheduler type')
    parser.add_argument('--warmup-epochs', type=int, default=3, help='number of warmup epochs for warmup-cosine scheduler')
    parser.add_argument('--min-lr-ratio', type=float, default=0.1, help='minimum lr ratio for cosine decay')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay for Adam')
    parser.add_argument('--label-smoothing', type=float, default=0.05, help='label smoothing for CrossEntropyLoss')
    parser.add_argument('--trace-lr-mult', type=float, default=0.05, help='learning rate multiplier for trace branch parameters')
    parser.add_argument('--alpha', type=int, default=6, help='channel expansion per graphified layer')
    parser.add_argument('--trace-init', type=float, default=0.005, help='initial trace enhancement scale')
    parser.add_argument('--phi-temp-init', type=float, default=1.5, help='initial temperature for phi output scaling')
    parser.add_argument('--phi-residual-init', type=float, default=0.05, help='initial residual strength for phi output')
    parser.add_argument('--classifier-dropout', type=float, default=0.1, help='dropout ratio in classifier head')
    parser.add_argument('--no-learnable-phi', action='store_true', help='disable learnable phi and use abs difference')
    parser.add_argument('--no-trace-branch', action='store_true', help='disable trace enhancement branch in graph layer')
    parser.add_argument('--no-edge-enhance', action='store_true', help='disable edge square/product enhancement terms')
    parser.add_argument('--no-directed-flow', action='store_true', help='disable directed incidence-flow enhancement')
    parser.add_argument('--no-cross-guidance', action='store_true', help='disable cross-stream guidance (C->T and T->C)')
    parser.add_argument('--use-chebyshev', action='store_true', help='enable chebyshev second-order topology fusion')
    parser.add_argument('--use-spatial-bridge', action='store_true', help='enable spatial bridge (depthwise dilated conv)')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--num-workers', type=int, default=0, help='number of DataLoader workers')
    parser.add_argument('--cpu', action='store_true', help='force cpu training')
    parser.add_argument('--save-best', action='store_true', help='save checkpoint with best test accuracy')
    parser.add_argument('--no-save-best', action='store_true', help='disable saving checkpoint with best test accuracy')
    parser.add_argument('--save-path', type=str, default=default_save_path, help='path to save best checkpoint')
    parser.add_argument('--resume', type=str, default='', help='checkpoint path to resume training from')
    parser.add_argument('--no-aug', action='store_true', help='disable data augmentation for training set')
    parser.add_argument('--model-version', type=str, default='current', choices=['current', 'legacy'], help='choose model architecture version')
    args = parser.parse_args()
    if not args.save_best and not args.no_save_best:
        args.save_best = True
    if args.no_save_best:
        args.save_best = False
    if args.save_path == default_save_path:
        if args.model_version == 'legacy':
            args.save_path = 'checkpoints/tenet_legacy_best.pth'
        else:
            args.save_path = 'checkpoints/tenet_current_best.pth'
    return args


def train_model(args):
    # 1. 硬件配置
    use_cuda = torch.cuda.is_available() and not args.cpu
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)
    print(f"Using device: {device}")

    # 2. 数据准备 (MNIST 28x28)
    if args.no_aug:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomAffine(
                degrees=10,
                translate=(0.1, 0.1),
                scale=(0.95, 1.05),
                fill=0
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(args.data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST(args.data_dir, train=False, transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # 3. 实例化模型、损失函数和优化器
    model_cls = MNIST_TENet if args.model_version == 'current' else MNIST_TENet_Legacy
    common_kwargs = dict(
        alpha=args.alpha,
        trace_init=args.trace_init,
        phi_temp_init=args.phi_temp_init,
        phi_residual_init=args.phi_residual_init,
        classifier_dropout=args.classifier_dropout,
        use_learnable_phi=not args.no_learnable_phi
    )
    if args.model_version == 'current':
        common_kwargs.update(
            use_trace_branch=not args.no_trace_branch,
            use_edge_enhance=not args.no_edge_enhance,
            use_directed_flow=not args.no_directed_flow,
            use_cross_stream_guidance=not args.no_cross_guidance,
            use_chebyshev=args.use_chebyshev,
            use_spatial_bridge=args.use_spatial_bridge
        )

    model = model_cls(**common_kwargs).to(device)
    trace_params, base_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'trace_proj' in name or 'trace_logit' in name or 'trace_scale' in name:
            trace_params.append(param)
        else:
            base_params.append(param)

    optimizer_param_groups = [{'params': base_params, 'lr': args.lr}]
    trace_lr = args.lr * args.trace_lr_mult
    if len(trace_params) > 0:
        optimizer_param_groups.append({'params': trace_params, 'lr': trace_lr})

    optimizer = optim.Adam(optimizer_param_groups, weight_decay=args.weight_decay)

    scheduler = None
    if args.lr_scheduler == 'warmup-cosine':
        warmup_epochs = max(int(args.warmup_epochs), 0)
        min_lr_ratio = float(max(min(args.min_lr_ratio, 1.0), 0.0))

        def lr_lambda(epoch_idx):
            if args.epochs <= 1:
                return 1.0
            if warmup_epochs > 0 and epoch_idx < warmup_epochs:
                return (epoch_idx + 1) / warmup_epochs
            decay_total = max(args.epochs - warmup_epochs, 1)
            decay_step = min(max(epoch_idx - warmup_epochs, 0), decay_total)
            cosine = 0.5 * (1.0 + math.cos(math.pi * decay_step / decay_total))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    print(
        f"Epochs: {args.epochs}, Batch size: {args.batch_size}, Test batch size: {args.test_batch_size}, "
        f"Learning rate: {args.lr}, LR scheduler: {args.lr_scheduler}, Warmup epochs: {args.warmup_epochs}, "
        f"Min LR ratio: {args.min_lr_ratio}, Weight decay: {args.weight_decay}, Label smoothing: {args.label_smoothing}, "
        f"Trace LR Mult: {args.trace_lr_mult}, Trace LR: {trace_lr}, Optimizer: Adam"
    )
    print(
        f"Alpha: {args.alpha}, Trace init: {args.trace_init}, "
        f"Phi temp init: {args.phi_temp_init}, Phi residual init: {args.phi_residual_init}, "
        f"Classifier dropout: {args.classifier_dropout}, "
        f"Learnable phi: {not args.no_learnable_phi}, Augment: {not args.no_aug}, "
        f"Model version: {args.model_version}, Seed: {args.seed}"
    )
    if args.model_version == 'current':
        print(
            f"Ablation switches -> Trace: {not args.no_trace_branch}, EdgeEnhance: {not args.no_edge_enhance}, "
            f"DirectedFlow: {not args.no_directed_flow}, CrossGuidance: {not args.no_cross_guidance}, "
            f"Chebyshev: {args.use_chebyshev}, SpatialBridge: {args.use_spatial_bridge}"
        )
    if args.save_best:
        print(f"Save best: True, Save path: {args.save_path}")
    else:
        print("Save best: False")
    print(f"Total params: {total_params:,}, Trainable params: {trainable_params:,}")

    start_epoch = 1
    best_test_acc = -1.0
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        ckpt_epoch = int(checkpoint.get('epoch', 0))
        start_epoch = ckpt_epoch + 1
        best_test_acc = float(checkpoint.get('test_acc', -1.0))
        print(f"Resumed from: {args.resume}")
        print(f"Resume epoch: {ckpt_epoch}, Start from epoch: {start_epoch}, Best Test Acc: {best_test_acc:.2f}%")

    # 4. 训练循环
    if start_epoch > args.epochs:
        print(f"No training run because start epoch {start_epoch} > total epochs {args.epochs}.")
        return

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_samples = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]", leave=False)
        for data, target in train_bar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            running_correct += pred.eq(target).sum().item()
            running_samples += data.size(0)

            train_bar.set_postfix({
                'loss': f"{running_loss / running_samples:.4f}",
                'acc': f"{100.0 * running_correct / running_samples:.2f}%"
            })

        train_loss = running_loss / running_samples
        train_acc = 100.0 * running_correct / running_samples

        # 5. 每个 Epoch 结束后进行测试
        test_loss, test_acc = test(model, device, test_loader, criterion, epoch, args.epochs)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}% | "
            f"Gap: {train_acc - test_acc:.2f}% | "
            f"LR(main): {optimizer.param_groups[0]['lr']:.2e}"
        )

        if scheduler is not None:
            scheduler.step()

        if args.save_best and test_acc > best_test_acc:
            best_test_acc = test_acc
            save_dir = os.path.dirname(args.save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                    'test_acc': test_acc,
                    'test_loss': test_loss,
                    'args': vars(args),
                },
                args.save_path
            )
            print(f"New best saved -> Epoch: {epoch}, Test Acc: {test_acc:.2f}%, Path: {args.save_path}")

def test(model, device, test_loader, criterion, epoch, epochs):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc=f"Epoch {epoch}/{epochs} [Test]", leave=False)
        for data, target in test_bar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            batch_loss = criterion(output, target).item()
            test_loss += batch_loss * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)

            test_bar.set_postfix({
                'loss': f"{test_loss / total:.4f}",
                'acc': f"{100.0 * correct / total:.2f}%"
            })

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy

if __name__ == "__main__":
    args = build_args()
    train_model(args)