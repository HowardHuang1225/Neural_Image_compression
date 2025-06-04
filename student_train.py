import os
import torch
import random
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

import argparse
import math
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.zoo import models
from pytorch_msssim import ms_ssim

from models import TCM
from models import student_TCM
from torch.utils.tensorboard import SummaryWriter   
import os


def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.)

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, type='mse'):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.type = type

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        if self.type == 'mse':
            out["mse_loss"] = self.mse(output["x_hat"], target)
            out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
        else:
            out['ms_ssim_loss'] = compute_msssim(output["x_hat"], target)
            out["loss"] = self.lmbda * (1 - out['ms_ssim_loss']) + out["bpp_loss"]

        return out

class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)
        
def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(
    student_model, # 將 model 改名為 student_model，更清晰
    teacher_model, # 新增教師模型參數
    criterion,
    train_dataloader,
    optimizer,
    aux_optimizer,
    epoch,
    clip_max_norm,
    type='mse',
    distillation_lambda_y=0.1,    # y 的蒸餾損失權重
    distillation_lambda_z=0.1,    # z 的蒸餾損失權重
    distillation_lambda_param=0.1 # means/scales 的蒸餾損失權重
):
    student_model.train()
    teacher_model.eval() # 確保教師模型在評估模式且不更新權重
    device = next(student_model.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device)
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        # --- Forward pass for Student Model ---
        out_student = student_model(d)

        # --- Forward pass for Teacher Model (no grad) ---
        with torch.no_grad():
            out_teacher = teacher_model(d)

        # --- Calculate Rate-Distortion Loss for Student Model ---
        out_criterion = criterion(out_student, d)
        total_loss = out_criterion["loss"]

        # --- Calculate Distillation Loss ---
        # 1. Latent Representation Distillation (y and z)
        # 確保學生和教師模型的輸出維度匹配 (在設計學生模型時需要注意)
        # 例如，如果學生模型的 M 和 N 減小了，那麼其 y 和 z 的維度也會變小
        # 這時候直接的 MSE 損失可能不適用，需要考慮投影或調整學生模型輸出維度。
        # 但通常蒸餾會讓學生模型輸出相同或相似維度的特徵。
        
        # 這裡假設學生模型 output 的 y 和 z 已經和教師模型匹配 (在您調整 N, M, config, head_dim 後，可能需要確保這點)
        # 或者，您可以選擇蒸餾 g_a 的輸出 'y' 和 h_a 的輸出 'z'
        # 如果維度不匹配，您可能需要額外的投影層 (projection layer) 將學生模型的特徵映射到教師模型的特徵空間
        # 為了簡潔起見，這裡假設您可以直接計算損失

        # 蒸餾 y:
        # out_student["para"]["y"] 是學生模型的 y， out_teacher["para"]["y"] 是教師模型的 y
        # 確保尺寸匹配或使用插值/投影
        distillation_loss_y = torch.nn.functional.mse_loss(out_student["para"]["y"], out_teacher["para"]["y"])
        total_loss += distillation_lambda_y * distillation_loss_y

        # 蒸餾 z:
        # 由於 entropy_bottleneck 的輸出 z 通常是量化過的，蒸餾 z_hat 可能比蒸餾原始 z 更好
        # out_student["z"] 是 entropy_bottleneck 壓縮前的 z， out_teacher["z"] 類似
        # 但 out_net["likelihoods"]["z"] 是 entropy_bottleneck 的輸出，通常是直接用於計算 BPP 的
        # 更常用的是對 z_hat 進行蒸餾 (量化後的潛在表示)
        # 這裡假設 out_student["z_hat"] 是學生模型量化後的 z
        # out_student["z"] 是 g_a 的輸出 y 經過 h_a 得到 z 的原始值，但壓縮時是 z_hat
        # 所以，如果您的模型中能取到 z_hat，用 z_hat 蒸餾更合理
        # 根據您提供的 TCM forward 函數，z_hat 是 ste_round(z_tmp) + z_offset
        # 您可能需要修改 forward 函數讓 z_hat 成為一個 output key
        # 為了示範，我們先用 z_likelihoods 進行蒸餾 (這類似於知識蒸餾中對 logits 的蒸餾)
        # 或者蒸餾原始的 z (來自 h_a 的輸出)
        # 讓我們使用 out_student["z"] (如果它代表 g_a 的輸出經過 h_a)
        # 但更直接的是蒸餾 `latent_means` 和 `latent_scales`
        
        # 蒸餾 z 蒸餾 y 一樣
        distillation_loss_z = torch.nn.functional.mse_loss(out_student["z"], out_teacher["z"])
        total_loss += distillation_lambda_z * distillation_loss_z

        # 2. Mean and Scale Distillation (條件機率分佈參數)
        # 這是非常關鍵的部分，讓學生模型學習教師模型如何預測壓縮參數
        distillation_loss_means = torch.nn.functional.mse_loss(out_student["para"]["means"], out_teacher["para"]["means"])
        distillation_loss_scales = torch.nn.functional.mse_loss(out_student["para"]["scales"], out_teacher["para"]["scales"])
        distillation_loss_param = distillation_loss_means + distillation_loss_scales
        total_loss += distillation_lambda_param * distillation_loss_param

        # --- Backpropagation ---
        total_loss.backward() # 對總損失進行反向傳播

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = student_model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 1000 == 0:
            # 打印蒸餾損失以便觀察
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tTotal Loss: {total_loss.item():.3f} |'
                f'\tRD Loss: {out_criterion["loss"].item():.3f} |' # 原來的率失真損失
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f'\tDistil Y: {distillation_loss_y.item():.3f} |'
                f'\tDistil Z: {distillation_loss_z.item():.3f} |'
                f'\tDistil Params: {distillation_loss_param.item():.3f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )
            if type == 'mse':
                print(f'\tMSE loss: {out_criterion["mse_loss"].item():.3f}')
            else:
                print(f'\tMS_SSIM loss: {out_criterion["ms_ssim_loss"].item():.3f}')


def test_epoch(epoch, test_dataloader, model, criterion, type='mse'):
    model.eval()
    device = next(model.parameters()).device
    if type == 'mse':
        loss = AverageMeter()
        bpp_loss = AverageMeter()
        mse_loss = AverageMeter()
        aux_loss = AverageMeter()

        with torch.no_grad():
            for d in test_dataloader:
                d = d.to(device)
                out_net = model(d)
                out_criterion = criterion(out_net, d)

                aux_loss.update(model.aux_loss())
                bpp_loss.update(out_criterion["bpp_loss"])
                loss.update(out_criterion["loss"])
                mse_loss.update(out_criterion["mse_loss"])

        print(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {loss.avg:.3f} |"
            f"\tMSE loss: {mse_loss.avg:.3f} |"
            f"\tBpp loss: {bpp_loss.avg:.2f} |"
            f"\tAux loss: {aux_loss.avg:.2f}\n"
        )

    else:
        loss = AverageMeter()
        bpp_loss = AverageMeter()
        ms_ssim_loss = AverageMeter()
        aux_loss = AverageMeter()

        with torch.no_grad():
            for d in test_dataloader:
                d = d.to(device)
                out_net = model(d)
                out_criterion = criterion(out_net, d)

                aux_loss.update(model.aux_loss())
                bpp_loss.update(out_criterion["bpp_loss"])
                loss.update(out_criterion["loss"])
                ms_ssim_loss.update(out_criterion["ms_ssim_loss"])

        print(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {loss.avg:.3f} |"
            f"\tMS_SSIM loss: {ms_ssim_loss.avg:.3f} |"
            f"\tBpp loss: {bpp_loss.avg:.2f} |"
            f"\tAux loss: {aux_loss.avg:.2f}\n"
        )

    return loss.avg


def save_checkpoint(state, is_best, epoch, save_path, filename):
    torch.save(state, save_path + "checkpoint_latest.pth.tar")
    if epoch % 5 == 0:
        torch.save(state, filename)
    if is_best:
        torch.save(state, save_path + "checkpoint_best.pth.tar")

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="bmshj2018-factorized",
        choices=models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=50,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=0,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=3,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, default=100, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--type", type=str, default='mse', help="loss type", choices=['mse', "ms-ssim"])
    parser.add_argument("--save_path", type=str, help="save_path")
    parser.add_argument(
        "--skip_epoch", type=int, default=0
    )
    parser.add_argument(
        "--N", type=int, default=128,
    )
    parser.add_argument(
        "--lr_epoch", nargs='+', type=int
    )
    parser.add_argument(
        "--continue_train", action="store_true", default=True
    )
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    type = args.type
    save_path = os.path.join(args.save_path, str(args.lmbda))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(save_path + "tensorboard/")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    writer = SummaryWriter(save_path + "tensorboard/")

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(device)
    device = 'cuda' # 強制使用cuda

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    # --- Student Model (學生模型) ---
    # 根據您的學生模型設計調整 N, M, config, head_dim
    student_net = student_TCM(config=[1, 1, 1, 1, 1, 1], head_dim=[4, 8, 16, 16, 8, 4], N=64, M=160)
    student_net = student_net.to(device)

    if args.cuda and torch.cuda.device_count() > 1:
        student_net = CustomDataParallel(student_net)

    # --- Teacher Model (教師模型) ---
    # 您需要指定教師模型的 N 和 M (通常會是原始較大的模型)
    # 並載入其預訓練權重
    teacher_N = 128 # 假設教師模型的 N 是 128
    teacher_M = 320 # 假設教師模型的 M 是 160 (或是 320 等)
    teacher_config = [2, 2, 2, 2, 2, 2] # 假設教師模型的 config
    teacher_head_dim = [8, 16, 32, 32, 16, 8] # 假設教師模型的 head_dim
    
    teacher_net = TCM(config=teacher_config, head_dim=teacher_head_dim, N=teacher_N, M=teacher_M)
    teacher_net = teacher_net.to(device)

    # 載入教師模型權重
    # 您需要提供教師模型檢查點的路徑
    teacher_checkpoint_path = "./out_0/0.05checkpoint_best.pth.tar" 
    print(f"Loading teacher model from {teacher_checkpoint_path}")
    teacher_checkpoint = torch.load(teacher_checkpoint_path, map_location=device)
    teacher_net.load_state_dict(teacher_checkpoint["state_dict"])
    teacher_net.eval() # 將教師模型設為評估模式，固定其參數
    for param in teacher_net.parameters():
        param.requires_grad = False # 確保教師模型參數不被訓練

    # --- Optimizer for Student Model ---
    optimizer, aux_optimizer = configure_optimizers(student_net, args)
    milestones = args.lr_epoch
    print("milestones: ", milestones)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)

    criterion = RateDistortionLoss(lmbda=args.lmbda, type=type)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint for student model
        print("Loading student model from", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        student_net.load_state_dict(checkpoint["state_dict"])
        if args.continue_train:
            last_epoch = checkpoint["epoch"] + 1
            optimizer.load_state_dict(checkpoint["optimizer"])
            aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            student_net, # 使用學生模型
            teacher_net, # 傳入教師模型
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            type
        )
        loss = test_epoch(epoch, test_dataloader, student_net, criterion, type) # 測試學生模型
        writer.add_scalar('test_loss', loss, epoch)
        lr_scheduler.step()

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": student_net.state_dict(), # 儲存學生模型權重
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                epoch,
                save_path,
                save_path + str(epoch) + "_checkpoint.pth.tar",
            )


if __name__ == "__main__":
    main(sys.argv[1:])
