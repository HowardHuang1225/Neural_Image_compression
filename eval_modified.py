import torch
import torch.nn.functional as F
from torchvision import transforms
from models import TCM
import warnings
import torch
import os
import sys
import math
import argparse
import time
import warnings
from pytorch_msssim import ms_ssim
from PIL import Image
warnings.filterwarnings("ignore")

print(torch.cuda.is_available())


def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

# def compute_msssim(a, b):
#     return -10 * math.log10(1-ms_ssim(a, b, data_range=1.).item())
def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()

def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
              for likelihoods in out_net['likelihoods'].values()).item()

def pad(x, p):
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)

def crop(x, padding):
    return F.pad(
        x,
        (-padding[0], -padding[1], -padding[2], -padding[3]),
    )

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example testing script.")
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--data", type=str, help="Path to dataset")
    parser.add_argument(
        "--real", action="store_true", default=True
    )
    parser.set_defaults(real=False)
    args = parser.parse_args(argv)
    return args


# def main(argv):
#     args = parse_args(argv)
#     p = 128
#     path = args.data
#     img_list = []
#     for file in os.listdir(path):
#         if file[-3:] in ["jpg", "png", "peg"]:
#             img_list.append(file)
#     if args.cuda:
#         device = 'cuda:0'
#     else:
#         device = 'cpu'
#     net = TCM(config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=128, M=320)
#     net = net.to(device)
#     net.eval()
#     count = 0
#     PSNR = 0
#     Bit_rate = 0
#     MS_SSIM = 0
#     total_time = 0
#     dictory = {}
#     if args.checkpoint:  # load from previous checkpoint
#         print("Loading", args.checkpoint)
#         checkpoint = torch.load(args.checkpoint, map_location=device)
#         for k, v in checkpoint["state_dict"].items():
#             dictory[k.replace("module.", "")] = v
#         net.load_state_dict(dictory)
#     if args.real:
#         net.update()
#         for img_name in img_list:
#             img_path = os.path.join(path, img_name)
#             img = transforms.ToTensor()(Image.open(img_path).convert('RGB')).to(device)
#             x = img.unsqueeze(0)
#             x_padded, padding = pad(x, p)
#             count += 1
#             with torch.no_grad():
#                 if args.cuda:
#                     torch.cuda.synchronize()
#                 s = time.time()
#                 out_enc = net.compress(x_padded)
#                 out_dec = net.decompress(out_enc["strings"], out_enc["shape"])
#                 if args.cuda:
#                     torch.cuda.synchronize()
#                 e = time.time()
#                 total_time += (e - s)
#                 out_dec["x_hat"] = crop(out_dec["x_hat"], padding)
#                 num_pixels = x.size(0) * x.size(2) * x.size(3)
#                 print(f'Bitrate: {(sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels):.3f}bpp')
#                 print(f'MS-SSIM: {compute_msssim(x, out_dec["x_hat"]):.2f}dB')
#                 print(f'PSNR: {compute_psnr(x, out_dec["x_hat"]):.2f}dB')
#                 Bit_rate += sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
#                 PSNR += compute_psnr(x, out_dec["x_hat"])
#                 MS_SSIM += compute_msssim(x, out_dec["x_hat"])

#     else:
#         for img_name in img_list:
#             img_path = os.path.join(path, img_name)
#             img = Image.open(img_path).convert('RGB')
#             x = transforms.ToTensor()(img).unsqueeze(0).to(device)
#             x_padded, padding = pad(x, p)
#             count += 1
#             with torch.no_grad():
#                 if args.cuda:
#                     torch.cuda.synchronize()
#                 s = time.time()
#                 out_net = net.forward(x_padded)
#                 if args.cuda:
#                     torch.cuda.synchronize()
#                 e = time.time()
#                 total_time += (e - s)
#                 out_net['x_hat'].clamp_(0, 1)
#                 out_net["x_hat"] = crop(out_net["x_hat"], padding)
#                 print(f'PSNR: {compute_psnr(x, out_net["x_hat"]):.2f}dB')
#                 print(f'MS-SSIM: {compute_msssim(x, out_net["x_hat"]):.2f}dB')
#                 print(f'Bit-rate: {compute_bpp(out_net):.3f}bpp')
#                 PSNR += compute_psnr(x, out_net["x_hat"])
#                 MS_SSIM += compute_msssim(x, out_net["x_hat"])
#                 Bit_rate += compute_bpp(out_net)
#     PSNR = PSNR / count
#     MS_SSIM = MS_SSIM / count
#     Bit_rate = Bit_rate / count
#     total_time = total_time / count
#     print(f'average_PSNR: {PSNR:.2f}dB')
#     print(f'average_MS-SSIM: {MS_SSIM:.4f}')
#     print(f'average_Bit-rate: {Bit_rate:.3f} bpp')
#     print(f'average_time: {total_time:.3f} ms')
def main(argv):
    args = parse_args(argv)
    p = 128
    path = args.data
    img_list = [f for f in os.listdir(path) if f[-3:] in ["jpg", "png", "peg"]]
    # img_list = img_list[:10]
    device = 'cuda:0' if args.cuda else 'cpu'

    net = TCM(config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8],
              drop_path_rate=0.0, N=128, M=320).to(device).eval()

    if args.checkpoint:
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
        net.load_state_dict(state_dict)

    PSNR, Bit_rate, MS_SSIM, total_time = 0, 0, 0, 0
    count_success = 0

    if args.real:
        net.update()

    for img_name in img_list:
        try:
            img_path = os.path.join(path, img_name)

            # Resize image to avoid OOM
            img = Image.open(img_path).convert('RGB')
            w, h = img.size
            img = img.resize((w // 2, h // 2), Image.BICUBIC)
            img_tensor = transforms.ToTensor()(img).to(device).unsqueeze(0)

            x_padded, padding = pad(img_tensor, p)

            with torch.no_grad():
                if args.cuda:
                    torch.cuda.synchronize()
                s = time.time()

                if args.real:
                    out_enc = net.compress(x_padded)
                    out_dec = net.decompress(out_enc["strings"], out_enc["shape"])
                    out_dec["x_hat"] = crop(out_dec["x_hat"], padding)
                    x_hat = out_dec["x_hat"]
                    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / (img_tensor.numel() / 3)
                    # # 儲存壓縮後的圖片（decoded reconstruction）
                    # save_dir = os.path.join(path, "compressed_output")
                    # os.makedirs(save_dir, exist_ok=True)
                    # save_path = os.path.join(save_dir, img_name)

                    # # 將 Tensor 轉為 PIL Image 並存檔
                    # to_pil = transforms.ToPILImage()
                    # recon_image = to_pil(x_hat.squeeze(0).cpu().clamp(0, 1))
                    # recon_image.save(save_path)

                else:
                    out_net = net.forward(x_padded)
                    out_net["x_hat"] = crop(out_net["x_hat"].clamp(0, 1), padding)
                    x_hat = out_net["x_hat"]
                    bpp = compute_bpp(out_net)
                    save_dir = os.path.join(path, "compressed_output")
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, img_name)
                    recon_image = transforms.ToPILImage()(x_hat.squeeze(0).cpu().clamp(0, 1))
                    recon_image.save(save_path)

                if args.cuda:
                    torch.cuda.synchronize()
                e = time.time()

            psnr = compute_psnr(img_tensor, x_hat)
            msssim = compute_msssim(img_tensor, x_hat)
            print(f'{img_name} - PSNR: {psnr:.2f}dB, MS-SSIM: {msssim:.2f}dB, Bit-rate: {bpp:.3f} bpp')
            PSNR += psnr
            MS_SSIM += msssim
            Bit_rate += bpp
            total_time += (e - s)
            count_success += 1

        except RuntimeError as e:
            print(f"Skip {img_name} due to error: {e}")
        finally:
            if args.cuda:
                torch.cuda.empty_cache()
            del img, img_tensor, x_padded, padding, x_hat  

    if count_success > 0:
        print(f'\nProcessed {count_success}/{len(img_list)} images successfully.')
        print(f'Average PSNR: {PSNR / count_success:.2f} dB')
        print(f'Average MS-SSIM: {MS_SSIM / count_success:.4f}')
        print(f'Average Bit-rate: {Bit_rate / count_success:.3f} bpp')
        print(f'Average Time per image: {1000 * total_time / count_success:.1f} ms')
    else:
        print("No images processed successfully.")


    

if __name__ == "__main__":
    print(torch.cuda.is_available())
    main(sys.argv[1:])
    