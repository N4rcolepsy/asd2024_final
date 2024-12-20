import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import logging
import torch
from scipy.sparse import csc_matrix
import numpy as np
import math
from csc_conv import * 

# 로그 설정
bit = 8
a   = 0     # 1 : activation 
            # 0 : kernel
if a==1:
    logging.basicConfig(
        filename="activation_compress_ratios.log",  # 로그 파일 이름
        filemode="a",  # "a"는 추가 모드, 기존 내용에 덧붙여 작성
        format="%(asctime)s - %(levelname)s - %(message)s",  # 로그 형식
        level=logging.INFO  # 로그 레벨 설정
    )
else :
    logging.basicConfig(
        filename="kernel_compress_ratios.log",  # 로그 파일 이름
        filemode="a",  # "a"는 추가 모드, 기존 내용에 덧붙여 작성
        format="%(asctime)s - %(levelname)s - %(message)s",  # 로그 형식
        level=logging.INFO  # 로그 레벨 설정
    )

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0]*window_size[1], C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    nwB, N, C = windows.shape
    windows = windows.view(-1, window_size[0], window_size[1], C)
    B = int(nwB / (H * W / window_size[0] / window_size[1]))
    x = windows.view(
        B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def pad_if_needed(x, size, window_size):
    n, h, w, c = size
    pad_h = math.ceil(h / window_size[0]) * window_size[0] - h
    pad_w = math.ceil(w / window_size[1]) * window_size[1] - w
    if pad_h > 0 or pad_w > 0:  # center-pad the feature on H and W axes
        img_mask = torch.zeros((1, h+pad_h, w+pad_w, 1))  # 1 H W 1
        h_slices = (
            slice(0, pad_h//2),
            slice(pad_h//2, h+pad_h//2),
            slice(h+pad_h//2, None),
        )
        w_slices = (
            slice(0, pad_w//2),
            slice(pad_w//2, w+pad_w//2),
            slice(w+pad_w//2, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(
            img_mask, window_size
        )  # nW, window_size*window_size, 1
        mask_windows = mask_windows.squeeze(-1)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(
            attn_mask != 0, float(-100.0)
        ).masked_fill(attn_mask == 0, float(0.0))
        return nn.functional.pad(
            x,
            (0, 0, pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
        ), attn_mask
    return x, None

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k."""

    with torch.no_grad():

        maxk = max(topk)
        batch_size = target.size(0)
        # topk 인덱스 얻기
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        # correct: pred와 target이 일치하는 부분을 True로 표시
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []

        for k in topk:
            # 각 k에 대해 정확도 계산
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        
        return res

def main():
    # GPU 사용 가능 시 사용
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ImageNet Validation 데이터셋 로드
    # ImageNet 디렉토리 구조: /path/to/imagenet/val/class_name/*.JPEG
    val_dir = "/home/ondevice/imagenet/val"  # 실제 데이터 경로로 수정

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # ImageNet mean/std
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=8, pin_memory=True)

    # Pretrained ResNet-50 모델 로드
    model = models.resnet50(pretrained=True)
    model = model.to(device)
    model.eval()  # 평가 모드

    # 정확도 측정을 위한 변수
    top1_correct = 0
    top5_correct = 0
    total_samples = 0

    # Hook 함수 정의
    activation_values = {}
    kernel_values = {}
    total_zero_activation_ratios = {}
    total_zero_kernel_ratios = {}
    total_compress_activation_ratios = {}
    total_compress_kernel_ratios = {}

    def get_activation(name):
        def hook(model, input, output):
            #print(input)
            #scale = (output.max()-output.min()) / 255
            w = output
            if w.dim() ==4 :
                w_min = w.amin(dim=[1, 2, 3], keepdim=True)  # 배치별 최소값
                w_max = w.amax(dim=[1, 2, 3], keepdim=True)  # 배치별 최대값
            else :
                #print(w.shape)
                w_min = w.amin(dim=[1], keepdim=True)  # 배치별 최소값
                w_max = w.amax(dim=[1], keepdim=True)  # 배치별 최대값

            # 스케일 계산
            scale = ((w_max - w_min) / (2**bit-1) )
            #.squeeze()
            zero_point = 0

            # 배치별 양자화 적용
            # quantized_output_list = []
            # for i in range(w.shape[0]):  # 배치별 양자화
            #     quantized_output = torch.quantize_per_tensor(w[i], scale=scale[i], zero_point=zero_point, dtype=torch.qint8)
            #     quantized_output_list.append( torch.dequantize(quantized_output))

            # # 결과 병합
            # activation_values[name] = torch.stack(quantized_output_list)
            quantized_output =torch.round(w*(1/scale))
            #print(quantized_output)
            activation_values[name] = quantized_output

        return hook
    
    def get_activation_i(name):
        def hook(model, input, output):
            #print(input)
            #scale = (output.max()-output.min()) / 255
            w = input[0]
            if w.dim() ==4 :
                w_min = w.amin(dim=[1, 2, 3], keepdim=True)  # 배치별 최소값
                w_max = w.amax(dim=[1, 2, 3], keepdim=True)  # 배치별 최대값
            else :
                #print(w.shape)
                w_min = w.amin(dim=[1], keepdim=True)  # 배치별 최소값
                w_max = w.amax(dim=[1], keepdim=True)  # 배치별 최대값

            # 스케일 계산
            scale = ((w_max - w_min) / (2**bit-1) )
            #.squeeze()
            zero_point = 0

            # 배치별 양자화 적용
            # quantized_output_list = []
            # for i in range(w.shape[0]):  # 배치별 양자화
            #     quantized_output = torch.quantize_per_tensor(w[i], scale=scale[i], zero_point=zero_point, dtype=torch.qint8)
            #     quantized_output_list.append( torch.dequantize(quantized_output))

            # # 결과 병합
            # activation_values[name] = torch.stack(quantized_output_list)
            quantized_output =torch.round(w*(1/scale))
            #print(quantized_output)
            activation_values[name] = quantized_output

        return hook
    

    def get_kernel(name):
        def hook(model, input, output):
            w=model.weight.data

            w_min = w.min()  # 배치별 최소값
            w_max = w.max()  # 배치별 최대값
            # 스케일 계산
            scale = ((w_max - w_min) / (2**bit-1))
            #quantized_kernel = torch.quantize_per_tensor(model.weight.data, scale=scale, zero_point=0, dtype=torch.qint8 )
            #kernel_values[name] = quantized_kernel
            quantized_kernel = torch.round(model.weight.data * 1/scale)
            #print(quantized_kernel)
            kernel_values[name] = quantized_kernel
        return hook

    # Hook 등록
    for name, layer in model.named_modules():

        if a == 1 :
            if isinstance(layer, (nn.Linear, nn.ReLU)):
                layer.register_forward_hook(get_activation(name))
        if a == 0: 
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                layer.register_forward_hook(get_kernel(name))
        
    with torch.no_grad():
        for images, targets in tqdm(val_loader):
            
            activation_values = {}
            kernel_values = {}

            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            # top-1, top-5 accuracy
            prec1, prec5 = accuracy(outputs, targets, topk=(1,5))
            top1_correct += prec1.item() * images.size(0) / 100.0
            top5_correct += prec5.item() * images.size(0) / 100.0
            total_samples += images.size(0)

    data=[]

    if a == 1:
        for name, activation in activation_values.items():
            # GPU 텐서라면 CPU로 이동
            if activation.dim() ==4:
                arr = activation.cpu().numpy()
                N, C, H, W = arr.shape
                arr = arr.reshape(N*C, H*W)
            else :
                arr = activation.cpu().numpy()
            # CSC 포맷 변환을 위한 numpy 변환            
            # CSC 포맷으로 변환
            mat_csc = csc_matrix(arr)
            # 원본 dense 크기 (float32 = 4바이트)
            original_size = arr.size             
            # CSC 포맷 크기
            ptr_bit=np.log(H*W)/8
            id_bit =np.log(H)/8

            ptr_bit=6/8
            id_bit =3/8

            csc_size = (mat_csc.data.size ) + (mat_csc.indices.size*id_bit ) + (mat_csc.indptr.size*ptr_bit ) # 8 4  4 = 16 
            
            # 압축 비율 계산
            compression_ratio = original_size / csc_size if csc_size > 0 else 1.0
            
            # zero ratio 계산
            zero_ratio = (activation < 1e-4).sum().item() / activation.numel()
            
            total_zero_activation_ratios.setdefault(name, []).append(zero_ratio)
            total_compress_activation_ratios.setdefault(name, []).append(compression_ratio)
            data.append({
                    "Layer": name,
                    "Avg Activation compress ratio": compression_ratio,
                })
            # 결과 출력 혹은 저장
            print(f"Activation - Layer: {name}, Zero Ratio: {zero_ratio:.4f}, CSC Compression Ratio: {compression_ratio:.4f}")

    elif a == 0:
        for name, kernel in kernel_values.items():
            kernel_f32 = kernel

            if kernel_f32.dim() ==4:
                arr = kernel_f32.cpu().numpy()
                N, C, H, W = arr.shape
                arr = arr.reshape(N*C, H*W)
            else :
                arr = kernel_f32.cpu().numpy()

            mat_csc = csc_matrix(arr)
            
            
            original_size = arr.size 
            csc_size = (mat_csc.data.size ) + (mat_csc.indices.size *2/8 ) + (mat_csc.indptr.size*4/8 ) # 3x3 , 2bit. 4bit
            compression_ratio = original_size / csc_size if csc_size > 0 else 1.0
            
            zero_ratio = (kernel_f32 ==0.).sum().item() / kernel_f32.numel()
            total_zero_kernel_ratios.setdefault(name, []).append(zero_ratio)
            total_compress_kernel_ratios.setdefault(name, []).append(compression_ratio)
            data.append({
                    "Layer": name,
                    "Avg kernel compress ratio": compression_ratio
                })
            print(f"Kernel - Layer: {name}, Zero Ratio: {zero_ratio:.4f}, CSC Compression Ratio: {compression_ratio:.4f}")


    #print('Total number of layers:', i + 1)

    # DataFrame 생성
    df = pd.DataFrame(data)
    # Plotting
    plt.figure(figsize=(10, 5))

    if a== 1:
        plt.plot(df["Layer"], df["Avg Activation compress ratio"], label="Avg Activation compress ratio", marker="o")
    if a==0:
        plt.plot(df["Layer"], df["Avg kernel compress ratio"], label="Avg kernel compress ratio", marker="x")

    # Adding labels, title, and formatting
    plt.xlabel("Layer", fontsize=12)
    plt.ylabel("Compress-Ratio X", fontsize=12)
    if a==1:
        plt.title(" Activation compres ratio per ResNet-50 Layer (imagenet)", fontsize=14)
    else:
        plt.title(" Kernel compress ratio per ResNet-50 Layer (imagenet)", fontsize=14)

    plt.xticks(rotation=90, fontsize=6)
    plt.ylim(0, 4)  # Y축 범위 설정
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.7)

    # Show the plot
    plt.tight_layout()
    if a==1:
        plt.savefig("resnet50_compress_ratios_act.png", dpi=300, bbox_inches="tight")  # 파일명, 해상도, 여백 설정
    if a==0:
        plt.savefig("resnet50_compress_ratios_kernel.png", dpi=300, bbox_inches="tight")  # 파일명, 해상도, 여백 설정

    plt.show()

    top1_acc = (top1_correct / total_samples) * 100.0
    top5_acc = (top5_correct / total_samples) * 100.0

    if a==1:
        for entry in data:
            logging.info(f"Layer: {entry['Layer']}, Avg Activation compress ratio: {entry['Avg Activation compress ratio']:.4f}")
    else:
        for entry in data:
            print(entry)
            logging.info(f"Layer: {entry['Layer']}, Avg kernel compress ratio: {entry['Avg kernel compress ratio']:.4f}")


    print(f'Top-1 Accuracy: {top1_acc:.2f}%')
    print(f'Top-5 Accuracy: {top5_acc:.2f}%')

if __name__ == "__main__":
    main()