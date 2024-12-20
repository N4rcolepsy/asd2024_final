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
a   = 1     # 1 : no csc apply
            # 0 : csc apply
flops=0

if a==1:
    logging.basicConfig(
        filename="flops.log",  # 로그 파일 이름
        filemode="a",  # "a"는 추가 모드, 기존 내용에 덧붙여 작성
        format="%(asctime)s - %(levelname)s - %(message)s",  # 로그 형식
        level=logging.INFO  # 로그 레벨 설정
    )
else :
    logging.basicConfig(
        filename="csc_flops.log",  # 로그 파일 이름
        filemode="a",  # "a"는 추가 모드, 기존 내용에 덧붙여 작성
        format="%(asctime)s - %(levelname)s - %(message)s",  # 로그 형식
        level=logging.INFO  # 로그 레벨 설정
    )

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
    total_flops={}
    total_latency={}
    flops_values={}
    latency_values={}


    def get_flops(name):
        def hook(model,input,output):
            w = model.weight
            act = output
            if w.dim()==4:
                #print(input[0].shape)
                B,N,H,W=input[0].shape
                B,M,E,F=output.shape
                N,M,K1,K2= w.shape
                flops= N*M*K1*K2*E*F
                latency= N*H*W
            else :
                B,C= input[0].shape
                N,M= w.shape
                flops = N*M  # E X C * C * F
                latency= C
            flops_values[name] = flops
            latency_values[name] = latency
            #print(name,flops)
        return hook




    # Hook 등록
    for name, layer in model.named_modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                layer.register_forward_hook(get_flops(name))
        
    with torch.no_grad():
        for i, (images, targets) in enumerate(tqdm(val_loader)):
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

            if a ==1: 
                for name, flop in flops_values.items():
                    total_flops.setdefault(name, []).append(flop)
                for name, latency in latency_values.items():
                    total_latency.setdefault(name, []).append(latency)
            if a==0:
                for name, kernel in kernel_values.items():
                    kernel_f32 = torch.dequantize(kernel)
                    zero_ratio = (kernel_f32 < 1e-8).sum().item() / kernel_f32.numel()
                    total_zero_kernel_ratios.setdefault(name, []).append(zero_ratio)



    data=[]

    total_gflops=0

    if a == 1:
        for name in total_flops :
            #print('name',name)
            #print('total_flops_len',len(total_flops[name]))
            gflops=  (sum(total_flops[name])/ len(total_flops[name])  ) / 1e+9
            latency= (sum(total_latency[name])/ len(total_latency[name])  ) 
            total_gflops+=gflops

            data.append({
                    "Layer": name,
                    "Avg GFlops": gflops,
                    "Avg latency": latency,
                })
            # 결과 출력 혹은 저장
            print(f"Activation - Layer: {name},  Gflops: {gflops:.4f}, latency : {latency},  CSC  Gflops: {gflops:.4f}")
            print('total_gflops',total_gflops)
    elif a == 0:
        for name, kernel in kernel_values.items():
            # kernel을 float32 CPU 텐서로 변환
            kernel_f32 = kernel
            #torch.dequantize(kernel).cpu()
            # CSC 포맷 변환
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
            
            zero_ratio = (kernel_f32 < 1e-4).sum().item() / kernel_f32.numel()
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
        plt.plot(df["Layer"], df["Avg GFlops"], df["Avg latency"], label="Avg Avg GFlops", marker="o")
    if a==0:
        plt.plot(df["Layer"], df["Avg kernel compress ratio"], label="Avg Avg GFlops", marker="x")

    # Adding labels, title, and formatting
    plt.xlabel("Layer", fontsize=12)
    plt.ylabel("Avg GFlops", fontsize=12)
    if a==1:
        plt.title(" Avg GFlops per ResNet-50 Layer (imagenet)", fontsize=14)
    else:
        plt.title(" Kernel compress ratio per ResNet-50 Layer (imagenet)", fontsize=14)

    plt.xticks(rotation=90, fontsize=6)
    plt.ylim(0, 4)  # Y축 범위 설정
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.7)

    # Show the plot
    plt.tight_layout()
    if a==1:
        plt.savefig("resnet50_Avg GFlops.png", dpi=300, bbox_inches="tight")  # 파일명, 해상도, 여백 설정
    if a==0:
        plt.savefig("resnet50_compress_ratios_kernel.png", dpi=300, bbox_inches="tight")  # 파일명, 해상도, 여백 설정

    plt.show()

    top1_acc = (top1_correct / total_samples) * 100.0
    top5_acc = (top5_correct / total_samples) * 100.0

    if a==1:
        for entry in data:
            logging.info(f"Layer: {entry['Layer']}, Avg Avg GFlops: {entry['Avg GFlops']:.4f},Avg Avg latency: {entry['Avg latency']}")
    else:
        for entry in data:
            print(entry)
            logging.info(f"Layer: {entry['Layer']}, Avg kernel compress ratio: {entry['Avg kernel compress ratio']:.4f}")


    print(f'Top-1 Accuracy: {top1_acc:.2f}%')
    print(f'Top-5 Accuracy: {top5_acc:.2f}%')

if __name__ == "__main__":
    main()