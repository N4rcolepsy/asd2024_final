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

# 로그 설정
bit = 8
a   = 1     # 1 : activation 
            # 0 : kernel
if a==1:
    logging.basicConfig(
        filename="activation_zero_ratios.log",  # 로그 파일 이름
        filemode="a",  # "a"는 추가 모드, 기존 내용에 덧붙여 작성
        format="%(asctime)s - %(levelname)s - %(message)s",  # 로그 형식
        level=logging.INFO  # 로그 레벨 설정
    )
else :
    logging.basicConfig(
        filename="kernel_zero_ratios.log",  # 로그 파일 이름
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
    print(model)
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
            w =torch.round(w*(1/scale))
            #print(quantized_output)
            #activation_values[name] = quantized_output
            activation_values[name] = w

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


            if a ==1: 
                for name, activation in activation_values.items():
                    activation_f32=activation
                    zero_ratio = (activation_f32 == 0.).sum().item() / activation_f32.numel()
                    total_zero_activation_ratios.setdefault(name, []).append(zero_ratio)
            if a==0:
                for name, kernel in kernel_values.items():
                    kernel_f32 = torch.dequantize(kernel)
                    zero_ratio = (kernel_f32 < 1e-8).sum().item() / kernel_f32.numel()
                    total_zero_kernel_ratios.setdefault(name, []).append(zero_ratio)

    data=[]

    total_zeros=0
    if a== 1:
        for i, name in enumerate(total_zero_activation_ratios):
            # 평균 계산
            if a == 1 :
                avg_activation_zero_ratio = sum(total_zero_activation_ratios[name]) / len(total_zero_activation_ratios[name])
            # 딕셔너리 형태로 리스트에 추가
            if a ==1:
                print(f"Layer: {name}, Avg Activation 0 ratio: {avg_activation_zero_ratio:.4f}")
                data.append({
                    "Layer": name,
                    "Avg Activation 0 ratio": avg_activation_zero_ratio,
                })
            total_zeros+=avg_activation_zero_ratio
        print('avg zero',total_zeros/i)

    else :
        for i, name in enumerate(total_zero_kernel_ratios):
            # 평균 계산
     
            if a== 0:
                avg_kernel_zero_ratio = sum(total_zero_kernel_ratios[name]) / len(total_zero_kernel_ratios[name])
            
            # 데이터 출력
            
            if a ==0:
                print(f"Layer: {name}, Avg Kernel 0 ratio: {avg_kernel_zero_ratio:.4f}")
                data.append({
                    "Layer": name,
                    "Avg Kernel 0 ratio": avg_kernel_zero_ratio
                })

    print('Total number of layers:', i + 1)

    # DataFrame 생성
    df = pd.DataFrame(data)
    # Plotting
    plt.figure(figsize=(10, 5))

    if a== 1:
        plt.plot(df["Layer"], df["Avg Activation 0 ratio"], label="Avg Activation 0 ratio", marker="o")
    if a==0:
        plt.plot(df["Layer"], df["Avg Kernel 0 ratio"], label="Avg Kernel 0 ratio", marker="x")

    # Adding labels, title, and formatting
    plt.xlabel("Layer", fontsize=12)
    plt.ylabel("Zero-Ratio", fontsize=12)
    if a==1:
        plt.title("Activation  per ResNet-50 Layer (imagenet)", fontsize=14)
    else:
        plt.title(" Kernel Zero-Ratio per ResNet-50 Layer (imagenet)", fontsize=14)

    plt.xticks(rotation=90, fontsize=6)
    plt.ylim(0, 1)  # Y축 범위 설정
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.7)

    # Show the plot
    plt.tight_layout()
    if a==1:
        plt.savefig("resnet50_zero_ratios_act.png", dpi=300, bbox_inches="tight")  # 파일명, 해상도, 여백 설정
    if a==0:
        plt.savefig("resnet50_zero_ratios_kernel.png", dpi=300, bbox_inches="tight")  # 파일명, 해상도, 여백 설정

    plt.show()

    top1_acc = (top1_correct / total_samples) * 100.0
    top5_acc = (top5_correct / total_samples) * 100.0

    if a==1:
        for entry in data:
            logging.info(f"Layer: {entry['Layer']}, Avg Activation 0 ratio: {entry['Avg Activation 0 ratio']:.4f}")
    else:
        for entry in data:
            print(entry)
            logging.info(f"Layer: {entry['Layer']}, Avg kernel 0 ratio: {entry['Avg Kernel 0 ratio']:.4f}")


    print(f'Top-1 Accuracy: {top1_acc:.2f}%')
    print(f'Top-5 Accuracy: {top5_acc:.2f}%')

if __name__ == "__main__":
    main()