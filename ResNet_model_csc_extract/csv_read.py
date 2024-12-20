import pandas as pd
import numpy as np

# CSV 파일 불러오기
csv_file = "/home/ondevice/activation_compress_ratios.csv"
act_comp_ratio = pd.read_csv(csv_file)

# NaN 값 제거 및 Layer id 정리
act_comp_ratio = act_comp_ratio.dropna(subset=['Layer id'])
act_comp_ratio['Layer id'] = act_comp_ratio['Layer id'].astype(str).str.strip().astype(int)

# i_bw 값 가져오기
bws = [10]  # user_dram_bandwidth 예제 값

for self_layer_id in range(54):
    if self_layer_id in act_comp_ratio['Layer id'].values:
        i_bw = act_comp_ratio.loc[act_comp_ratio['Layer id'] == self_layer_id, 'i_bw'].values[0]
        ifmap_backing_bw = np.floor(bws[0] * i_bw).astype(int)
        print("i_bw:", i_bw)
        print("ifmap_backing_bw:", ifmap_backing_bw)
    else:
        print(f"Layer id {self_layer_id} not found in CSV!")
