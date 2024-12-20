import numpy as np

def csc_convolution(input_csc_values, input_row_ind, input_col_ptr, 
                    filter_matrix, input_shape, filter_shape, stride=1):
    """
    CSC 형식으로 희소 행렬 Convolution과 FLOPs 계산

    Args:
        input_csc_values: 입력 행렬의 비영 요소 값
        input_row_ind: 입력 행렬의 행 인덱스 (CSC)
        input_col_ptr: 입력 행렬의 열 포인터 (CSC)
        filter_matrix: 필터 (Dense 형태)
        input_shape: 입력 행렬의 크기 (행, 열)
        filter_shape: 필터의 크기 (행, 열)
        stride: 슬라이딩 윈도우의 보폭 크기

    Returns:
        output: 출력 행렬 (Dense 형태)
        flops: 총 FLOPs 수
    """
    input_rows, input_cols = input_shape
    filter_rows, filter_cols = filter_shape
    
    # 출력 행렬 크기 계산
    output_rows = (input_rows - filter_rows) // stride + 1
    output_cols = (input_cols - filter_cols) // stride + 1
    output = np.zeros((output_rows, output_cols))
    
    flops = 0  # FLOPs 계산
    
    # 슬라이딩 윈도우로 Convolution 수행
    for out_row in range(output_rows):
        for out_col in range(output_cols):
            # 입력 데이터의 시작점 계산
            input_row_start = out_row * stride
            input_col_start = out_col * stride

            # 필터와 입력 값 곱셈 및 덧셈
            for f_row in range(filter_rows):
                for f_col in range(filter_cols):
                    input_row = input_row_start + f_row
                    input_col = input_col_start + f_col
                    
                    # CSC 형식에서 입력 값 가져오기
                    for idx in range(input_col_ptr[input_col], input_col_ptr[input_col + 1]):
                        if input_row == input_row_ind[idx]:
                            input_value = input_csc_values[idx]
                            filter_value = filter_matrix[f_row, f_col]
                            output[out_row, out_col] += input_value * filter_value
                            flops += 1  # 곱셈
                            if f_row > 0 or f_col > 0:
                                flops += 1  # 덧셈
    return flops
