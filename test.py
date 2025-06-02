from qhtt import giai_tu_dong

# Ví dụ dữ liệu giả lập
loai_bt = 'max'
c_original = [-1, -3]  # hàm mục tiêu: 3x1 + 5x2
A_original = [
    [1, 0],
    [0, 2],
    [3, 2]
]
b_original = [4, 12, 18]
rls = ['<=', '<=', '<=']
n_original_vars = 2
var_types = ['>=0', '>=0']

# Gọi hàm giải tự động
ket_qua = giai_tu_dong(loai_bt, c_original, A_original, b_original, rls, n_original_vars, var_types, phuong_phap='2')

print(ket_qua)
