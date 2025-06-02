import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import combinations
from shapely.geometry import Polygon, Point, LineString
from scipy.optimize import linprog

def nhap_bai_toan():
    """
    Hàm này cho phép người dùng nhập các thông số của bài toán quy hoạch tuyến tính.
    Bao gồm số biến, số ràng buộc, loại bài toán (tối đa/tối thiểu), hệ số hàm mục tiêu và các ràng buộc.
    """
    print("=== Nhập bài toán quy hoạch tuyến tính ===")

    while True:
        try:
            n = int(input("Nhập số biến (n, phải >= 2): "))
            if n < 2:
                print("Số biến phải từ 2 trở lên để có thể giải. Vui lòng nhập lại.")
            else:
                break
        except ValueError:
            print("Đầu vào không hợp lệ. Vui lòng nhập một số nguyên.")

    while True:
        try:
            m = int(input("Nhập số ràng buộc (m): "))
            if m <= 0:
                print("Số ràng buộc phải là một số nguyên dương. Vui lòng nhập lại.")
            else:
                break
        except ValueError:
            print("Lỗi: Số ràng buộc phải là một số nguyên. Vui lòng nhập lại.")

    loai_bt = input("Bài toán là TỐI ĐA (max) hay TỐ THIỂU (min)? [max/min]: ").strip().lower()
    while loai_bt not in ['max', 'min']:
        loai_bt = input("Vui lòng nhập lại (max/min): ").strip().lower()

    print("\nNhập hệ số hàm mục tiêu Z = c1*x1 + c2*x2 + ...:")
    while True:
        c_str = input(f"Hệ số c (cách nhau bởi dấu cách, có {n} hệ số): ").split()
        if len(c_str) != n:
            print(f"Phải nhập đúng {n} hệ số. Vui lòng nhập lại.")
        else:
            try:
                c = list(map(float, c_str))
                break
            except ValueError:
                print("Hệ số phải là số. Vui lòng nhập lại.")

    A = []
    b = []
    rls = []  # Dấu ràng buộc: '<=', '>=', '='

    print("\nNhập từng ràng buộc dạng: a1 a2 ... [dấu] b")
    print("Ví dụ: x1 + 0x2 +2x3 <= 10")
    print("Nhập là: 1 0 2 <= 10 (phải nhập đúng {n} hệ số a)\n")

    for i in range(m):
        while True:
            parts = input(f"Ràng buộc {i+1}: ").split()
            if len(parts) != n + 2:
                print(f"Phải nhập đúng {n} hệ số, [dấu] và 1 số b. Chú ý có khoảng cách. Vui lòng nhập lại.")
                continue
            *a_coeffs_str, op, b_val_str = parts
            if op not in ['<=', '>=', '=']:
                print("Dấu ràng buộc phải là '<=', '>=', hoặc '='. Nhập lại.")
                continue
            try:
                a_coeffs = list(map(float, a_coeffs_str))
                b_val = float(b_val_str)
                break
            except ValueError:
                print("Hệ số và b phải là số. Nhập lại.")

        A.append(a_coeffs)
        rls.append(op)
        b.append(b_val)

    # Nhập ràng buộc dấu cho từng xj
    var_types = []
    print("\nNhập ràng buộc dấu cho từng xj:")
    print(" - Nhập là: '>=' cho xj >= 0")
    print(" - Nhập là: '<=' cho xj <= 0")
    print(" - Nhập là: 'free' cho xj tự do")
    for i in range(n):
        while True:
            var_type = input(f"Ràng buộc dấu của x{i+1} [>=/<=/free]: ").strip().lower()
            if var_type in ['>=', '<=', 'free']:
                var_types.append(var_type)
                break
            else:
                print("Không hợp lệ. Vui lòng nhập: '<=', '>=', hoặc 'free'. Nhập lại.")

    return loai_bt, np.array(c), np.array(A, dtype=float), np.array(b, dtype=float), rls, n, var_types

def chuyen_ve_dang_chuan(loai_bt, c, A, b, rls, var_types):
    """
    Chuyển bài toán quy hoạch tuyến tính về dạng chuẩn (min C^T x , Ax <= b, x >= 0).
    Nếu là bài toán max, đổi dấu hàm mục tiêu.
    Nếu có ràng buộc '>=', đổi dấu cả hàng.
    Nếu có ràng buộc '=', tách thành hai ràng buộc '<=' và '>='.
    Xử lý biến tự do và biến không dương.
    """
    print("******************************************************")

    original_n_vars = c.shape[0]

    # Bước 1: Xử lý hàm mục tiêu (chuyển về min)
    if loai_bt == 'max':
        c_std = -c  # Chuyển bài toán max Z thành min -Z
    else:
        c_std = np.copy(c) # Tạo bản sao để không thay đổi c gốc

    # Bước 2: Xử lý biến tự do và biến không dương
    new_n_vars = 0
    variable_transformations = []
    standardized_var_names = [] # Danh sách các tên biến mới sau khi chuẩn hóa

    # Tính toán số biến mới sau khi chuẩn hóa
    for i in range(original_n_vars):
        if var_types[i] == 'free':
            new_n_vars += 2
        else: # '>= 0' hoặc '<= 0'
            new_n_vars += 1

    # Khởi tạo ma trận A_new và vector c_new với kích thước mới
    c_new = np.zeros(new_n_vars)
    A_new = np.zeros((A.shape[0], new_n_vars))

    current_original_col = 0 # Chỉ số cột trong ma trận A gốc
    current_new_col = 0      # Chỉ số cột trong ma trận A_new

    for i in range(original_n_vars):
        if var_types[i] == 'free':
            # x_i = x_i^+ - x_i^- với x_i^+, x_i^- >= 0
            variable_transformations.append(f"x{i+1} = x{i+1}^+ - x{i+1}^-")
            standardized_var_names.append(f"x{i+1}^+")
            standardized_var_names.append(f"x{i+1}^-")

            c_new[current_new_col] = c_std[current_original_col]
            c_new[current_new_col + 1] = -c_std[current_original_col]
            A_new[:, current_new_col] = A[:, current_original_col]
            A_new[:, current_new_col + 1] = -A[:, current_original_col]

            current_new_col += 2
        elif var_types[i] == '<=': # Lưu ý: nhập từ người dùng là '<=', không phải '<= 0'
            # y_i = -x_i với y_i >= 0 => x_i = -y_i
            variable_transformations.append(f"y{i+1} = -x{i+1}")
            standardized_var_names.append(f"y{i+1}")

            c_new[current_new_col] = -c_std[current_original_col]
            A_new[:, current_new_col] = -A[:, current_original_col]

            current_new_col += 1
        else: # '>=' (giữ nguyên x_i >= 0)
            variable_transformations.append(f"x{i+1} (giữ nguyên)")
            standardized_var_names.append(f"x{i+1}")

            c_new[current_new_col] = c_std[current_original_col]
            A_new[:, current_new_col] = A[:, current_original_col]
            
            current_new_col += 1
        current_original_col += 1 # Tăng chỉ số biến gốc sau mỗi lần xử lý

    # In ra các phép thế biến
    if variable_transformations:
        print("\nPhép thế biến:")
        for trans in variable_transformations:
            print(f" Đặt {trans}")

    c_std = c_new
    A_std = A_new

    # Bước 3: Xử lý các ràng buộc
    A_final_list = []
    b_final_list = []

    for i in range(len(rls)):
        if rls[i] == '<=':
            A_final_list.append(A_std[i])
            b_final_list.append(b[i])
        elif rls[i] == '>=':  # Đổi dấu để thành <=
            A_final_list.append(-A_std[i])
            b_final_list.append(-b[i])
        elif rls[i] == '=':   # Tách thành 2 ràng buộc <= và >=
            A_final_list.append(A_std[i])
            b_final_list.append(b[i])
            A_final_list.append(-A_std[i])
            b_final_list.append(-b[i])
        else:
            raise ValueError(f"Dấu ràng buộc không hợp lệ: {rls[i]}")

    A_std_final = np.array(A_final_list, dtype=float)
    b_std_final = np.array(b_final_list, dtype=float)

    # In bài toán đã chuyển về dạng chuẩn
    print("---- Bài toán sau khi đã chuyển về dạng chuẩn ----")

    z_str = "min Z = "
    for i in range(len(c_std)):
        if abs(c_std[i]) > 1e-10: # Kiểm tra giá trị khác 0 đáng kể
            sign = "+" if c_std[i] > 0 else "-"
            if i == 0: # Số hạng đầu tiên không cần dấu '+' nếu nó dương
                sign = "-" if c_std[i] < 0 else ""
            z_str += f"{sign} {abs(c_std[i]):.4f}{standardized_var_names[i]}"

    print(z_str.replace("+-", "- ").replace("++", "+ ").strip()) # Clean up double signs

    print("\nRàng buộc:")
    for i in range(A_std_final.shape[0]):
        constraint_str = ""
        for j in range(A_std_final.shape[1]):
            coeff = A_std_final[i, j]
            if abs(coeff) > 1e-10: # Kiểm tra giá trị khác 0 đáng kể
                sign = "+" if coeff > 0 else "-"
                if j == 0: # Số hạng đầu tiên không cần dấu '+' nếu nó dương
                    sign = "-" if coeff < 0 else ""
                constraint_str += f"{sign} {abs(coeff):.4f}{standardized_var_names[j]}"

        constraint_str = constraint_str.strip().replace("+-", "- ")
        if constraint_str.startswith("+"):
            constraint_str = constraint_str[1:].strip()
        print(f"{i+1}. {constraint_str} <= {b_std_final[i]:.4f}")

    print("\nRàng buộc dấu:")
    for i in range(new_n_vars):
        print(f"{standardized_var_names[i]} >= 0")

    return c_std, A_std_final, b_std_final, new_n_vars, standardized_var_names

def xet_phuong_phap(n, b_std):
    """
    Xác định các phương pháp giải phù hợp dựa trên số biến và giá trị b_std,
    sau đó cho phép người dùng lựa chọn. Phương pháp hình học chỉ áp dụng cho 2 biến.
    """
    # Định nghĩa các phương pháp:
    # 1: PP hình học (chỉ giải cho trường hợp 2 biến)
    # 2: PP đơn hình
    # 3: PP Bland
    # 4: PP hai pha

    print("------ Lựa chọn phương pháp ------")

    if n == 2:
        if np.any(b_std < -1e-9): # Nếu có b_i âm sau chuẩn hóa
            print("\nCác phương pháp gợi ý cho bài toán 2 biến và có b < 0:")
            print(" 1 - Phương pháp hình học")
            print(" 4 - Phương pháp hai pha")
            valid_choices = [1, 4]
        elif np.any(np.abs(b_std) < 1e-9): # Nếu có b_i bằng 0 sau chuẩn hóa
            print("\nCác phương pháp gợi ý cho bài toán 2 biến và có b = 0:")
            print(" 1 - Phương pháp hình học")
            print(" 3 - Phương pháp Bland")
            valid_choices = [1, 3]
        else: # Các trường hợp còn lại (b_std > 0)
            print("\nCác phương pháp gợi ý cho bài toán 2 biến và b > 0:")
            print(" 1 - Phương pháp hình học")
            print(" 2 - Phương pháp đơn hình")
            print(" 3 - Phương pháp Bland")
            valid_choices = [1, 2, 3]
    else: # n > 2
        if np.any(b_std < -1e-9): # Nếu có b_i âm sau chuẩn hóa
            print("\nVới nhiều hơn 2 biến và có b < 0, phương pháp phù hợp nhất là:")
            print(" 4 - Phương pháp hai pha")
            valid_choices = [4]
        elif np.any(np.abs(b_std) < 1e-9): # Nếu có b_i bằng 0 sau chuẩn hóa
            print("\nVới nhiều hơn 2 biến và có b = 0, các phương pháp gợi ý:")
            print(" 3 - Phương pháp Bland")
            valid_choices = [3]
        else: # Các trường hợp còn lại (n > 2, b_std > 0)
            print("\nVới nhiều hơn 2 biến và b > 0, các phương pháp gợi ý:")
            print(" 2 - Phương pháp đơn hình")
            print(" 3 - Phương pháp Bland")
            valid_choices = [2, 3]

    # Vòng lặp để người dùng nhập lựa chọn hợp lệ
    while True:
        try:
            choice = int(input(f"Vui lòng nhập lựa chọn của bạn {valid_choices}: "))
            if choice in valid_choices:
                return choice
            else:
                print(f"Lựa chọn không hợp lệ. Vui lòng nhập một trong các số sau: {valid_choices}")
        except ValueError:
            print("Đầu vào không hợp lệ. Vui lòng nhập một số nguyên.")

def khoi_tao_bang_tu_vung(c_std, A_std, b_std):
    """
    Khởi tạo bảng từ vựng ban đầu.
    Thêm các biến bù vào ma trận A và tạo hàng mục tiêu (hàng 0).
    """
    m_std, n_original_vars = A_std.shape

    I = np.eye(m_std)

    B = np.zeros((m_std + 1, 1 + n_original_vars + m_std))

    B[0, 0] = 0.0
    B[0, 1:1 + n_original_vars] = c_std

    B[1:, 0] = b_std
    B[1:, 1:1 + n_original_vars] = A_std
    B[1:, 1 + n_original_vars:] = I

    co_so = [n_original_vars + i for i in range(m_std)]

    return B, co_so

def in_bang_tu_vung(B, bien_names, co_so, bien_vao_col_idx, pivot_count, n_standardized_vars, is_optimal_tableau=False):
    """
    In bảng từ vựng ra màn hình.
    Hiển thị rõ ràng các biến cơ sở, biến không cơ sở, hệ số và tỉ lệ.
    """
    m_std = B.shape[0] - 1

    if is_optimal_tableau:
        print("\n--- Từ vựng tối ưu ---")
    else:
        print(f"\n--- Bảng từ vựng {pivot_count} ---")

    column_headers = bien_names[:n_standardized_vars] + \
                     [f"w{i+1}" for i in range(m_std)] + [' ']

    header_line = " " * 13
    for h in column_headers:
        header_line += f"{h:>10}"
    print(header_line)
    print("-" * (13 + 10 * len(column_headers)))

    line = f"{'z':<12}|"
    for j in range(1, B.shape[1]):
        line += f"{B[0, j]:10.3f}"
    line += f"{B[0, 0]:10.3f}"
    print(line)
    print("-" * (13 + 10 * len(column_headers)))

    ti_le_strs = []
    if bien_vao_col_idx != -1:
        for i in range(1, m_std + 1):
            a_ij = B[i, bien_vao_col_idx]
            if a_ij > 1e-10:
                ratio = B[i, 0] / a_ij
                ti_le_strs.append(f"({B[i,0]:.3f}/{a_ij:.3f}={ratio:.3f})")
            else:
                ti_le_strs.append("   ")
    else:
        ti_le_strs = [""] * m_std

    for i in range(1, m_std + 1):
        var_idx_in_basis = co_so[i-1]
        row_name = bien_names[var_idx_in_basis]

        line = f"{row_name:<12}|"
        for j in range(1, B.shape[1]):
            line += f"{B[i, j]:10.3f}"
        line += f"{B[i, 0]:10.3f}     | {ti_le_strs[i-1]}"
        print(line)
    print("-" * (13 + 10 * len(column_headers)))

def hinh_hoc(A, b, c, loai, rls, var_types, standardized_var_names):
    """
    Giải bài toán quy hoạch tuyến tính 2 biến bằng phương pháp hình học.

    Args:
        A (np.array): Ma trận hệ số của các ràng buộc.
        b (np.array): Vector vế phải của các ràng buộc.
        c (np.array): Vector hệ số hàm mục tiêu.
        loai (str): 'max' hoặc 'min' (tối đa hoặc tối thiểu).
        rls (list): Danh sách các dấu ràng buộc ('<=', '>=', '=').
        var_types (list): Ràng buộc dấu của các biến ('free', '>=', '<=').
        standardized_var_names (list): Tên các biến đã chuẩn hóa.
    """

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    n_vars = A.shape[1]

    if n_vars != 2:
        print("Phương pháp hình học chỉ áp dụng cho bài toán 2 biến.")
        return None, None

    # --- Điều chỉnh phạm vi vẽ đồ thị ban đầu dựa trên var_types ---
    x_min_plot = -15 # Mặc định cho biến tự do hoặc <= 0
    x_max_plot = 15
    y_min_plot = -15 # Mặc định cho biến tự do hoặc <= 0
    y_max_plot = 15

    if var_types[0] == '>=':
        x_min_plot = 0
    elif var_types[0] == '<=':
        x_max_plot = 0 # x1 <= 0, nên max x1 là 0

    if var_types[1] == '>=':
        y_min_plot = 0
    elif var_types[1] == '<=':
        y_max_plot = 0 # x2 <= 0, nên max x2 là 0

    # Điều chỉnh giới hạn plot để bao phủ khu vực có thể có nghiệm
    # Có thể cần mở rộng thêm nếu các ràng buộc đẩy miền khả thi ra xa gốc
    x_min, x_max = x_min_plot, x_max_plot
    y_min, y_max = y_min_plot, y_max_plot
    # Cung cấp một khoảng đệm lớn hơn cho các đường thẳng
    plot_range_buffer = 100

    # Tìm các điểm giao của tất cả các đường thẳng
    all_intersection_points = []
    lines = []

    # Tạo các đối tượng LineString từ các ràng buộc để dễ dàng tìm giao điểm
    for i in range(len(A)):
        line_a = A[i]
        line_b = b[i]

        # Kiểm tra để tránh chia cho 0
        if line_a[1] != 0: # a2 != 0, đường thẳng không song song với trục y
            line = LineString([(x_val, (line_b - line_a[0]*x_val) / line_a[1]) for x_val in [-plot_range_buffer, plot_range_buffer]])
            lines.append(line)
        elif line_a[0] != 0: # a1 != 0 và a2 = 0, đường thẳng song song với trục y (x = const)
            line = LineString([(line_b / line_a[0], y_val) for y_val in [-plot_range_buffer, plot_range_buffer]])
            lines.append(line)
        # Nếu cả a1 và a2 đều bằng 0, đây không phải là đường thẳng mà là ràng buộc dạng 0 <= b (đã được xử lý ở chuyen_ve_dang_chuan)

    # Thêm các trục tọa độ (x=0, y=0) vào danh sách các đường thẳng để tìm giao điểm
    # Chỉ thêm nếu biến không bị ràng buộc dấu >= 0
    if var_types[0] != '>=': # Nếu x1 không phải >= 0, thì trục y (x1=0) là một đường ràng buộc tiềm năng
        lines.append(LineString([(0, -plot_range_buffer), (0, plot_range_buffer)]))
    if var_types[1] != '>=': # Nếu x2 không phải >= 0, thì trục x (x2=0) là một đường ràng buộc tiềm năng
        lines.append(LineString([(-plot_range_buffer, 0), (plot_range_buffer, 0)]))

    # Tìm giao điểm giữa các cặp đường thẳng
    for i, line1 in enumerate(lines):
        for j, line2 in enumerate(lines):
            if i < j:
                intersection = line1.intersection(line2)
                if intersection.geom_type == 'Point':
                    all_intersection_points.append([intersection.x, intersection.y])

    # Lọc các điểm khả thi
    feasible_points = []
    for point_coords in all_intersection_points:
        point = np.array(point_coords)
        is_feasible = True
        # Kiểm tra từng ràng buộc gốc
        for k in range(len(A)):
            val = np.dot(A[k], point)
            if rls[k] == '<=' and val > b[k] + 1e-7:
                is_feasible = False
                break
            elif rls[k] == '>=' and val < b[k] - 1e-7:
                is_feasible = False
                break
            elif rls[k] == '=' and abs(val - b[k]) > 1e-7:
                is_feasible = False
                break

        # --- Kiểm tra ràng buộc dấu của biến (var_types) ---
        if var_types[0] == '>=' and point[0] < -1e-7: # x1 >= 0
            is_feasible = False
        elif var_types[0] == '<=' and point[0] > 1e-7: # x1 <= 0
            is_feasible = False
        # else: 'free', không cần kiểm tra

        if var_types[1] == '>=' and point[1] < -1e-7: # x2 >= 0
            is_feasible = False
        elif var_types[1] == '<=' and point[1] > 1e-7: # x2 <= 0
            is_feasible = False
        # else: 'free', không cần kiểm tra


        if is_feasible:
            feasible_points.append(point)

    if not feasible_points:
        print("\n⇒ Bài toán vô nghiệm.")
        plt.figure(figsize=(8, 8))
        colors = matplotlib.colormaps['tab10'].resampled(len(A))
        for i in range(len(A)):
            x_vals_plot = np.linspace(x_min, x_max, 400)
            if A[i][1] != 0:
                y_vals_plot = (b[i] - A[i][0] * x_vals_plot) / A[i][1]
                plt.plot(x_vals_plot, y_vals_plot, label=f'{A[i][0]:.2f}x1 + {A[i][1]:.2f}x2 {rls[i]} {b[i]:.2f}', linestyle='--', color=colors(i))
            elif A[i][0] != 0:
                plt.axvline(x=b[i] / A[i][0], label=f'{A[i][0]:.2f}x1 {rls[i]} {b[i]:.2f}', linestyle='--', color=colors(i))
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.axvline(0, color='black', linewidth=0.8)
        plt.axhline(0, color='black', linewidth=0.8)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title("Bài toán vô nghiệm: Không có miền khả thi")
        plt.legend()
        plt.grid(True)
        plt.show()
        return None, None

    poly = None
    try:
        unique_feasible_points = []
        seen_points = set()
        for p in feasible_points:
            p_tuple = tuple(np.round(p, 6))
            if p_tuple not in seen_points:
                unique_feasible_points.append(p)
                seen_points.add(p_tuple)

        if len(unique_feasible_points) >= 3:
            poly = Polygon(unique_feasible_points).convex_hull
            if not poly.is_valid or poly.area < 1e-9:
                poly = None
        elif len(unique_feasible_points) == 1:
            poly = Point(unique_feasible_points[0]).buffer(1e-6)
        elif len(unique_feasible_points) == 2:
            poly = LineString(unique_feasible_points).buffer(1e-6)

    except Exception as e:
        print(f"Lỗi khi cố gắng tạo miền khả thi dưới dạng đa giác: {e}")
        poly = None

    # Kiểm tra không giới nội bằng linprog
    A_ub_lp, b_ub_lp, A_eq_lp, b_eq_lp = [], [], [], []
    for i in range(len(A)):
        if rls[i] == '<=':
            A_ub_lp.append(A[i])
            b_ub_lp.append(b[i])
        elif rls[i] == '>=':
            A_ub_lp.append(-A[i])
            b_ub_lp.append(-b[i])
        elif rls[i] == '=':
            A_eq_lp.append(A[i])
            b_eq_lp.append(b[i])

    # --- Điều chỉnh bounds cho linprog dựa trên var_types ---
    bounds_lp = []
    for i in range(n_vars):
        if var_types[i] == '>=':
            bounds_lp.append((0, None))
        elif var_types[i] == '<=':
            bounds_lp.append((None, 0))
        else: # 'free'
            bounds_lp.append((None, None))

    res = linprog(
        c=-c if loai == 'max' else c,
        A_ub=np.array(A_ub_lp) if A_ub_lp else None,
        b_ub=np.array(b_ub_lp) if b_ub_lp else None,
        A_eq=np.array(A_eq_lp) if A_eq_lp else None,
        b_eq=np.array(b_eq_lp) if b_eq_lp else None,
        bounds=bounds_lp, # Sử dụng bounds_lp đã điều chỉnh
        method='highs'
    )

    if res.status == 3: # Status 3 means unbounded
        unbounded_value = "+∞" if loai == 'max' else "-∞"
        print(f"\n⇒ Bài toán không giới nội. Hàm mục tiêu đạt {loai} tại {unbounded_value}.")

        plt.figure(figsize=(8, 8))
        colors = matplotlib.colormaps['tab10'].resampled(len(A))
        for i in range(len(A)):
            x_vals_plot = np.linspace(x_min, x_max, 400)
            if A[i][1] != 0:
                y_vals_plot = (b[i] - A[i][0] * x_vals_plot) / A[i][1]
                plt.plot(x_vals_plot, y_vals_plot, label=f'{A[i][0]:.2f}x1 + {A[i][1]:.2f}x2 {rls[i]} {b[i]:.2f}', color=colors(i), linestyle='-')
            elif A[i][0] != 0:
                plt.axvline(x=b[i] / A[i][0], label=f'{A[i][0]:.2f}x1 {rls[i]} {b[i]:.2f}', color=colors(i), linestyle='-')

        if poly and poly.geom_type == 'Polygon':
            x_poly, y_poly = poly.exterior.xy
            plt.fill(x_poly, y_poly, color='lightgreen', alpha=0.4, label='Miền khả thi', zorder=1)

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.axvline(0, color='black', linewidth=0.8, zorder=0)
        plt.axhline(0, color='black', linewidth=0.8, zorder=0)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title("Bài toán không giới nội")
        plt.legend()
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
        return None, None

    # Tính giá trị hàm mục tiêu tại các điểm khả thi
    Z_vals = [np.dot(c, x) for x in feasible_points]

    if not Z_vals:
        print("Bài toán vô nghiệm (không có điểm khả thi sau khi lọc).")
        return None, None

    if loai == 'min':
        val_opt = min(Z_vals)
        cmp_func = lambda z: np.isclose(z, val_opt, atol=1e-6)
    else:
        val_opt = max(Z_vals)
        cmp_func = lambda z: np.isclose(z, val_opt, atol=1e-6)

    diem_toi_uu_all = []
    seen = set()
    for i in range(len(Z_vals)):
        if cmp_func(Z_vals[i]):
            point_rounded = tuple(np.round(feasible_points[i], 6))
            if point_rounded not in seen:
                seen.add(point_rounded)
                diem_toi_uu_all.append(feasible_points[i])

    final_x_values = None
    val_primal = val_opt

    if len(diem_toi_uu_all) == 1:
        final_x_values = diem_toi_uu_all[0]
    elif len(diem_toi_uu_all) > 1:
        diem_toi_uu_all_sorted = sorted(diem_toi_uu_all, key=lambda p: (p[0], p[1]))
        final_x_values = diem_toi_uu_all_sorted

    # Hiển thị kết quả
    print("\n--- Kết quả Phương pháp hình học ---")
    print("Các điểm khả thi:")
    for point, val in zip(feasible_points, Z_vals):
        print(f"x = ({point[0]:.4f}, {point[1]:.4f}), Z = {val:.4f}")

    if len(diem_toi_uu_all) > 1:
        print("\n⇒ Bài toán có vô số nghiệm tối ưu.")
        p1 = final_x_values[0]
        p2 = final_x_values[-1]
        print(f"Tập nghiệm tối ưu là đoạn thẳng từ ({p1[0]:.4f}, {p1[1]:.4f}) đến ({p2[0]:.4f}, {p2[1]:.4f})")
        print("⇒ Các nghiệm có dạng:")
        print(f"   x = (1 - t)*({p1[0]:.4f}, {p1[1]:.4f}) + t*({p2[0]:.4f}, {p2[1]:.4f}) với t ∈ [0, 1]")
    else:
        print("\n⇒ Bài toán có nghiệm tối ưu duy nhất.")

    print(f"Giá trị tối ưu Z = {val_primal:.4f}")

    if len(diem_toi_uu_all) == 1:
        print("Nghiệm tối ưu:")
        print(f"  x1 = {final_x_values[0]:.4f}, x2 = {final_x_values[1]:.4f}")


    # Vẽ miền nghiệm và nghiệm tối ưu
    plt.figure(figsize=(10, 8))

    colors = matplotlib.colormaps['tab10'].resampled(len(A))

    for i in range(len(A)):
        x_vals_line = np.linspace(x_min, x_max, 400)
        if A[i][1] != 0:
            y_vals_line = (b[i] - A[i][0] * x_vals_line) / A[i][1]
            plt.plot(x_vals_line, y_vals_line, label=f'{A[i][0]:.2f}x1 + {A[i][1]:.2f}x2 {rls[i]} {b[i]:.2f}',
                     color=colors(i), linestyle='-', linewidth=2, zorder=3)
        elif A[i][0] != 0:
            plt.axvline(x=b[i] / A[i][0], label=f'{A[i][0]:.2f}x1 {rls[i]} {b[i]:.2f}',
                        color=colors(i), linestyle='-', linewidth=2, zorder=3)

    if poly and poly.geom_type == 'Polygon':
        x_poly, y_poly = poly.exterior.xy
        plt.fill(x_poly, y_poly, color='lightgreen', alpha=0.4, label='Miền khả thi', zorder=1)

    if feasible_points:
        xs, ys = zip(*feasible_points)
        plt.scatter(xs, ys, color='blue', zorder=4, label='Các đỉnh khả thi', s=50, edgecolors='black')

    # Vẽ nghiệm tối ưu
    if final_x_values is not None:
        if isinstance(final_x_values, list) and len(final_x_values) > 1: # Vô số nghiệm (đoạn thẳng)
            p1, p2 = final_x_values[0], final_x_values[-1]
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', lw=3, label='Đoạn nghiệm tối ưu', zorder=5)
            plt.scatter([p1[0], p2[0]], [p1[1], p2[1]], color='red', marker='o', s=100, zorder=6, edgecolors='black')
        elif isinstance(final_x_values, np.ndarray): # Nghiệm duy nhất (một điểm)
            x_opt, y_opt = final_x_values[0], final_x_values[1]
            plt.scatter(x_opt, y_opt, color='red', marker='o', s=100, label='Nghiệm tối ưu', zorder=6, edgecolors='black')

    # Vẽ các trục tọa độ chính (x=0, y=0)
    plt.axvline(0, color='black', linewidth=0.8, zorder=0)
    plt.axhline(0, color='black', linewidth=0.8, zorder=0)

    # Điều chỉnh giới hạn trục để đảm bảo tất cả đều hiển thị
    all_x = [p[0] for p in feasible_points]
    all_y = [p[1] for p in feasible_points]

    if final_x_values is not None:
        if isinstance(final_x_values, list): # Trường hợp vô số nghiệm
            all_x.extend([p[0] for p in final_x_values])
            all_y.extend([p[1] for p in final_x_values])
        elif isinstance(final_x_values, np.ndarray): # Trường hợp nghiệm duy nhất
            all_x.append(final_x_values[0])
            all_y.append(final_x_values[1])

    # Đảm bảo có ít nhất một điểm để tính toán giới hạn
    if all_x and all_y:
        x_buffer = max(1, (max(all_x) - min(all_x)) * 0.1)
        y_buffer = max(1, (max(all_y) - min(all_y)) * 0.1)

        # Sử dụng x_min_plot, y_min_plot để đảm bảo trục bắt đầu đúng
        final_x_min = min(x_min_plot, min(all_x) - x_buffer)
        final_x_max = max(x_max_plot, max(all_x) + x_buffer)
        final_y_min = min(y_min_plot, min(all_y) - y_buffer)
        final_y_max = max(y_max_plot, max(all_y) + y_buffer)

        plt.xlim(final_x_min, final_x_max)
        plt.ylim(final_y_min, final_y_max)
    else: # Trường hợp không có điểm khả thi nào để tính toán giới hạn
        plt.xlim(x_min_plot, x_max_plot)
        plt.ylim(y_min_plot, y_max_plot)


    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(f"Miền nghiệm và nghiệm tối ưu ({loai.upper()} Z)")
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    return final_x_values, val_primal

def bland(c_std, A_std, b_std, loai_bt, n_original_vars, var_types, standardized_var_names):
    """
    Giải bài toán quy hoạch tuyến tính bằng phương pháp Bland để tránh vòng lặp.
    Hàm này thực hiện các bước pivot cho đến khi tìm được nghiệm tối ưu hoặc kết luận bài toán không giới nội.
    """
    m_std, n_std_vars = A_std.shape # m_std là số ràng buộc sau khi chuyển về dạng chuẩn, n_std_vars là số biến sau khi chuyển đổi (x', x'' và x)

    # Khởi tạo bảng từ vựng và danh sách biến cơ sở
    B, co_so = khoi_tao_bang_tu_vung(c_std, A_std, b_std)

    # bien_names bây giờ chính là standardized_var_names đã được truyền vào
    bien_names = standardized_var_names + [f"w{i+1}" for i in range(m_std)]

    pivot_count = 0 # Đếm số lần pivot

    final_x_values = None
    val_primal = None

    while True:
        pivot_count += 1
        bien_vao_col_idx = -1 # Chỉ số cột của biến vào (trong bảng B)

        # Bước 1: Chọn biến vào (theo Bland: hệ số âm đầu tiên)
        for j in range(1, 1 + n_std_vars + m_std): # Duyệt qua các cột biến (x và w)
            if B[0, j] < -1e-10:
                bien_vao_col_idx = j
                break

        # Bước 2: Kiểm tra điều kiện dừng (tối ưu)
        if bien_vao_col_idx == -1:
            # Nếu không tìm thấy hệ số âm nào trong hàng mục tiêu, bài toán đã tối ưu
            in_bang_tu_vung(B, bien_names, co_so, -1, pivot_count, n_std_vars, is_optimal_tableau=True) # In bảng tối ưu

            # Giá trị tối ưu của hàm mục tiêu (Z)
            z_opt_std = - B[0, 0] # Giá trị Z trong bảng từ vựng
            val_primal = -z_opt_std if loai_bt == 'max' else z_opt_std

            # Bước 3: Kiểm tra vô số nghiệm tối ưu
            has_multiple_solutions = False
            non_basic_vars_with_zero_cost = []

            for j in range(1, 1 + n_std_vars + m_std):
                # Kiểm tra xem biến có phải là biến không cơ sở không
                # co_so chứa chỉ số 0-indexed của biến trong bien_names
                # j-1 là chỉ số 0-indexed của biến trong bien_names tương ứng với cột j trong bảng B
                if (j - 1) not in co_so:
                    # Nếu là biến không cơ sở và có hệ số trong hàng mục tiêu xấp xỉ 0
                    if abs(B[0, j]) < 1e-10:
                        has_multiple_solutions = True
                        non_basic_vars_with_zero_cost.append(bien_names[j-1]) # Thêm tên biến vào danh sách

            if has_multiple_solutions:
                print("\nBài toán có vô số nghiệm tối ưu.")
                print("Kết luận nghiệm và giá trị tối ưu bài toán gốc (P):")

                if len(non_basic_vars_with_zero_cost) >= 1:
                    free_var_name = non_basic_vars_with_zero_cost[0]
                    free_var_std_idx = standardized_var_names.index(free_var_name)
                    free_var_col_B_index = free_var_std_idx + 1

                    upper_bound = float('inf')
                    for i in range(1, m_std + 1):
                        coeff_in_row = B[i, free_var_col_B_index]
                        if coeff_in_row > 1e-10:
                            ratio = B[i, 0] / coeff_in_row
                            upper_bound = min(upper_bound, ratio)

                    lower_bound = 0.0

                    final_solution_strings = []
                    current_std_idx_for_recon = 0

                    for k in range(n_original_vars):
                        original_x_name = f"x{k+1}"

                        # Lấy các chỉ số biến chuẩn hóa tương ứng với biến gốc x_k
                        std_var_indices_for_xk = []
                        if var_types[k] == 'free':
                            std_var_indices_for_xk = [current_std_idx_for_recon, current_std_idx_for_recon + 1]
                            current_std_idx_for_recon += 2
                        else: # >= 0 hoặc <= 0
                            std_var_indices_for_xk = [current_std_idx_for_recon]
                            current_std_idx_for_recon += 1

                        # Kiểm tra xem biến gốc (hoặc một phần của nó) có phải là biến tự do được chọn không
                        is_this_the_free_var_relevance = False
                        for idx in std_var_indices_for_xk:
                            if idx < len(standardized_var_names) and standardized_var_names[idx] == free_var_name:
                                is_this_the_free_var_relevance = True
                                break

                        if is_this_the_free_var_relevance:
                            final_solution_strings.append(f"{original_x_name} (biến tự do)")
                            continue # Chuyển sang biến gốc tiếp theo

                        # Logic cho các biến gốc khác (KHÔNG phải biến tự do)
                        const_term = 0.0
                        coeff_term = 0.0
                        is_current_x_basic = False

                        if len(std_var_indices_for_xk) == 1: # x_k = y_k hoặc x_k = -y_k
                            std_var_idx = std_var_indices_for_xk[0]
                            if std_var_idx in co_so: # Nếu biến chuẩn hóa này là biến cơ sở
                                is_current_x_basic = True
                                row_in_basis = co_so.index(std_var_idx) + 1
                                const_term = B[row_in_basis, 0]
                                coeff_term = -B[row_in_basis, free_var_col_B_index]

                            if var_types[k] == '<= 0': # x_k = -y_k, đảo dấu kết quả
                                const_term = -const_term
                                coeff_term = -coeff_term
                        elif len(std_var_indices_for_xk) == 2: # x_k = x_k^+ - x_k^- (biến tự do ban đầu)
                            x_plus_idx = std_var_indices_for_xk[0]
                            x_minus_idx = std_var_indices_for_xk[1]

                            # Kiểm tra xem ít nhất một trong x_k^+ hoặc x_k^- có phải là biến cơ sở không
                            if x_plus_idx in co_so or x_minus_idx in co_so:
                                is_current_x_basic = True

                                val_x_plus = 0.0
                                coeff_x_plus_free_var = 0.0
                                if x_plus_idx in co_so:
                                    row_in_basis_plus = co_so.index(x_plus_idx) + 1
                                    val_x_plus = B[row_in_basis_plus, 0]
                                    coeff_x_plus_free_var = B[row_in_basis_plus, free_var_col_B_index]

                                val_x_minus = 0.0
                                coeff_x_minus_free_var = 0.0
                                if x_minus_idx in co_so:
                                    row_in_basis_minus = co_so.index(x_minus_idx) + 1
                                    val_x_minus = B[row_in_basis_minus, 0]
                                    coeff_x_minus_free_var = B[row_in_basis_minus, free_var_col_B_index]

                                const_term = val_x_plus - val_x_minus
                                coeff_term = -(coeff_x_plus_free_var - coeff_x_minus_free_var)

                        if is_current_x_basic:
                            expr_str = f"{original_x_name} = {const_term:.4f}"
                            if abs(coeff_term) > 1e-10:
                                if coeff_term > 0:
                                    expr_str += f" + {abs(coeff_term):.4f} * {free_var_name}"
                                else:
                                    expr_str += f" - {abs(coeff_term):.4f} * {free_var_name}"
                            final_solution_strings.append(expr_str)
                        else: # Biến không cơ sở (không phải biến tự do được chọn), nên giá trị bằng 0
                            final_solution_strings.append(f"{original_x_name} = 0.0000")

                    print("Các biến cơ sở được biểu diễn theo biến tự do:")
                    for sol_str in final_solution_strings:
                        print(f"    {sol_str}")

                    if upper_bound != float('inf'):
                        print(f"Với {free_var_name} là biến tự do, {lower_bound:.4f} <= {free_var_name} <= {upper_bound:.4f}")
                    else:
                        print(f"Với {free_var_name} là biến tự do và {free_var_name} >= {lower_bound:.4f} (không giới hạn trên)")
                    print(f"Giá trị tối ưu của (P): {val_primal:.4f}")
                    return None, val_primal

            else: # Nghiệm tối ưu duy nhất
                print("\nBài toán có nghiệm tối ưu duy nhất.")
                print("\nKết luận nghiệm và giá trị tối ưu bài toán gốc (P):")
                x_opt_std_vars = np.zeros(n_std_vars)
                for i, var_idx in enumerate(co_so):
                    if var_idx < n_std_vars:
                        x_opt_std_vars[var_idx] = B[i+1, 0]

                final_x_values = np.zeros(n_original_vars)
                current_std_idx = 0
                for i in range(n_original_vars):
                    if var_types[i] == 'free':
                        final_x_values[i] = x_opt_std_vars[current_std_idx] - x_opt_std_vars[current_std_idx + 1]
                        current_std_idx += 2
                    elif var_types[i] == '<= 0':
                        final_x_values[i] = -x_opt_std_vars[current_std_idx]
                        current_std_idx += 1
                    else: # >= 0
                        final_x_values[i] = x_opt_std_vars[current_std_idx]
                        current_std_idx += 1

                for i in range(n_original_vars):
                    print(f"x{i+1} = {final_x_values[i]:.4f}")
                print(f"Giá trị tối ưu của (P): {val_primal:.4f}")
                return final_x_values, val_primal

        in_bang_tu_vung(B, bien_names, co_so, bien_vao_col_idx, pivot_count, n_std_vars, is_optimal_tableau=False)

        # Bước 4: Chọn biến ra (Quy tắc Bland: chọn hàng có tỉ lệ nhỏ nhất,
        # Nếu bằng nhau thì ưu tiên biến cơ sở có chỉ số nhỏ nhất)

        bien_ra_row_idx = -1
        min_ratio = float('inf')

        for i in range(1, m_std + 1):
            a_ij = B[i, bien_vao_col_idx]
            if a_ij > 1e-10:
                ratio = B[i, 0] / a_ij

                if ratio < min_ratio - 1e-10 or \
                   (abs(ratio - min_ratio) < 1e-10 and \
                    (bien_ra_row_idx == -1 or co_so[i-1] < co_so[bien_ra_row_idx-1])):
                    min_ratio = ratio
                    bien_ra_row_idx = i

        if bien_ra_row_idx == -1:
            print("Không tồn tại biến ra cho biến vào => bài toán không giới nội.")
            if loai_bt == 'min':
                print("Giá trị tối ưu của bài toán gốc (P) là -oo (âm vô cùng).")
            else:
                print("Giá trị tối ưu của bài toán gốc (P) là +oo (dương vô cùng).")
            return None, None

        bien_co_so_cu_idx = co_so[bien_ra_row_idx-1]
        print(f"Biến vào: {bien_names[bien_vao_col_idx-1]}, biến ra: {bien_names[bien_co_so_cu_idx]}")

        co_so[bien_ra_row_idx-1] = bien_vao_col_idx - 1

        pivot_element = B[bien_ra_row_idx, bien_vao_col_idx]
        B[bien_ra_row_idx, :] = B[bien_ra_row_idx, :] / pivot_element

        for i in range(B.shape[0]):
            if i != bien_ra_row_idx:
                B[i, :] -= B[i, bien_vao_col_idx] * B[bien_ra_row_idx, :]


def don_hinh(c_std, A_std, b_std, loai_bt, n_original_vars, var_types, standardized_var_names):
    """
    Giải bài toán quy hoạch tuyến tính bằng phương pháp đơn hình tiêu chuẩn.
    Hàm này thực hiện các bước pivot cho đến khi tìm được nghiệm tối ưu hoặc kết luận bài toán không giới nội.
    """
    m_std, n_std_vars = A_std.shape

    # Khởi tạo bảng từ vựng và danh sách biến cơ sở
    B, co_so = khoi_tao_bang_tu_vung(c_std, A_std, b_std)

    # bien_names là tên của tất cả các biến (biến gốc chuẩn hóa và biến bù)
    bien_names = standardized_var_names + [f"w{i+1}" for i in range(m_std)]

    pivot_count = 0 # Đếm số lần pivot

    print("\n--- Bắt đầu giải bằng phương pháp đơn hình tiêu chuẩn ---")

    # Kiểm tra điều kiện đầu vào: đơn hinh yêu cầu tất cả b_std phải không âm
    # để có thể bắt đầu từ một giải pháp cơ sở khả thi (các biến bù là biến cơ sở ban đầu).
    if np.any(b_std < -1e-9):
        print("\n!!! Lỗi: Phương pháp đơn hình tiêu chuẩn yêu cầu tất cả các giá trị vế phải (b_i) phải không âm.")
        print("!!! Bài toán của bạn có b_i âm sau khi chuyển về dạng chuẩn.")
        print("!!! Để giải bài toán này, bạn cần sử dụng Phương pháp Hai Pha hoặc biến đổi thêm để đảm bảo tính khả thi ban đầu.")
        print("Bài toán không thể giải được bằng Phương pháp đơn hình tiêu chuẩn theo cách này.")
        return None, None # Trả về None nếu không thể giải

    while True:
        pivot_count += 1
        bien_vao_col_idx = -1 # Chỉ số cột của biến vào (trong bảng B)
        max_negative_coeff = 0.0 # Để tìm hệ số âm lớn nhất (tức là âm nhất)

        # Bước 1: Chọn biến vào (theo đơn hinh: hệ số âm có giá trị tuyệt đối lớn nhất)
        # Duyệt qua các cột biến (từ x1...xn, w1...wm) trong hàng mục tiêu
        for j in range(1, 1 + n_std_vars + m_std):
            if B[0, j] < -1e-10: # Nếu hệ số trong hàng mục tiêu âm
                if B[0, j] < max_negative_coeff: # So sánh với hệ số âm lớn nhất đã tìm thấy
                    max_negative_coeff = B[0, j]
                    bien_vao_col_idx = j

        # Bước 2: Kiểm tra điều kiện dừng (tối ưu)
        if bien_vao_col_idx == -1:
            # Nếu không tìm thấy hệ số âm nào trong hàng mục tiêu, bài toán đã tối ưu
            in_bang_tu_vung(B, bien_names, co_so, -1, pivot_count, n_std_vars, is_optimal_tableau=True) # In bảng tối ưu

            # Giá trị tối ưu của hàm mục tiêu (Z) trong bài toán chuẩn (min -Z hoặc min Z)
            z_opt_std = -B[0, 0]
            # Chuyển về giá trị tối ưu của bài toán gốc (max Z hoặc min Z)
            val_primal = -z_opt_std if loai_bt == 'max' else z_opt_std

            # Bước 3: Kiểm tra vô số nghiệm tối ưu
            has_multiple_solutions = False
            non_basic_vars_with_zero_cost = []

            for j in range(1, 1 + n_std_vars + m_std):
                # Kiểm tra xem biến có phải là biến không cơ sở không
                # j-1 là chỉ số 0-indexed của biến trong bien_names tương ứng với cột j trong bảng B
                if (j - 1) not in co_so:
                    # Nếu là biến không cơ sở và có hệ số trong hàng mục tiêu xấp xỉ 0
                    if abs(B[0, j]) < 1e-10:
                        has_multiple_solutions = True
                        non_basic_vars_with_zero_cost.append(bien_names[j-1]) # Thêm tên biến vào danh sách

            if has_multiple_solutions:
                print("\nBài toán có vô số nghiệm tối ưu.")
                print("Kết luận nghiệm và giá trị tối ưu bài toán gốc (P):")

                if len(non_basic_vars_with_zero_cost) >= 1:
                    # Lấy biến tự do đầu tiên để biểu diễn các biến khác theo
                    free_var_name = non_basic_vars_with_zero_cost[0]
                    # Tìm chỉ số cột của biến tự do này trong bảng B
                    free_var_col_B_index = bien_names.index(free_var_name) + 1

                    upper_bound = float('inf')
                    # Tìm giới hạn trên của biến tự do
                    for i in range(1, m_std + 1):
                        coeff_in_row = B[i, free_var_col_B_index]
                        # Nếu hệ số dương, tính tỉ lệ để tìm giới hạn trên
                        if coeff_in_row > 1e-10:
                            ratio = B[i, 0] / coeff_in_row
                            upper_bound = min(upper_bound, ratio)

                    lower_bound = 0.0 # Biến chuẩn hóa luôn >= 0

                    final_solution_strings = []
                    current_std_idx_for_recon = 0

                    # Tái tạo nghiệm cho các biến gốc x_k
                    for k in range(n_original_vars):
                        original_x_name = f"x{k+1}"

                        # Lấy các chỉ số biến chuẩn hóa tương ứng với biến gốc x_k
                        std_var_indices_for_xk = []
                        if var_types[k] == 'free':
                            std_var_indices_for_xk = [current_std_idx_for_recon, current_std_idx_for_recon + 1]
                            current_std_idx_for_recon += 2
                        else: # >= 0 hoặc <= 0
                            std_var_indices_for_xk = [current_std_idx_for_recon]
                            current_std_idx_for_recon += 1

                        # Kiểm tra xem biến gốc (hoặc một phần của nó) có phải là biến tự do được chọn không
                        is_this_the_free_var_relevance = False
                        for idx in std_var_indices_for_xk:
                            if idx < len(standardized_var_names) and standardized_var_names[idx] == free_var_name:
                                is_this_the_free_var_relevance = True
                                break

                        if is_this_the_free_var_relevance:
                            final_solution_strings.append(f"{original_x_name} = {free_var_name} (biến tự do)")
                            continue

                        # Logic cho các biến gốc khác (KHÔNG phải biến tự do)
                        const_term = 0.0
                        coeff_term = 0.0
                        is_current_x_basic = False

                        if len(std_var_indices_for_xk) == 1: # x_k = y_k hoặc x_k
                            std_var_idx = std_var_indices_for_xk[0]
                            if std_var_idx in co_so: # Nếu biến chuẩn hóa này là biến cơ sở
                                is_current_x_basic = True
                                row_in_basis = co_so.index(std_var_idx) + 1
                                const_term = B[row_in_basis, 0]
                                coeff_term = -B[row_in_basis, free_var_col_B_index]

                            if var_types[k] == '<=': # x_k = -y_k, đảo dấu kết quả
                                const_term = -const_term
                                coeff_term = -coeff_term
                        elif len(std_var_indices_for_xk) == 2: # x_k = x_k^+ - x_k^- (biến tự do ban đầu)
                            x_plus_idx = std_var_indices_for_xk[0]
                            x_minus_idx = std_var_indices_for_xk[1]

                            # Kiểm tra xem ít nhất một trong x_k^+ hoặc x_k^- có phải là biến cơ sở không
                            if x_plus_idx in co_so or x_minus_idx in co_so:
                                is_current_x_basic = True

                                val_x_plus = 0.0
                                coeff_x_plus_free_var = 0.0
                                if x_plus_idx in co_so:
                                    row_in_basis_plus = co_so.index(x_plus_idx) + 1
                                    val_x_plus = B[row_in_basis_plus, 0]
                                    coeff_x_plus_free_var = B[row_in_basis_plus, free_var_col_B_index]

                                val_x_minus = 0.0
                                coeff_x_minus_free_var = 0.0
                                if x_minus_idx in co_so:
                                    row_in_basis_minus = co_so.index(x_minus_idx) + 1
                                    val_x_minus = B[row_in_basis_minus, 0]
                                    coeff_x_minus_free_var = B[row_in_basis_minus, free_var_col_B_index]

                                const_term = val_x_plus - val_x_minus
                                coeff_term = -(coeff_x_plus_free_var - coeff_x_minus_free_var)

                        if is_current_x_basic:
                            expr_str = f"{original_x_name} = {const_term:.4f}"
                            if abs(coeff_term) > 1e-10:
                                if coeff_term > 0:
                                    expr_str += f" + {abs(coeff_term):.4f} * {free_var_name}"
                                else:
                                    expr_str += f" - {abs(coeff_term):.4f} * {free_var_name}"
                            final_solution_strings.append(expr_str)
                        else: # Biến không cơ sở (không phải biến tự do được chọn), nên giá trị bằng 0
                            final_solution_strings.append(f"{original_x_name} = 0.0000")

                    print("Các biến cơ sở được biểu diễn theo biến tự do:")
                    for sol_str in final_solution_strings:
                        print(f"    {sol_str}")

                    if upper_bound != float('inf'):
                        print(f"Với {free_var_name} là biến tự do, {lower_bound:.4f} <= {free_var_name} <= {upper_bound:.4f}")
                    else:
                        print(f"Với {free_var_name} là biến tự do và {free_var_name} >= {lower_bound:.4f} (không giới hạn trên)")
                    print(f"Giá trị tối ưu của (P): {val_primal:.4f}")
                    return None, val_primal

            else: # Nghiệm tối ưu duy nhất
                print("\nBài toán có nghiệm tối ưu duy nhất.")
                print("\nKết luận nghiệm và giá trị tối ưu bài toán gốc (P):")
                x_opt_std_vars = np.zeros(n_std_vars)
                for i, var_idx in enumerate(co_so):
                    if var_idx < n_std_vars:
                        x_opt_std_vars[var_idx] = B[i+1, 0]

                final_x_values = np.zeros(n_original_vars)
                current_std_idx = 0
                for i in range(n_original_vars):
                    if var_types[i] == 'free':
                        final_x_values[i] = x_opt_std_vars[current_std_idx] - x_opt_std_vars[current_std_idx + 1]
                        current_std_idx += 2
                    elif var_types[i] == '<=':
                        final_x_values[i] = -x_opt_std_vars[current_std_idx]
                        current_std_idx += 1
                    else: # >=
                        final_x_values[i] = x_opt_std_vars[current_std_idx]
                        current_std_idx += 1

                for i in range(n_original_vars):
                    print(f"x{i+1} = {final_x_values[i]:.4f}")
                print(f"Giá trị tối ưu của (P): {val_primal:.4f}")
                return final_x_values, val_primal

        # In bảng từ vựng hiện tại trước khi pivot
        in_bang_tu_vung(B, bien_names, co_so, bien_vao_col_idx, pivot_count, n_std_vars, is_optimal_tableau=False)

        # Bước 4: Chọn biến ra (Quy tắc tỉ lệ tối thiểu: chọn hàng có tỉ lệ dương nhỏ nhất)

        bien_ra_row_idx = -1
        min_ratio = float('inf')

        for i in range(1, m_std + 1):
            a_ij = B[i, bien_vao_col_idx]
            if a_ij > 1e-10: # Chỉ xét các hệ số dương để đảm bảo biến có thể giảm giá trị
                ratio = B[i, 0] / a_ij
                if ratio < min_ratio - 1e-10: # Nếu tỉ lệ nhỏ hơn
                    min_ratio = ratio
                    bien_ra_row_idx = i

        # Bước 5: Kiểm tra bài toán không giới nội
        if bien_ra_row_idx == -1:
            print("\nKhông tồn tại biến ra cho biến vào. Tất cả hệ số trong cột của biến vào đều <= 0.")
            print("=> Bài toán không giới nội (Unbounded).")
            if loai_bt == 'min':
                print("Giá trị tối ưu của bài toán gốc (P) là -∞ (âm vô cùng).")
            else:
                print("Giá trị tối ưu của bài toán gốc (P) là +∞ (dương vô cùng).")
            return None, None

        bien_co_so_cu_idx = co_so[bien_ra_row_idx-1]
        print(f"\nChọn biến vào: {bien_names[bien_vao_col_idx-1]}, biến ra: {bien_names[bien_co_so_cu_idx]}")

        # Bước 6: Thực hiện phép pivot
        # Cập nhật biến cơ sở
        co_so[bien_ra_row_idx-1] = bien_vao_col_idx - 1 # Biến vào thay thế biến ra

        # Chia hàng biến ra cho phần tử pivot
        pivot_element = B[bien_ra_row_idx, bien_vao_col_idx]
        B[bien_ra_row_idx, :] = B[bien_ra_row_idx, :] / pivot_element

        # Biến đổi các hàng khác
        for i in range(B.shape[0]):
            if i != bien_ra_row_idx:
                B[i, :] -= B[i, bien_vao_col_idx] * B[bien_ra_row_idx, :]

def hai_pha(A,b,c,loai):

    def print_tableau(tableau, step=None):
            print("\n========== Bảng từ vựng {} ==========".format(step if step is not None else ""))
            print(np.round(tableau, 4)) 
        
        
    def pivot_step(tableau, row, col):
            pivot = tableau[row, col]
            tableau[row, :] = tableau[row, :] / pivot
            for i in range(len(tableau)):
                if i != row:
                    tableau[i, :] -= tableau[i, col] * tableau[row, :]
            tableau[row, :] *= -1
            return tableau

    def pha1(tableau, x0_col):
    
        def don_hinh_pha1(tableau, x0_col, start_step=1):
            # Bước xoay
            step = start_step

            while True:
                # Dòng cuối của tableau (Z)
                # [:-1] : bỏ cột cuối(chỉ lấy hệ số biến)
                cj = tableau[-1, :-1]
                # Dừng nếu hệ số x0 là 1, các biến còn lại = 0
                is_done = np.isclose(cj[x0_col], 1.0) and np.all(np.isclose(np.delete(cj, x0_col), 0))
                if is_done:
                    print("\n Dừng: Hàm mục tiêu Z = x0.")
                    print_tableau(tableau, f"kết thúc pha 1")
                    return tableau

                # Tìm cột có cj âm nhất (trừ x0)
                cj_temp = cj.copy()
                #cj_temp[x0_col] = 0  # không chọn x0 làm biến vào nữa
                candidates = np.where(cj_temp < 0)[0]

                if len(candidates) == 0:
                    print("\n Không còn cột cj âm để chọn → bài toán vô nghiệm.")
                    return None

                # cj_temp : hệ số cj trong Z bỏ cột b
                # candidate : chỉ số các cột có cj âm
                # cj_temp[candidate] : gia trị âm 
                #np.argmin(..): trả về chỉ số tại biến nhỏ nhất
                # canditate[np.argmin(..)] : trả về vị trí thật trong mảng cj_temp
                # ví dụ : cj_temp = [-1 -1 0 1 0 0 ]
                        # candidate = [0 1]
                        # cj_temp[candidate] = [-1 -1]
                        # np.argmin(cj_temp[candidate]) = 0
                        # candidate[0] = 0
                # => col: lấy biến vào (hệ số cj âm nhất)
                col = candidates[np.argmin(cj_temp[candidates])]

                #Chuẩn bị chọn biến ra:
                #[:-1, col] : lấy cột thứ col của tất cả các hàng(trừ dòng cuối)
                # giúp lấy ra các hàng có hệ số âm
                col_vals = tableau[:-1, col]
                # Lấy tất cả giá trị bi trong từng ràng buộc (ko lấy dòng z)
                b_vals = tableau[:-1, -1]

                # Tập chấp nhận: chỉ xét a_ij < 0
                # Danh sách aij âm trong cột biến vào ở các hàng
                valid_rows = [i for i in range(len(col_vals)) if col_vals[i] < 0]
                if not valid_rows:
                    print("\n Tập chấp nhận rỗng → bài toán vô nghiệm trong pha 1.")
                    return None

                # ví dụ:
                    # b_vals = [3 2 7]
                    # col_vals = [-1 0 -2]
                    # valid_rows = [0 2]
                    # ratios = [3/|-1|; 7/|-2|]
                    #np.argmin(ratios) = 0
                    # valid_rows[0] = 0 -> row : biến ra
                ratios = [b_vals[i] / abs(col_vals[i]) for i in valid_rows]
                row = valid_rows[np.argmin(ratios)]

                print(f"\n Pivot tại row = {row}, col = {col} (đơn hình pha 1)")
                tableau = pivot_step(tableau, row, col)
                step += 1
                print_tableau(tableau, step)

        print_tableau(tableau, "ban đầu")
        # Pivot đầu tiên: vào x0, ra hàng có b âm nhất
        b_vals = tableau[:-1, -1]
        row = np.argmin(b_vals)  # hàng có bi âm nhất
        print(f"\nPivot đầu tiên: vào x0 (col={x0_col}), ra row={row}")
        tableau = pivot_step(tableau, row, x0_col)
        print_tableau(tableau, "sau pivot x0")

        # Dùng đơn hình trong pha 1
        tableau = don_hinh_pha1(tableau, x0_col, start_step=1)

        return tableau


    def pha2(tableau,c, A, b):
    
        def tao_lai_z_pha2(tableau, c, x0_col):
            """
            Tạo lại hàng Z trong pha 2 sau khi đã thực hiện pha 1 của phương pháp hai pha.

            Args:
                tableau: Bảng tableau sau pha 1 (numpy array), gồm m hàng và n+1 cột (cột cuối là vế phải).
                c: Hệ số hàm mục tiêu đã chuẩn hóa (Min Cᵗx), kích thước (n, )

            Returns:
                tableau: bảng tableau với hàng Z mới được cập nhật (dòng cuối).
            """
            tableau = np.delete(tableau, x0_col, axis=1)
            m_tab, n_tab = tableau.shape  # m dòng, n+1 cột (bao gồm cột b)
            m = m_tab - 1            # số biến (không tính cột b)
            n = len(c)
            
            #z_moi = tableau[-1, :]
            z_moi = np.zeros(n_tab)      # Khởi tạo hàng z mới: n biến + 1 cột b

            # Duyệt qua từng cột j để tìm biến cơ sở
            for j in range(n):
                col = tableau[:-1, j]                  # lấy cột j (loại dòng z)
                if np.count_nonzero(col) == 1 and np.sum(col) == -1:
                    i = np.where(col == -1)[0][0]     # tìm dòng i chứa -1 → biến cơ sở nằm ở hàng i
                    c_j = c[j]                         # hệ số tương ứng trong hàm mục tiêu

                    z_moi[j] = 0
                    for k in range(n_tab - 1):  # duyệt tất cả cột trừ cột b
                        if k != j:  # bỏ qua cột tại vị trí biến cơ sở j
                            z_moi[k] += c_j * tableau[i, k]
                            # vẫn cộng phần hệ số tự do như cũ:
                    z_moi[-1] += c_j * tableau[i, -1]


            # Cộng thêm hệ số c cho các biến không cơ sở
            for j in range(n):
                col = tableau[:-1, j]
                if not (np.count_nonzero(col) == 1 and np.sum(col) == -1):
                    z_moi[j] += c[j]

            # Gán hàng z mới
            tableau[-1, :] = z_moi

            return tableau
        
        def don_hinh_pha2(tableau, start_step=1):
            step = start_step
            while True:
                cj = tableau[-1, :-1]
                if np.all(cj >= 0):
                    print("\nĐạt nghiệm tối ưu (pha 2) ")
                    return tableau
                
                cj_temp = cj.copy()
                # chỉ số của hệ số âm
                candidate = np.where(cj_temp < 0)[0]
                # Không có cj âm => không có biến vào 
                if len(candidate) == 0:
                    print("\n Không tìm được biến vào → đạt nghiệm tối ưu.")
                    return tableau
                # chỉ số của cột có hệ số âm
                col = candidate[np.argmin(cj_temp[candidate])] 
                # lấy ra cột col của tất cả các hàng ( trừ dòng z)
                col_vals = tableau[:-1, col]
                # Lấy tất cả giá trị bi trong từng ràng buộc (ko lấy dòng z)
                b_vals = tableau[:-1, -1]
                # Danh sách aij âm trong cột biến vào  => không giới nội
                valid_rows = [i for i in range(len(col_vals)) if col_vals[i] < 0]
                if not valid_rows:
                    print("\n không tìm được biến ra  => Bài toán không giới nội")
                    return "UNBOUNDED"
                
                # tính tỷ lệ
                ratios = [b_vals[i] / abs(col_vals[i]) for i in valid_rows]
                row = valid_rows[np.argmin(ratios)]

                print(f"\n Pivot tại row = {row}, col = {col} (đơn hình pha 2)")
                tableau = pivot_step(tableau, row, col)
                step += 1
                print_tableau(tableau, step)



        tableau = tao_lai_z_pha2(tableau, c, x0_col = A.shape[1])
        print_tableau(tableau,"Cho x0 = 0")
        result = don_hinh_pha2(tableau, start_step=1)
        print_tableau(result)
        if isinstance(result, str) and result == "UNBOUNDED":
            return "UNBOUNDED"
        return result

    def create_initial_tableau(A, b):
        """
        Tạo tableau ban đầu cho phương pháp hai pha.
        
        Input:
            A: ma trận ràng buộc (numpy array) dạng Ax <= b
            b: vector hằng số phía phải (numpy array)
            
        Output:
            tableau: ma trận tableau đầu tiên với dạng [-A | x0 | I | b]
        """
        m, n = A.shape  # m ràng buộc, n biến
        A = np.array(A)
        b = np.array(b)

        
        # Tạo -A
        neg_A = -1 * A
        
        # Tạo cột x0 với toàn -1 (m x 1)
        x0_col = 1 * np.ones((m, 1))
        
        # Tạo ma trận đơn vị (m x m)
        I = -1 * np.eye(m)
        
        # Chuyển b thành vector cột nếu cần
        b_col = b.reshape(-1, 1)
        
        # Ghép các thành phần: [-A | x0 | I | b]
        tableau = np.hstack((neg_A, x0_col, I, b_col))

        # Tạo hàm mục tiêu
        total_cols = n + 1 + m + 1

        z = np.zeros((1, total_cols))
        z[0, n] = 1

        tableau = np.vstack((tableau, z))
        
        return tableau

    def in_ket_qua_cuoi_cung(tableau, c, A, loai):
        if tableau is None:
            print("\n Bài toán vô nghiệm.")
            return
        
        if isinstance(tableau, str) and tableau == "UNBOUNDED":
            if loai == 'max':
                print("\n Bài toán không bị chặn trên ⇒ Z = +∞")
            else:
                print("\n Bài toán không bị chặn dưới ⇒ Z = -∞")
            return

        print("\nKết quả cuối cùng:")
        m, n_plus_m_plus_1 = tableau.shape
        m -= 1  # số ràng buộc
        n = len(c)  # số biến ban đầu

        z = tableau[-1, -1]
        if loai == 'max':
            z *= -1  # do ban đầu chuyển max → min

        print(f"\n Giá trị tối ưu của hàm mục tiêu Z = {z:.4f}")

        # Xác định biến cơ sở
        x_vals = np.zeros(n)
        for j in range(n):
            col = tableau[:-1, j]
            if np.count_nonzero(col) == 1 and np.sum(col) == -1:
                i = np.where(col == -1)[0][0]
                x_vals[j] = tableau[i, -1]

        for i, val in enumerate(x_vals):
            print(f"x{i+1} = {val:.4f}")
    
    tableau = create_initial_tableau(A,b)
    print_tableau(tableau)
    tableau = pha1(tableau, x0_col= A.shape[1])
    if tableau is not None:
        tableau = pha2(tableau, c, A, b)
        
        if isinstance(tableau, np.ndarray):
            in_ket_qua_cuoi_cung(tableau, c, A, loai)
        elif tableau == "UNBOUNDED":
            in_ket_qua_cuoi_cung("UNBOUNDED", c, A, loai)
    else:
        in_ket_qua_cuoi_cung(None, c, A, loai)
    
    return final_x_values, val_primal



if __name__ == "__main__":
    loai_bt, c_original, A_original, b_original, rls, n_original_vars, var_types = nhap_bai_toan()

    # Chuyển bài toán về dạng chuẩn:
    c_std, A_std, b_std, n_std_vars, standardized_var_names = chuyen_ve_dang_chuan(loai_bt, c_original, A_original, b_original, rls, var_types)

    # Gọi hàm để xác định và cho phép người dùng chọn phương pháp
    phuong_phap_duoc_chon = xet_phuong_phap(n_original_vars, b_std)

    if phuong_phap_duoc_chon == 1:
        print("\n--- Bạn đã chọn: Giải bằng phương pháp hình học ---")
        final_x_values, val_primal = hinh_hoc(A_original, b_original, c_original, loai_bt, rls, var_types, standardized_var_names)
    elif phuong_phap_duoc_chon == 2:
        print("\n--- Bạn đã chọn: Giải bằng phương pháp đơn hình ---")
        final_x_values, val_primal= don_hinh(c_std, A_std, b_std, loai_bt, n_original_vars, var_types, standardized_var_names)
    elif phuong_phap_duoc_chon == 3:
        print("\n--- Bạn đã chọn: Giải bằng phương pháp Bland ---")
        final_x_values, val_primal = bland(c_std, A_std, b_std, loai_bt, n_original_vars, var_types, standardized_var_names)
    else: # phuong_phap_duoc_chon == 4
        print("\n--- Bạn đã chọn: Giải bằng phương pháp hai pha ---")
        final_x_values, val_primal = hai_pha(A_std,b_std,c_std,loai_bt)
        # Gọi hàm giải bằng phương hai pha ở đây

# Thêm vào cuối file qhtt.py
def giai_tu_dong(loai_bt, c_original, A_original, b_original, rls, n_original_vars, var_types, phuong_phap = None):

    c_original = np.array(c_original)
    A_original = np.array(A_original)
    b_original = np.array(b_original)
    
    c_std, A_std, b_std, n_std, std_var_names = chuyen_ve_dang_chuan(loai_bt, c_original, A_original, b_original, rls, var_types)

    #phuong_phap_duoc_chon = xet_phuong_phap(n_original_vars, b_std)
    if phuong_phap is None:
        phuong_phap_duoc_chon = xet_phuong_phap(n_original_vars, b_std)
    else:
        phuong_phap_duoc_chon = int(phuong_phap)  # hoặc chuyển tên thành số tùy cách bạn map

    if phuong_phap_duoc_chon == 1:
        final_x_values, val_primal = hinh_hoc(A_original, b_original, c_original, loai_bt, rls, var_types, std_var_names)
    elif phuong_phap_duoc_chon == 2:
        final_x_values, val_primal= don_hinh(c_std, A_std, b_std, loai_bt, n_original_vars, var_types, std_var_names)
    elif phuong_phap_duoc_chon == 3:
        final_x_values, val_primal = bland(c_std, A_std, b_std, loai_bt, n_original_vars, var_types, std_var_names)
    elif phuong_phap_duoc_chon == 4:
        final_x_values, val_primal = hai_pha(A_std,b_std,c_std,loai_bt)
    else:
        return "Phương pháp không hợp lệ!"

    # Trả chuỗi kết quả để hiển thị
    return f'Nghiệm tối ưu: {final_x_values}\nGiá trị tối ưu: {val_primal}'