from flask import Flask, render_template, request
import numpy as np
from qhtt import giai_tu_dong

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    methods = None
    message = None
    result = None

    if request.method == 'POST':
        action = request.form.get('action')

        # Lấy dữ liệu nhập đúng tên trường
        loai_bt = request.form.get('problem_type')
        c_raw = request.form.get('objective_coeffs', '')
        constraints_raw = request.form.get('constraints', '').strip()
        var_bounds_raw = request.form.get('variable_bounds', '').strip()

        # Phân tích dữ liệu nhập
        try:
            c = list(map(float, c_raw.split()))
            
            A = []
            b = []
            rls = []
            for line in constraints_raw.split('\n'):
                parts = line.split()
                if len(parts) < 3:
                    raise ValueError(f"Ràng buộc không đúng định dạng: {line}")
                *coef_str, sign, rhs_str = parts
                A.append(list(map(float, coef_str)))
                rls.append(sign)
                b.append(float(rhs_str))

            var_types = [line.strip() for line in var_bounds_raw.split('\n') if line.strip()]

        except Exception as e:
            message = f"Lỗi dữ liệu nhập: {e}"
            return render_template('index.html', message=message, methods=None,
                                   problem_type=loai_bt, objective_coeffs=c_raw,
                                   constraints=constraints_raw, variable_bounds=var_bounds_raw)

        if action == 'get_methods':
            ALL_METHODS = ['geometry', 'simplex', 'bland', 'two_phase']
            methods = ALL_METHODS.copy()
            if len(c) > 3 and 'geometry' in methods:
                methods.remove('geometry')

        elif action == 'solve':
            phuong_phap = request.form.get('method')
            try:
                result = giai_tu_dong(loai_bt, np.array(c), np.array(A), np.array(b), rls, var_types, phuong_phap)
                message = f"Đã giải xong với phương pháp {phuong_phap}!"
            except Exception as e:
                message = f"Lỗi khi giải bài toán: {e}"

            methods = ['geometry', 'simplex', 'bland', 'two_phase']

    else:
        loai_bt = ''
        c_raw = ''
        constraints_raw = ''
        var_bounds_raw = ''

    return render_template('index.html', methods=methods, message=message, result=result,
                           problem_type=loai_bt,
                           objective_coeffs=c_raw,
                           constraints=constraints_raw,
                           variable_bounds=var_bounds_raw)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
