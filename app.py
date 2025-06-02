import numpy as np
from flask import Flask, render_template, request, jsonify
from qhtt import giai_tu_dong  # Giữ nguyên hàm giải của bạn

app = Flask(__name__)

def get_valid_methods(n, b_std):
    valid_choices = []

    if n == 2:
        if np.any(b_std < -1e-9):
            valid_choices = [1, 4]
        elif np.any(np.abs(b_std) < 1e-9):
            valid_choices = [1, 3]
        else:
            valid_choices = [1, 2, 3]
    else:
        if np.any(b_std < -1e-9):
            valid_choices = [4]
        elif np.any(np.abs(b_std) < 1e-9):
            valid_choices = [3]
        else:
            valid_choices = [2, 3]

    pp_dict = {
        1: "Phương pháp hình học",
        2: "Phương pháp đơn hình",
        3: "Phương pháp Bland",
        4: "Phương pháp hai pha"
    }

    return [{'id': str(i), 'name': pp_dict[i]} for i in valid_choices]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_methods', methods=['POST'])
def get_methods():
    data = request.json
    num_vars = int(data.get('num_vars', 0))
    num_cons = int(data.get('num_cons', 0))

    b_list = data.get('b', None)
    if b_list is None or len(b_list) != num_cons:
        # Nếu không có hoặc sai, giả sử b dương (1.0)
        b_std = np.ones(num_cons)
    else:
        try:
            b_std = np.array([float(x) for x in b_list])
        except:
            b_std = np.ones(num_cons)

    methods = get_valid_methods(num_vars, b_std)
    return jsonify({'methods': methods})

@app.route('/solve', methods=['POST'])
def solve():
    data = request.json
    loai_bt = data.get('loai_bt')
    num_vars = int(data.get('num_vars', 0))
    num_cons = int(data.get('num_cons', 0))
    c = list(map(float, data.get('c', '').strip().split()))

    A = []
    rls = []
    b = []
    constraints = data.get('constraints', [])
    for constr in constraints:
        parts = constr.strip().split()
        A.append(list(map(float, parts[:num_vars])))
        rls.append(parts[num_vars])
        b.append(float(parts[num_vars + 1]))

    var_types = data.get('var_types', '').strip().split()
    phuong_phap = data.get('phuong_phap')

    ket_qua = giai_tu_dong(loai_bt, c, A, b, rls, var_types, phuong_phap)
    return jsonify({'ket_qua': ket_qua})

if __name__ == '__main__':
    app.run(debug=True)
