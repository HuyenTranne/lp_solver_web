from flask import Flask, render_template, request, jsonify
from qhtt import giai_tu_dong, xet_phuong_phap

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_methods', methods=['POST'])
def get_methods():
    data = request.json
    num_vars = int(data.get('num_vars', 0))
    num_cons = int(data.get('num_cons', 0))
    methods = xet_phuong_phap(num_vars, num_cons)  # giả sử trả list tên phương pháp
    
    # Tạo danh sách dict gồm id và tên để frontend dễ chọn
    methods_list = [{'id': i+1, 'name': name} for i, name in enumerate(methods)]
    
    return jsonify({'methods': methods_list})

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
    phuong_phap = data.get('phuong_phap')  # id phương pháp dạng số hoặc tên
    
    # Gọi giai_tu_dong đúng tham số (sửa lại thứ tự tham số và thêm phuong_phap cuối)
    ket_qua = giai_tu_dong(loai_bt, c, A, b, rls, num_vars, var_types, phuong_phap)
    
    return jsonify({'ket_qua': ket_qua})

if __name__ == '__main__':
    app.run(debug=True)
