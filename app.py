from flask import Flask, render_template, request
import numpy as np
from qhtt import giai_tu_dong

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/solve', methods=['POST'])
def solve():
    try:
        loai_bt = request.form['loai']
        phuong_phap = request.form['phuong_phap']
        c = list(map(float, request.form['c'].split()))
        A = [list(map(float, row.split())) for row in request.form['A'].strip().split('\n')]
        b = list(map(float, request.form['b'].split()))
        rls = request.form['rls'].strip().split('\n')
        var_types = request.form['var_types'].strip().split()

        result = giai_tu_dong(loai_bt, np.array(c), np.array(A), np.array(b), rls, var_types, phuong_phap)

        return render_template('index.html', message="Đã giải xong! Xem kết quả trên terminal hoặc biểu đồ.")
    except Exception as e:
        return render_template('index.html', message=f"Lỗi: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
