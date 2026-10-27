from flask import Flask, render_template, request, redirect, url_for, session

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # 세션 보안

# 사용자 정보 (실제 DB 없이 하드코딩)
users = {
    'test1': {'password': 'pass1234', 'balance': 1000},
    'test2': {'password': 'pass234', 'balance': 2000}
}

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        userid = request.form['userid']
        password = request.form['password']
        user = users.get(userid)
        if user and user['password'] == password:
            session['userid'] = userid
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error='로그인 실패')
    return render_template('login.html')

@app.route('/main', methods=['GET', 'POST'])
def dashboard():
    if 'userid' not in session:
        return redirect(url_for('login'))

    userid = session['userid']
    user = users[userid]
    message = ''

    if request.method == 'POST':
        action = request.form['action']
        amount = int(request.form['amount'])

        if action == 'deposit':
            user['balance'] += amount
            message = f'{amount}원이 입금되었습니다.'
        elif action == 'withdraw':
            if user['balance'] >= amount:
                user['balance'] -= amount
                message = f'{amount}원이 출금되었습니다.'
            else:
                message = '잔액이 부족합니다.'

    return render_template('main.html', userid=userid, balance=user['balance'], message=message)

@app.route('/logout')
def logout():
    session.pop('userid', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
