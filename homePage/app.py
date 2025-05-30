from flask import Flask, render_template, request, redirect, url_for, flash, session
from pymongo import MongoClient

app = Flask(__name__)
app.secret_key = 'Azsdxcfgvbhnjkml,wsdfghjkl'

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['lawbot_db']
users_collection = db['users']

@app.route('/auth', methods=['GET'])
def auth():
    return render_template('auth.html', title="LawBot - Login & Sign Up")

@app.route('/auth/login', methods=['POST'])
def auth_login():
    email = request.form.get('login-email')
    password = request.form.get('login-password')

    user = users_collection.find_one({'email': email})

    if user and user['password'] == password:  # Directly compare plain-text passwords
        session['email'] = email
        flash('Login successful!', 'success')
        return redirect(url_for('dashboard'))
    else:
        flash('Invalid credentials, please try again.', 'danger')
        return redirect(url_for('auth'))

@app.route('/auth/register', methods=['POST'])
def auth_register():
    full_name = request.form.get('signup-name')
    email = request.form.get('signup-email')
    password = request.form.get('signup-password')
    confirm = request.form.get('signup-confirm')

    if password != confirm:
        flash('Passwords do not match!', 'danger')
        return redirect(url_for('auth'))

    if users_collection.find_one({'email': email}):
        flash('Email already exists. Choose another.', 'warning')
        return redirect(url_for('auth'))

    # Store password directly (no hashing)
    users_collection.insert_one({
        'full_name': full_name,
        'email': email,
        'password': password  # Storing password as plain text
    })
    
    flash('Registration successful! Please log in.', 'success')
    return redirect(url_for('auth'))

@app.route('/')
def dashboard():
    email = session.get('email')
    return render_template('dashboard.html', email=email)

@app.route('/logout')
def logout():
    session.pop('email', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('auth'))

@app.route("/chat", methods=["POST"])
def chat_login():
    email = request.form.get("login-email")

    user = users_collection.find_one({'email': email})

    if user:  # Only check if the email exists
        session['email'] = email
        return redirect("http://127.0.0.1:3000/")  # Redirect to chat app
    else:
        flash("Invalid email", "danger")
        return redirect(url_for('auth'))  # Redirect to login page

if __name__ == '__main__':
    app.run(debug=True)
