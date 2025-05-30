from flask import Flask, flash, redirect, render_template, request, session, url_for
from flask_bcrypt import Bcrypt
from pymongo import MongoClient
from flask_session import Session
import random
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# MongoDB Configuration
app.config['MONGO_URI'] = 'mongodb://localhost:27017/telelaw'
app.config['SESSION_TYPE'] = 'mongodb'
app.config['SESSION_MONGODB'] = MongoClient(app.config['MONGO_URI'])
app.config['SESSION_MONGODB_DB'] = 'telelaw'
app.config['SESSION_MONGODB_COLLECTION'] = 'sessions'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SECRET_KEY'] = 'uegahhsgjrhriuwt4u3'

# Initialize MongoDB session
Session(app)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads/profile_pics'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure the upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# MongoDB Connection
client = MongoClient(app.config["MONGO_URI"])
db = client['telelaw']
lawyers_collection = db['lawyers']
bcrypt = Bcrypt(app)

# Sample Meet Links
MEET_LINKS = [
    "https://meet.google.com/efw-nrnb-efm",
    "https://meet.google.com/puo-hsqy-mda",
    "https://meet.google.com/puo-hsqy-mda",
    "https://meet.google.com/vdm-pcrv-duf",
    "https://meet.google.com/phc-jfxq-gft"
]

# Helper function for file validation
def allowed_file(filename):
    """Check if the uploaded file is an allowed image format."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Home Route
@app.route('/')
def home():
    return render_template('home.html')


# Signup Route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')
        specialization = request.form['specialization']
        description = request.form['description']
        experience = request.form['experience']

        # Check if email already exists
        if lawyers_collection.find_one({'email': email}):
            flash('Email already registered. Please log in.', 'warning')
            return redirect(url_for('login'))

        # Save lawyer details to MongoDB
        lawyers_collection.insert_one({
            'name': name,
            'email': email,
            'password': password,
            'specialization': specialization,
            'description': description,
            'experience': experience,
            'meet_links': [],
            'total_consultations': 0,
            'client_rating': 'N/A',
            'active_days': 0,
            'profile_pic': None
        })

        flash('Signup successful! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')


# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        lawyer = lawyers_collection.find_one({'email': email})

        if lawyer and bcrypt.check_password_hash(lawyer['password'], password):
            session['user_id'] = str(lawyer['_id'])
            session['email'] = lawyer['email']
            flash("Login successful!", "success")
            return redirect(url_for('profile'))
        else:
            flash("Invalid credentials. Try again.", "danger")

    return render_template('login.html')


# Lawyer Profile Page
@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'email' not in session:
        flash("Please log in first.", "warning")
        return redirect(url_for('login'))

    lawyer = lawyers_collection.find_one({'email': session['email']})

    if request.method == 'POST':
        updated_specialization = request.form['specialization']
        updated_description = request.form['description']
        updated_experience = request.form['experience']

        update_data = {
            'specialization': updated_specialization,
            'description': updated_description,
            'experience': updated_experience
        }

        # Handle profile picture upload
        if 'profile_pic' in request.files:
            file = request.files['profile_pic']
            if file and file.filename != '' and allowed_file(file.filename):
                filename = secure_filename(f"{session['email']}_{file.filename}")
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

                # Save the profile picture
                file.save(file_path)

                # Update the lawyer's profile picture in MongoDB
                update_data['profile_pic'] = f"/static/uploads/profile_pics/{filename}"

        # Update lawyer profile
        lawyers_collection.update_one(
            {'email': session['email']},
            {'$set': update_data}
        )

        flash('Profile updated successfully!', 'success')
        return redirect(url_for('profile'))

    # Initialize meet links if missing
    if 'meet_links' not in lawyer:
        lawyers_collection.update_one(
            {'email': session['email']},
            {'$set': {'meet_links': []}}
        )
        lawyer = lawyers_collection.find_one({'email': session['email']})

    return render_template('profile.html', lawyer=lawyer)


# Add Meet Link
@app.route('/add_meet_link', methods=['POST'])
def add_meet_link():
    if 'email' not in session:
        flash("Please log in first.", "warning")
        return redirect(url_for('login'))

    # Fetch lawyer and existing links
    lawyer = lawyers_collection.find_one({'email': session['email']})
    existing_links = lawyer.get('meet_links', [])

    # Select a random Meet link
    available_links = [link for link in MEET_LINKS if link not in existing_links]

    if available_links:
        random_link = random.choice(available_links)

        # Add the new link
        lawyers_collection.update_one(
            {'email': session['email']},
            {'$push': {'meet_links': random_link}}
        )

        flash('Google Meet link added to your profile!', 'success')
    else:
        flash('No more available Meet links.', 'warning')

    return redirect(url_for('profile'))


# Remove Meet Link
@app.route('/remove_meet_link/<path:link>', methods=['POST'])
def remove_meet_link(link):
    if 'email' not in session:
        flash("Please log in first.", "warning")
        return redirect(url_for('login'))

    # Remove the specified link
    lawyers_collection.update_one(
        {'email': session['email']},
        {'$pull': {'meet_links': link}}
    )

    flash('Google Meet link removed from your profile!', 'success')
    return redirect(url_for('profile'))


# Logout Route
@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out successfully!", "info")
    return redirect(url_for('home'))


# Run the Flask server
if __name__ == '__main__':
    app.run(debug=True, port=3002)
