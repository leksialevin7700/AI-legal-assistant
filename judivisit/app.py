from flask import Flask, render_template, request
from pymongo import MongoClient

app = Flask(__name__)

# MongoDB connection
app.config["MONGO_URI"] = "mongodb://localhost:27017/legalservices"
client = MongoClient(app.config["MONGO_URI"])
db = client['telelaw']
users_collection = db['lawyers']

# Route to display lawyer listings with search
@app.route('/')
def lawyer_listings():
    specialization = request.args.get('specialization')
    
    # Create the query based on specialization if provided
    query = {"specialization": {"$regex": specialization, "$options": "i"}} if specialization else {}
    
    # Ensure we're projecting all the fields we need, including profile_pic and meet_links
    projection = {
        "name": 1, 
        "email": 1, 
        "specialization": 1, 
        "description": 1, 
        "experience": 1,
        "profile_pic": 1,
        "meet_links": 1
    }
    
    # Find lawyers matching the query
    lawyers = list(users_collection.find(query, projection))
    
    return render_template('lawyers.html', 
        lawyers=lawyers,
        specialization=specialization
    )

if __name__ == '__main__':
    app.run(debug=True, port=5001)