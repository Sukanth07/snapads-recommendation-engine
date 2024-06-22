from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import hashlib

app = Flask(__name__)

boards_data = None
user_preferences = None

# Function to perform hash encoding
def hash_encode(value):
    return int(hashlib.md5(value.encode()).hexdigest(), 16) % 10**8  # Limiting the hash to 8 digits for simplicity

@app.route('/')
def hello_world():
    return 'Hello World!'

# Endpoint to receive JSON data from frontend
@app.route('/recommend_boards', methods=['POST'])
def recommend_boards():
    global boards_data, user_preferences
    
    # Receive JSON data from frontend
    request_data = request.get_json()
    
    # Extract hoarding board dataset and user preferences
    try:
        boards_data = pd.DataFrame(request_data['boards'])
        user_preferences = request_data['user_preferences']
    
        X = boards_data.copy()
        X = X.drop(columns=['board_id','board_name'])
        
        # Hash encoding for categorical variables
        X['state'] = X['state'].apply(hash_encode)
        X['city'] = X['city'].apply(hash_encode)
        
        # Convert the DataFrame to numpy array
        X_values = X.values
        
        # Convert user_preferences JSON to Python dictionary
        user_pref_encoded = [
            hash_encode(user_preferences['state']),
            hash_encode(user_preferences['city']),
            user_preferences['banner_height'],
            user_preferences['banner_width'],
            user_preferences['price'],
            user_preferences['light_type']
        ]
        user_pref_array = np.array(user_pref_encoded).reshape(1,-1)

        # Perform similarity calculation
        similarity_matrix = cosine_similarity(user_pref_array, X)
        
        # Get the indices sorted by similarity (descending order)
        sorted_indices = similarity_matrix.argsort()[0][::-1]
        
        # Get the sorted board_ids based on similarity
        sorted_board_ids = boards_data.iloc[sorted_indices]['board_id'].tolist()
        
        # Prepare response as JSON
        response = {
            "sorted_board_ids": sorted_board_ids
        }
        
        # Return JSON response
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
