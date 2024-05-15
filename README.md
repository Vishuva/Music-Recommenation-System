pip install numpy pandas scikit-surprise

import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Sample data
data = {
    'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
    'song_id': [101, 102, 103, 101, 102, 104, 101, 103, 104],
    'rating': [5, 3, 2, 4, 2, 5, 5, 4, 3]
}

df = pd.DataFrame(data)

# Convert the dataframe to a Surprise dataset
reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(df[['user_id', 'song_id', 'rating']], reader)

# Train-test split
trainset, testset = train_test_split(dataset, test_size=0.25)

# Use SVD for matrix factorization
model = SVD()

# Train the model
model.fit(trainset)

# Predict ratings for the testset
predictions = model.test(testset)

# Compute and print the RMSE (Root Mean Squared Error)
rmse = accuracy.rmse(predictions)
print(f'RMSE: {rmse}')

# Function to get top-N recommendations for a user
def get_top_n_recommendations(predictions, user_id, n=5):
    # Filter predictions for the given user_id
    user_predictions = [pred for pred in predictions if pred.uid == user_id]
    
    # Sort the predictions by estimated rating
    user_predictions.sort(key=lambda x: x.est, reverse=True)
    
    # Get top-N recommendations
    top_n = user_predictions[:n]
    
    return [(pred.iid, pred.est) for pred in top_n]

# Example usage: Get top-5 song recommendations for user with user_id=1
user_id = 1
top_n_recommendations = get_top_n_recommendations(predictions, user_id, n=5)
print(f'Top-5 recommendations for user {user_id}: {top_n_recommendations}')
