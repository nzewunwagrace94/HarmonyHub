# recommender.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Dummy user-song interaction matrix
user_song_matrix = np.array([
    [1, 0, 0, 1, 1],
    [0, 1, 1, 0, 0],
    [1, 1, 0, 0, 1],
    [0, 0, 1, 1, 0]
])

def compute_user_similarity(user_song_matrix):
    """
    Computes the cosine similarity between users based on their song preferences.

    Parameters:
    - user_song_matrix (numpy array): Matrix of user-song interactions.

    Returns:
    - user_similarity (numpy array): Matrix of user similarities.
    """
    return cosine_similarity(user_song_matrix)

def recommend_songs(user_id, user_song_matrix, user_similarity, num_recommendations=5):
    """
    Recommends songs for a given user based on the similarity of their preferences with other users.

    Parameters:
    - user_id (int): The ID of the user for whom to recommend songs.
    - user_song_matrix (numpy array): Matrix of user-song interactions.
    - user_similarity (numpy array): Matrix of user similarities.
    - num_recommendations (int): Number of song recommendations to return. Default is 5.

    Returns:
    - recommended_songs (numpy array): Array of recommended song indices.
    """
    user_index = user_id
    similar_users = user_similarity[user_index]
    
    # Calculate song scores by multiplying the song interaction matrix with the similarity scores
    song_scores = user_song_matrix.T.dot(similar_users)
    recommended_songs = np.argsort(song_scores)[::-1]
    
    return recommended_songs[:num_recommendations]

if __name__ == "__main__":
    user_id = 0  # Example user ID for whom to recommend songs

    # Compute user similarity matrix
    user_similarity = compute_user_similarity(user_song_matrix)

    # Get recommended songs for the example user
    recommended_songs = recommend_songs(user_id, user_song_matrix, user_similarity)

    print(f"Recommended songs for user {user_id}: {recommended_songs}")
