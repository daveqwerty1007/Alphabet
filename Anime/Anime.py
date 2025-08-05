import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Load datasets
animelist = pd.read_csv('/Users/zhaojianbo/Desktop/Alphabet/Week2/anime/animelist.csv')
anime = pd.read_csv('/Users/zhaojianbo/Desktop/Alphabet/Week2/anime/anime.csv')

# preping data
merged_data = pd.merge(animelist, anime, left_on='anime_id', right_on='MAL_ID')
user_anime_matrix = merged_data.pivot_table(index='user_id', columns='Name', values='score').fillna(0)

# Fit KNN model
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(user_anime_matrix.values)

# Find the nearest neighbors for each anime
distances, indices = knn.kneighbors(user_anime_matrix.values, n_neighbors=6)

# Function to predict ratings
def predict_ratings(user_index, movie_index, indices, distances, user_anime_matrix):
    sim_movies = indices[movie_index].tolist()
    movie_distances = distances[movie_index].tolist()
    
    # Exclude the movie itself from the similarity list
    if movie_index in sim_movies:
        id_movie = sim_movies.index(movie_index)
        sim_movies.remove(movie_index)
        movie_distances.pop(id_movie)
    
    movie_similarity = [1 - x for x in movie_distances]
    movie_similarity_copy = movie_similarity.copy()
    nominator = 0
    
    for s in range(0, len(movie_similarity)):
        if user_anime_matrix.iloc[user_index, sim_movies[s]] == 0:
            movie_similarity_copy.pop(s)
        else:
            nominator += movie_similarity[s] * user_anime_matrix.iloc[user_index, sim_movies[s]]
    
    if len(movie_similarity_copy) > 0:
        predicted_rating = nominator / sum(movie_similarity_copy)
    else:
        predicted_rating = 0
    
    return predicted_rating

# Predict ratings for given anime matrix
user_index = 0 
predicted_ratings = []
for i in range(user_anime_matrix.shape[1]):
    if user_anime_matrix.iloc[user_index, i] == 0:
        predicted_ratings.append((user_anime_matrix.columns[i], 
                                  predict_ratings(user_index, i, indices, distances, user_anime_matrix)))

# Get top 5 recommendations
recommendations = sorted(predicted_ratings, key=lambda x: x[1], reverse=True)[:5]

# Create the recommendation dataframe
recommendation_df = pd.DataFrame(recommendations, columns=['Name', 'Predicted Score'])
recommendation_df = recommendation_df.merge(anime[['Name', 'MAL_ID', 'Score', 'Type', 'Source', 'synopsis']], on='Name')

print(recommendation_df)
