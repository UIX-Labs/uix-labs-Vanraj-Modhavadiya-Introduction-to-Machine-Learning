from recommenders import data_movies, data_ratings, default_poster_url
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from recommenders.content_based import matching_score

def collaborative_recommendation(movie):
    # Copy the data_movies dataframe
    movies = data_movies.copy()
    
    # Function to get title from index
    def get_title_from_index(index):
        return movies[movies.index == index]['title'].values[0]
    
    # Function to convert title to index
    def get_index_from_title(title):
        return movies[movies.title == title].index.values[0]
    
    # Function to return the most similar title to the input
    def find_closest_title(title):
        leven_scores = list(enumerate(movies['title'].apply(matching_score, b=title)))
        sorted_leven_scores = sorted(leven_scores, key=lambda x: x[1], reverse=True)
        closest_title = get_title_from_index(sorted_leven_scores[0][0])
        distance_score = sorted_leven_scores[0][1]
        return closest_title, distance_score

    # Find the closest title and its distance score
    closest_title, distance_score = find_closest_title(movie)
    response = []
    
    # Construct the title string based on whether there is a misspelling
    if distance_score == 100:
        title_string = f"Here\'s the list of movies similar to {str(closest_title)} \n"
    else:
        title_string = f'Did you mean {str(closest_title)} ?\n Here\'s the list of movies similar to {str(closest_title)} \n'

    # Combine the data_ratings and movies using pivot table => movies_df
    movies_df = pd.merge(data_ratings, movies, on='movieId').pivot_table(index='title', columns='userId', values='rating').fillna(0)
    
    # Convert the pivot table to a matrix
    movies_df_matrix = csr_matrix(movies_df.values)
    
    # Build the model
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    
    # Fit the model
    model_knn.fit(movies_df_matrix)
    
    # Get the index of the closest title
    index_value = movies_df.index.get_loc(closest_title)
    
    # Find the movies related to the closest title
    distances, indices = model_knn.kneighbors(movies_df.iloc[index_value,:].values.reshape(1,-1), n_neighbors=21)

    sorted_indices = np.argsort(distances.flatten())[::-1]
    sorted_distances = distances.flatten()[sorted_indices]
    sorted_neighbor_indices = indices.flatten()[sorted_indices]
    
    response = []
    
    # Loop through the nearest neighbors and append movie recommendations to the response
    for i in range(0, len(sorted_distances)):
        if i == 0:
            pass
        else:
            title = movies_df.index[sorted_neighbor_indices[i]]
            movie_record = data_movies[data_movies.title == title].iloc[0]
            movie_poster = str(movie_record.poster_link)
            response.append({
                "movieId": int(movie_record.movieId),
                "title": str(title),
                "image": movie_poster if movie_poster != "nan" else default_poster_url,
                "genres": str(movie_record.genres).split("|")
            })

    return {"status": True, "data": {"message": title_string, "results": response}}
