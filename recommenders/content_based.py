from recommenders import data_movies, default_poster_url
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz

def matching_score(a, b):
    return fuzz.ratio(a, b)

def extract_title(title):
    year = title[-5:-1]
    if year.isnumeric():
        title_no_year = title[:-7].strip()
        return title_no_year
    else:
        return title.strip()

def extract_year(title):
    year = title[-5:-1]
    if year.isnumeric():
        return int(year)
    else:
        return np.nan

def content_based_recommendation(movie):
    movies = data_movies.copy()
    movies.rename(columns={'title': 'title_year'}, inplace=True)
    movies['title_year'] = movies['title_year'].str.strip()
    movies['title'] = movies['title_year'].apply(extract_title)
    movies['year'] = movies['title_year'].apply(extract_year)

    movies = movies[~(movies['genres'] == '(no genres listed)')].reset_index(drop=True)
    movies['genres'] = movies['genres'].str.replace('|', ' ')
    movies['genres'] = movies['genres'].str.replace('Sci-Fi', 'SciFi')
    movies['genres'] = movies['genres'].str.replace('Film-Noir', 'Noir')

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(movies['genres'])

    sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

    def get_title_year_from_index(index):
        return movies.loc[index, 'title_year']

    def get_title_from_index(index):
        return movies.loc[index, 'title']

    def get_index_from_title(title):
        return movies[movies['title'] == title].index[0]

    def find_closest_title(title):
        leven_scores = list(enumerate(movies['title'].apply(matching_score, b=title)))
        sorted_leven_scores = sorted(leven_scores, key=lambda x: x[1], reverse=True)
        closest_title = get_title_from_index(sorted_leven_scores[0][0])
        distance_score = sorted_leven_scores[0][1]
        return closest_title, distance_score

    def contents_based_recommender(movie_user_likes, how_many):
        closest_title, distance_score = find_closest_title(movie_user_likes)
        response = []
        if distance_score == 100:
            title_string = f"Here's the list of movies similar to {str(movie_user_likes)} \n"
            movie_index = get_index_from_title(closest_title)
            movie_list = list(enumerate(sim_matrix[movie_index]))
            similar_movies = sorted(movie_list, key=lambda x: x[1], reverse=True)[1:how_many+1]

            for i, s in similar_movies:
                title = get_title_year_from_index(i)
                movie_record = data_movies[data_movies['title'] == title].iloc[0]
                response.append({
                    "movieId": int(movie_record['movieId']),
                    "title": str(title),
                    "genres": str(movie_record['genres']).split("|")
                })
        else:
            title_string = f'Did you mean {str(closest_title)} ?\n Here\'s the list of movies similar to {str(closest_title)} \n'
            movie_index = get_index_from_title(closest_title)
            movie_list = list(enumerate(sim_matrix[movie_index]))
            similar_movies = sorted(movie_list, key=lambda x: x[1], reverse=True)[:how_many]

            for i, s in similar_movies:
                title = get_title_year_from_index(i)
                movie_record = data_movies[data_movies['title'] == title].iloc[0]
                movie_poster = str(movie_record['poster_link'])
                response.append({
                    "movieId": int(movie_record['movieId']),
                    "title": str(title),
                    "image": movie_poster if movie_poster != "nan" else default_poster_url,
                    "genres": str(movie_record['genres']).split("|")
                })
        return {'message': title_string, "results": response}

    data = contents_based_recommender(movie_user_likes=movie, how_many=20)
    return {"status": True, "data": data}
