from recommenders import data_movies, default_poster_url
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
# create a function to find the closest title
def matching_score(a,b):
   return fuzz.ratio(a,b)

def extract_title(title):
   
   year = title[len(title)-5:len(title)-1]
   
   # some movies do not have the info about year in the column title. So, we should take care of the case as well.
   
   if year.isnumeric():
      title_no_year = title[:len(title)-7]
      return title_no_year
   else:
      return title
   
   
# the function to extract years
def extract_year(title):
   
   year = title[len(title)-5:len(title)-1]
   # some movies do not have the info about year in the column title. So, we should take care of the case as well.
   if year.isnumeric():
      return int(year)
   else:
      return np.nan

    
def content_based_recommendation(movie):
    # the function to extract titles 
    movies = data_movies.copy()
    # change the column name from title to title_year
    movies.rename(columns={'title':'title_year'}, inplace=True) 
    # remove leading and ending whitespaces in title_year
    movies['title_year'] = movies['title_year'].apply(lambda x: x.strip()) 
    # create the columns for title and year
    movies['title'] = movies['title_year'].apply(extract_title) 
    movies['year'] = movies['title_year'].apply(extract_year) 
    
    #r,c = movies[movies['genres']=='(no genres listed)'].shape
    # remove the movies without genre information and reset the index
    movies = movies[~(movies['genres']=='(no genres listed)')].reset_index(drop=True)

    movies['genres'] = movies['genres'].str.replace('|',' ')

    # change 'Sci-Fi' to 'SciFi' and 'Film-Noir' to 'Noir'
    movies['genres'] = movies['genres'].str.replace('Sci-Fi','SciFi')
    movies['genres'] = movies['genres'].str.replace('Film-Noir','Noir')

    ############################################################################################
    '''
    complete the code below:
    '''


    # create an object for TfidfVectorizer: tfidf_vector
    tfidf_vector = None
    # apply the object to the genres column: tfidf_matrix
    tfidf_matrix = None
    # create the cosine similarity matrix: sim_matrix
    sim_matrix = None
    ############################################################################################
    # a function to convert index to title_year
    def get_title_year_from_index(index):
        return movies[movies.index == index]['title_year'].values[0]
    # a function to convert index to title
    def get_title_from_index(index):
        return movies[movies.index == index]['title'].values[0]
    # a function to convert title to index
    def get_index_from_title(title):
        return movies[movies.title == title].index.values[0]
    
    # a function to return the most similar title to the words a user type
    def find_closest_title(title):
        leven_scores = list(enumerate(movies['title'].apply(matching_score, b=title)))
        sorted_leven_scores = sorted(leven_scores, key=lambda x: x[1], reverse=True)
        closest_title = get_title_from_index(sorted_leven_scores[0][0])
        distance_score = sorted_leven_scores[0][1]
        return closest_title, distance_score

    def contents_based_recommender(movie_user_likes, how_many):
        closest_title, distance_score = find_closest_title(movie_user_likes)
        response = []
        # When a user does not make misspellings
        if distance_score == 100:
            
            title_string = f"Here\'s the list of movies similar to {str(movie_user_likes)} \n"

            movie_index = get_index_from_title(closest_title)
            movie_list = list(enumerate(sim_matrix[int(movie_index)]))
            # remove the typed movie itself
            similar_movies = list(filter(lambda x:x[0] != int(movie_index), sorted(movie_list,key=lambda x:x[1], reverse=True))) 
            
            #print('Here\'s the list of movies similar to '+'\033[1m'+str(closest_title)+'\033[0m'+'.\n')

            for i,s in similar_movies[:how_many]:
                title = get_title_year_from_index(i)
                movie_record = data_movies[data_movies.title == title].iloc[0]
                response.append({
                                    "movieId": int(movie_record.movieId),
                                    "title": str(title),
                                    "genres": str(movie_record.genres).split("|")
                                })

        # When a user makes misspellings    
        else:
            title_string =  f'Did you mean {str(closest_title)} ?\n Here\'s the list of movies similar to {str(closest_title)} \n'
            movie_index = get_index_from_title(closest_title)
            movie_list = list(enumerate(sim_matrix[int(movie_index)]))
            similar_movies = list(filter(lambda x:x[0] != int(movie_index), sorted(movie_list,key=lambda x:x[1], reverse=True)))
            #print('Here\'s the list of movies similar to '+'\033[1m'+str(closest_title)+'\033[0m'+'.\n')
            
            for i,s in similar_movies[:how_many]:
                title = get_title_year_from_index(i)
                movie_record = data_movies[data_movies.title == title].iloc[0]
                movie_poster = str(movie_record.poster_link)
                response.append({
                                    "movieId": int(movie_record.movieId),
                                    "title": str(title),
                                    "image": movie_poster if movie_poster != "nan" else default_poster_url,
                                    "genres": str(movie_record.genres).split("|")
                                })
        return {'message': title_string, "results": response}

    data = contents_based_recommender(movie_user_likes=movie, how_many=20)
    return {"status": True, "data": data}