import pandas as pd
import numpy as np
import os

#print(os.getcwd())

default_poster_url = "https://gurukul-be.s3.ap-south-1.amazonaws.com/0c3df9f4dc56471d88bb88255bb79548.jpg"

####################################################################################################################################

'''
Complete the code below

Things to do: load csv in dataframes

data_ratings => file path : 'data_files/ratings.csv'
data_movies => file path : 'data_files/movies.csv'
data_posters => file path : 'data_files/movie_poster.csv'

'''
data_ratings =None # empty df
data_movies = None # empty df
data_posters = None # empty df


'''
Uncomment the below line after you complete the code above.
'''
#data_movies = pd.merge(data_movies, data_posters, how="left", on="movieId")

#####################################################################################################################################


