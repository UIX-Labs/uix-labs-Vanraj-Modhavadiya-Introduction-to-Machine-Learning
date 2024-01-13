import os
from datetime import datetime

from dotenv import load_dotenv
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from recommenders.popularity_recommender import recommend_popular_movies
from recommenders.content_based import content_based_recommendation
from recommenders.collaborative_filtering import collaborative_recommendation
from urllib.parse import unquote, quote

app = Flask(__name__)


@app.route('/')
def index():
    data = "Welcome to Instalearn ML Tutorial students!"
    return {"message": data}


@app.route('/recommend/popular')
def popular_movies():
    data = recommend_popular_movies()
    return data


@app.route('/html/popular')
def popular():
    movies = recommend_popular_movies()
    return render_template('popular.html', movies=movies['data']['results'])


@app.route('/html/content_based')
def content_based_html():
    if request.method == "GET":
        movie_name = request.args.get('movie_name', default="", type=str)
        movie_name = unquote(movie_name)
        data = content_based_recommendation(movie_name)
        return render_template('content_based.html', movies=data['data']['results'])


@app.route('/html/collaborative')
def collaborative_html():
    if request.method == "GET":
        movie_name = request.args.get('movie_name', default="", type=str)
        movie_name = unquote(movie_name)
        data = collaborative_recommendation(movie_name)
        return render_template('collaborative.html', movies=data['data']['results'])


@app.route('/recommend/content_based', methods=['GET'])
def content_based():
    if request.method == "GET":
        movie_name = request.args.get('movie_name', default="", type=str)
        movie_name = unquote(movie_name)
        data = content_based_recommendation(movie_name)
        return data


@app.route('/recommend/collaborative', methods=['GET'])
def collaborative_recommend():
    if request.method == "GET":
        movie_name = request.args.get('movie_name', default="", type=str)
        movie_name = unquote(movie_name)
        data = collaborative_recommendation(movie_name)
        return data


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=0)
