import pandas as pd

def load_data():
    ratings = pd.read_csv("data/ml-100k/u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])
    items = pd.read_csv("data/ml-100k/u.item", sep="|", encoding='latin-1', header=None,
                        names=["movie_id", "title", "release_date", "video_release", "IMDb", "unknown", "Action",
                               "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama",
                               "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller",
                               "War", "Western"])
    return ratings, items

def get_movies_by_genre(items, genre):
    """Get movies that match a single genre"""
    return items[items[genre] == 1]

def get_movies_by_genres(items, genres):
    """Filter movies by multiple genres (AND condition)"""
    filtered = items.copy()
    for genre in genres:
        filtered = filtered[filtered[genre] == 1]
    return filtered

def get_movie_mean_scores(ratings):
    """Get the mean rating score for each movie"""
    return ratings.groupby("item_id")["rating"].mean().to_dict()

def get_movie_title(items, movie_id):
    """Get the title of a movie by its ID"""
    title_matches = items[items["movie_id"] == movie_id]["title"].values
    if len(title_matches) > 0:
        return title_matches[0]
    return None

def get_movie_info(items, movie_id, ratings_dict):
    """Get detailed movie information including score"""
    movie = items[items["movie_id"] == movie_id]
    if movie.empty:
        return None
    
    info = {
        "id": int(movie_id),
        "title": movie["title"].values[0],
        "score": round(ratings_dict.get(movie_id, 0), 2),
        "genres": []
    }
    # Extract all genres for this movie
    genre_columns = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime", 
                     "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", 
                     "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    for genre in genre_columns:
        if movie[genre].values[0] == 1:
            info["genres"].append(genre)
            
    return info