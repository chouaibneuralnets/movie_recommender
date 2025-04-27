import streamlit as st
from recommender import load_data, get_movies_by_genres, get_movie_mean_scores, get_movie_info
from pso_code import pso
import pandas as pd

st.set_page_config(page_title="Movie Recommender PSO", layout="wide")

st.title("🎬 Movie Recommender using PSO")
st.markdown("Choisissez jusqu'à **5 genres de films** et obtenez des recommandations optimisées avec PSO.")

# Load data
@st.cache_data
def get_cached_data():
    return load_data()

ratings, items = get_cached_data()

# Extract available genres in the dataset
genre_columns = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

# Sidebar for filters
with st.sidebar:
    st.header("Filtres")
    
    # Multi-select for genres (limited to 5)
    selected_genres = st.multiselect(
        "Choisir jusqu'à 5 genres :",
        options=genre_columns,
        max_selections=5,
        default=["Action"]
    )
    
    # Min rating slider
    min_rating = st.slider(
        "Note minimale :",
        min_value=0.0,
        max_value=5.0,
        value=3.0,
        step=0.1
    )
    
    # Number of recommendations
    num_recommendations = st.slider(
        "Nombre de recommandations :",
        min_value=1,
        max_value=15,
        value=15
    )
    
    # Advanced settings
    with st.expander("Paramètres avancés"):
        n_particles = st.slider("Nombre de particules PSO", 5, 30, 15)
        n_iterations = st.slider("Nombre d'itérations PSO", 10, 50, 25)

# Main content
if not selected_genres:
    st.warning("Veuillez sélectionner au moins un genre.")
else:
    # Filter movies by selected genres
    filtered_movies = get_movies_by_genres(items, selected_genres)
    
    # Calculate mean scores for movies
    movie_scores = get_movie_mean_scores(ratings)
    
    # Filter by minimum rating
    filtered_movie_ids = [
        movie_id for movie_id in filtered_movies['movie_id'] 
        if movie_scores.get(movie_id, 0) >= min_rating
    ]
    
    filtered_movies = filtered_movies[filtered_movies['movie_id'].isin(filtered_movie_ids)]
    
    # Create list of candidate movie IDs
    candidates = list(filtered_movies["movie_id"])
    
    if not candidates:
        st.error(f"Aucun film trouvé pour les genres {', '.join(selected_genres)} avec une note minimale de {min_rating}.")
    else:
        st.info(f"Trouvé {len(candidates)} films correspondant à vos critères.")
        
        # Score function to evaluate a movie
        def score_func(movie_id):
            return movie_scores.get(movie_id, 0)
        
        # Run PSO to find top recommendations
        with st.spinner("Recherche des meilleures recommandations..."):
            top_movie_ids = pso(
                candidates, 
                score_func, 
                n_particles=n_particles, 
                n_iterations=n_iterations,
                top_n=num_recommendations
            )
        
        # Display recommendations
        if not top_movie_ids:
            st.error("Désolé, aucun film n'a été trouvé pour la recommandation.")
        else:
            st.success(f"🎉 Voici les {len(top_movie_ids)} meilleures recommandations de films :")
            
            # Create a list of movie info for display
            movies_info = []
            for movie_id in top_movie_ids:
                info = get_movie_info(items, movie_id, movie_scores)
                if info:
                    movies_info.append(info)
            
            # Display as a DataFrame for better formatting
            if movies_info:
                # Create a DataFrame for display
                df = pd.DataFrame([
                    {
                        "Titre": info["title"],
                        "Note": f"{info['score']:.2f} / 5.0",
                        "Genres": ", ".join(info["genres"])
                    }
                    for info in movies_info
                ])
                
                st.dataframe(
                    df,
                    use_container_width=True, 
                    hide_index=True
                )
            
            # Individual movie cards
            st.header("Détails des films recommandés")
            
            # Create 3 columns for movie cards
            cols = st.columns(3)
            
            for i, movie in enumerate(movies_info):
                with cols[i % 3]:
                    with st.container(border=True):
                        st.subheader(movie["title"])
                        st.metric("Note", f"{movie['score']:.2f}/5.0")
                        st.caption(f"Genres: {', '.join(movie['genres'])}")