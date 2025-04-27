import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class Particle:
    def __init__(self, position, score_func, candidates):
        self.position = position
        self.velocity = 0.1  # Initial exploration velocity
        self.best_position = position
        self.best_score = score_func(position)
        self.score_func = score_func
        self.candidates = candidates
        
    def update(self, global_best_position, w=0.7, c1=1.5, c2=1.5):
        """
        Update particle position using improved PSO algorithm
        w: inertia weight
        c1: cognitive parameter
        c2: social parameter
        """
        r1, r2 = random.random(), random.random()
        
        # Calculate velocity vector (conceptual in discrete space)
        cognitive = c1 * r1
        social = c2 * r2
        
        # Probability of moving towards global best vs exploring
        p_global = min(0.8, social / (social + cognitive))
        p_local = min(0.6, cognitive / (social + cognitive))
        
        # Choose next position with probability-based approach
        if random.random() < p_global:
            # Move towards global best
            self.position = global_best_position
        elif random.random() < p_local:
            # Move towards personal best
            self.position = self.best_position
        else:
            # Explore new space
            self.position = random.choice(self.candidates)
            
        # Evaluate new position
        score = self.score_func(self.position)
        if score > self.best_score:
            self.best_score = score
            self.best_position = self.position
        
        return score

def pso(candidates, score_func, n_particles=15, n_iterations=25, top_n=15):
    """
    Enhanced PSO algorithm that returns top N recommendations
    
    Args:
        candidates: List of movie IDs to choose from
        score_func: Function to evaluate movie quality
        n_particles: Number of particles in the swarm
        n_iterations: Number of iterations to run
        top_n: Number of top recommendations to return
    
    Returns:
        List of top N movie IDs with highest scores
    """
    if not candidates:
        return []
    
    # If we have fewer candidates than requested top_n, return all sorted by score
    if len(candidates) <= top_n:
        return sorted(candidates, key=score_func, reverse=True)
    
    # Initialize particle swarm with random positions
    particles = [
        Particle(random.choice(candidates), score_func, candidates)
        for _ in range(min(n_particles, len(candidates)))
    ]
    
    # Track global best
    global_best = {"position": particles[0].position, "score": particles[0].best_score}
    
    # Track all discovered positions and scores
    all_positions = {}
    
    # Run PSO iterations
    for i in range(n_iterations):
        # Adaptive inertia weight (decreases over time)
        w = 0.9 - 0.5 * (i / n_iterations)
        
        for particle in particles:
            score = particle.update(global_best["position"], w=w)
            
            # Track all positions and their scores
            all_positions[particle.position] = score
            
            # Update global best if needed
            if score > global_best["score"]:
                global_best["score"] = score
                global_best["position"] = particle.position
    
    # Get top N movies based on discovered scores
    top_movies = sorted(all_positions.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [movie_id for movie_id, _ in top_movies]