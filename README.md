# analogdots-shoe-recommendation-assignment
AnalogDots Shoe Recommendation System
Project Overview
This project implements a shoe recommendation system and personalized services for AnalogDots' Machine Learning Engineer/Data Scientist role assessment. It uses a hybrid recommendation algorithm combining collaborative and content-based filtering, implemented with LightFM. The system includes personalized services (care notifications, replacement suggestions) and an extra outfit/event recommendation feature. A Streamlit app provides an interactive interface, and a PostgreSQL schema ensures scalable data storage. The dataset includes 50 users, 100 shoes, 500 interactions, and 250 care records.

Recommendation Algorithm
Algorithm: Hybrid (Collaborative + Content-Based) using LightFM with logistic loss.
Rationale: Combines user interaction patterns (e.g., views, purchases) with shoe attributes (e.g., type, color, material) to provide robust recommendations, effective for sparse data and cold-start scenarios.
Implementation: Data is preprocessed into user-item interaction matrices and item feature embeddings. The model is trained on synthetic data and outputs top-5 recommendations per user.


Personalized Service Logic
Proactive Care Notifications:
Logic: Triggers if a shoe hasn’t been cleaned in 30 days and has high usage frequency (>10 interactions) under rainy or humid conditions.
Example: "Time to clean your Nike Air 100!"

Shoe Replacement Suggestions:
Logic: Suggests replacements based on usage duration exceeding type-specific lifespans (6 months for running shoes, 12 for sneakers, 18 for dress shoes) with high wear frequency.
Example: "Consider replacing your Adidas Run 200."

Outfit/Event Recommendations (Extra Feature):
Logic: Maps event types (e.g., Formal to dress shoes, Workout to running shoes) and uses LightFM to recommend suitable shoes.
Example: Recommends dress shoes for a formal event.


Database Schema
Tables:
users: Stores user profiles (user_id, age, gender, typical_usage).
shoe_catalog: Stores shoe catalog (shoe_id, brand, model, type, color, size, material, care_requirements).
user_interactions: Logs user interactions (interaction_id, user_id, shoe_id, interaction_type, interaction_timestamp).
care_history: Tracks shoe care records (care_id, user_id, shoe_id, care_type, care_timestamp).
recommendation_logs: Logs recommendation outputs for A/B testing.


Design Logic:
Primary and foreign keys ensure data integrity.
Primary key constraints on user_id and shoe_id provide implicit indexing for efficient queries.
Scalable for large datasets with efficient joins.
Note: The generated dataset omits some schema fields (e.g., age, gender, size, care_requirements) for simplicity.

Synthetic Dataset
Structure:
users.csv: user_id, typical_usage, preferred_color.
shoes.csv: shoe_id, brand, model, type, color, material.
interactions.csv: user_id, shoe_id, interaction_type, interaction_timestamp.
care_history.csv: user_id, shoe_id, care_type, care_timestamp.


Simulation: Generated using faker for realistic user profiles and shoe data, and numpy.random for interactions, mimicking real-world e-commerce data (50 users, 100 shoes, 500 interactions, 250 care records).

Setup Instructions
Clone the repository:git clone https://github.com/aarushim23/analogdots-shoe-recommendation-assignment.git

Create a virtual environment (recommended):python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts Elias

Install dependencies:pip install -r requirements.txt

Generate synthetic data:python generate_data.py

Run the Streamlit app:streamlit run recommendation_system/app.py

View the database schema in schema.sql.


Running the System
Open the Streamlit app 
Navigate tabs to:
Select a user ID for personalized recommendations.
View care notifications and replacement suggestions.
Access care tips based on lifestyle and weather.
Explore visualizations of interaction patterns.
Get shoe recommendations for specific outfits or events (e.g., Casual, Formal, Workout).


Notes

The outfit/event recommendation feature enhances user experience by tailoring recommendations to specific events, adding value to the system.
The code is modular with logging for debugging and clear documentation.
Ensure data files are generated before running the app to avoid errors.
