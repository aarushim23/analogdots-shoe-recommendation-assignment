import pandas as pd
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
import os
import logging
from scipy.sparse import coo_matrix

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(data_dir='data'):
    try:
        users = pd.read_csv(os.path.join(data_dir, 'users.csv'))
        shoes = pd.read_csv(os.path.join(data_dir, 'shoes.csv'))
        interactions = pd.read_csv(os.path.join(data_dir, 'interactions.csv'))
        care_history = pd.read_csv(os.path.join(data_dir, 'care_history.csv'))
        logger.info("Data loaded successfully: %d users, %d shoes, %d interactions, %d care records",
                    len(users), len(shoes), len(interactions), len(care_history))
        return users, shoes, interactions, care_history
    except Exception as e:
        logger.error("Error loading data: %s", str(e))
        raise

def prepare_lightfm_data(interactions, users, shoes):
    try:
        # Define all possible feature values
        possible_types = ['sneaker', 'running shoe', 'dress shoe']
        possible_colors = ['Black', 'White', 'Blue', 'Red']
        possible_materials = ['Leather', 'Mesh', 'Synthetic']
        all_features = possible_types + possible_colors + possible_materials
        logger.info("Defined feature values: %s", all_features)

        # Log unique values for debugging
        logger.info("Unique types: %s", shoes['type'].unique())
        logger.info("Unique colors: %s", shoes['color'].unique())
        logger.info("Unique materials: %s", shoes['material'].unique())

        # Clean and standardize shoe features
        shoes = shoes.copy()
        shoes['type'] = shoes['type'].str.lower().str.strip().replace({t.lower(): t for t in possible_types})
        shoes['color'] = shoes['color'].str.capitalize().str.strip().replace({c.lower(): c for c in possible_colors})
        shoes['material'] = shoes['material'].str.capitalize().str.strip().replace({m.lower(): m for m in possible_materials})

        # Handle null values
        initial_rows = len(shoes)
        shoes = shoes.dropna(subset=['type', 'color', 'material', 'shoe_id'])
        if len(shoes) < initial_rows:
            logger.warning("Dropped %d rows due to null values in type, color, material, or shoe_id", initial_rows - len(shoes))

        # Filter out invalid feature values
        shoes = shoes[shoes['type'].isin(possible_types) & shoes['color'].isin(possible_colors) & shoes['material'].isin(possible_materials)]
        if len(shoes) < initial_rows:
            logger.warning("Dropped %d rows due to invalid feature values", initial_rows - len(shoes))

        # Log cleaned values
        logger.info("Cleaned types: %s", shoes['type'].unique())
        logger.info("Cleaned colors: %s", shoes['color'].unique())
        logger.info("Cleaned materials: %s", shoes['material'].unique())

        # Validate shoe_id
        if shoes['shoe_id'].duplicated().any():
            logger.warning("Duplicate shoe_id values found in shoes DataFrame")
            shoes = shoes.drop_duplicates(subset=['shoe_id'])

        # Validate user_id and shoe_id in interactions
        invalid_user_ids = interactions[~interactions['user_id'].isin(users['user_id'])]['user_id'].unique()
        if invalid_user_ids.size > 0:
            logger.warning("Found %d invalid user_id values in interactions: %s", len(invalid_user_ids), invalid_user_ids)
        invalid_shoe_ids = interactions[~interactions['shoe_id'].isin(shoes['shoe_id'])]['shoe_id'].unique()
        if invalid_shoe_ids.size > 0:
            logger.warning("Found %d invalid shoe_id values in interactions: %s", len(invalid_shoe_ids), invalid_shoe_ids)

        dataset = Dataset()
        logger.info("Fitting dataset with %d users and %d items", len(users['user_id'].unique()), len(shoes['shoe_id'].unique()))
        dataset.fit(
            users=users['user_id'].unique(),
            items=shoes['shoe_id'].unique(),
            item_features=all_features
        )

        logger.info("Building interactions matrix")
        valid_interactions = interactions[interactions['shoe_id'].isin(shoes['shoe_id']) & interactions['user_id'].isin(users['user_id'])]
        interactions_matrix, _ = dataset.build_interactions(
            (row.user_id, row.shoe_id) for row in valid_interactions.itertuples()
        )
        logger.info("Interactions matrix shape: %s, non-zero entries: %d, sparsity: %.2f%%",
                    interactions_matrix.shape, interactions_matrix.nnz,
                    (interactions_matrix.nnz / (interactions_matrix.shape[0] * interactions_matrix.shape[1])) * 100)

        logger.info("Building item features")
        feature_data = []
        for row in shoes.itertuples():
            features = [row.type, row.color, row.material]
            if all(f in all_features for f in features):
                feature_data.append((row.shoe_id, features))
            else:
                logger.warning("Skipping shoe_id %s due to invalid features: %s", row.shoe_id, features)

        if not feature_data:
            logger.error("No valid feature data to build item features")
            raise ValueError("No valid feature data available after filtering")

        item_features = dataset.build_item_features(feature_data)
        logger.info("Item features built successfully, shape: %s", item_features.shape)
        return dataset, interactions_matrix, item_features, shoes
    except Exception as e:
        logger.error("Error in prepare_lightfm_data: %s", str(e))
        raise

def train_model(interactions_matrix, item_features):
    try:
        logger.info("Initializing LightFM model with minimal parameters (logistic loss)")
        model = LightFM(loss='logistic', learning_rate=0.01, no_components=3, random_state=42)
        logger.info("Starting model.fit with interactions shape: %s, item features shape: %s",
                    interactions_matrix.shape, item_features.shape)
        model.fit(interactions_matrix, item_features=item_features, epochs=3, num_threads=1, verbose=True)
        logger.info("Model.fit completed successfully")
        return model
    except Exception as e:
        logger.error("Error in train_model during model.fit: %s", str(e))
        raise

def get_recommendations(model, dataset, user_id, shoes, n=5):
    try:
        user_mapping, _, item_mapping, _ = dataset.mapping()
        if user_id not in user_mapping:
            logger.warning("User ID %s not found in mapping", user_id)
            return pd.DataFrame(columns=['shoe_id', 'brand', 'model', 'type', 'color'])
        user_idx = user_mapping[user_id]
        n_items = len(item_mapping)
        scores = model.predict(user_idx, np.arange(n_items))
        top_items = np.argsort(-scores)[:n]
        recommendations = shoes[shoes['shoe_id'].isin([list(item_mapping.keys())[i] for i in top_items])][['shoe_id', 'brand', 'model', 'type', 'color']]
        logger.info("Generated recommendations for user %s", user_id)
        return recommendations
    except Exception as e:
        logger.error("Error in get_recommendations: %s", str(e))
        raise

def personalized_services(care_history, shoes, interactions, weather_condition='Sunny'):
    try:
        notifications = []
        replacements = []
        weather_data = {
            'current_condition': weather_condition,
            'humidity': 70 if weather_condition == 'Humid' else 40,
            'temperature': 5 if weather_condition == 'Cold' else 20
        }
        care_threshold_days = 30
        lifespan_thresholds = {'running shoe': 6, 'sneaker': 12, 'dress shoe': 18}
        wear_threshold = 10

        for user_id in care_history['user_id'].unique():
            user_shoes = care_history[care_history['user_id'] == user_id]
            user_interactions = interactions[(interactions['user_id'] == user_id) & (interactions['interaction_type'] == 'purchase')]
            frequent_shoes = interactions[(interactions['user_id'] == user_id) & (interactions['interaction_type'] == 'view')]['shoe_id'].value_counts()

            for shoe_id in user_shoes['shoe_id'].unique():
                last_care = pd.to_datetime(user_shoes[user_shoes['shoe_id'] == shoe_id]['care_timestamp']).max()
                days_since_care = (pd.Timestamp.now() - last_care).days
                wear_frequency = frequent_shoes.get(shoe_id, 0)
                shoe = shoes[shoes['shoe_id'] == shoe_id]
                if shoe.empty:
                    logger.warning("Shoe ID %s not found in shoes DataFrame", shoe_id)
                    continue
                shoe = shoe.iloc[0]
                if days_since_care > care_threshold_days and wear_frequency > wear_threshold and weather_data['current_condition'] in ['Rainy', 'Humid']:
                    notifications.append(f"Time to clean your {shoe['model']} due to frequent use and {weather_data['current_condition']} conditions!")

                first_purchase = pd.to_datetime(user_interactions[user_interactions['shoe_id'] == shoe_id]['interaction_timestamp']).min()
                if not pd.isna(first_purchase):
                    usage_duration_months = (pd.Timestamp.now() - first_purchase).days / 30
                    if usage_duration_months > lifespan_thresholds.get(shoe['type'], 12) and wear_frequency > wear_threshold:
                        replacements.append(f"Consider replacing your {shoe['model']} due to extensive use.")

        logger.info("Generated %d notifications and %d replacements", len(notifications), len(replacements))
        return notifications, replacements
    except Exception as e:
        logger.error("Error in personalized_services: %s", str(e))
        raise

def personalized_care_tips(users, care_history, shoes, weather_condition='Sunny'):
    try:
        care_tips = []
        weather_data = {
            'current_condition': weather_condition,
            'humidity': 70 if weather_condition == 'Humid' else 40,
            'temperature': 5 if weather_condition == 'Cold' else 20
        }
        for user_id in users['user_id'].unique():
            user_lifestyle = users[users['user_id'] == user_id]['typical_usage'].iloc[0]
            user_shoes = care_history[care_history['user_id'] == user_id]['shoe_id'].unique()
            for shoe_id in user_shoes:
                last_care = pd.to_datetime(care_history[care_history['shoe_id'] == shoe_id]['care_timestamp']).max()
                days_since_care = (pd.Timestamp.now() - last_care).days
                shoe = shoes[shoes['shoe_id'] == shoe_id]
                if shoe.empty:
                    logger.warning("Shoe ID %s not found in shoes DataFrame", shoe_id)
                    continue
                shoe = shoe.iloc[0]
                if days_since_care > 15:
                    if user_lifestyle == 'running' and shoe['type'] == 'running shoe':
                        if weather_data['humidity'] > 70:
                            care_tips.append(f"For your {shoe['model']}, apply a waterproof spray to protect against high humidity.")
                        else:
                            care_tips.append(f"Use a breathable mesh cleaner for your {shoe['model']} to maintain ventilation.")
                    elif user_lifestyle == 'formal' and shoe['material'] == 'Leather':
                        care_tips.append(f"Polish your {shoe['model']} leather shoes weekly to maintain shine for formal occasions.")
                    elif weather_data['temperature'] < 5:
                        care_tips.append(f"Store your {shoe['model']} in a dry place to prevent cold-weather cracking.")
        logger.info("Generated %d care tips", len(care_tips))
        return care_tips
    except Exception as e:
        logger.error("Error in personalized_care_tips: %s", str(e))
        raise