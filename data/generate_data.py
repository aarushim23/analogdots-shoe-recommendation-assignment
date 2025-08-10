import pandas as pd
import numpy as np
from faker import Faker
import os
from datetime import datetime, timedelta

fake = Faker()
Faker.seed(42)
np.random.seed(42)

def generate_users(n=50):
    users = []
    for i in range(n):
        users.append({
            'user_id': i + 1,
            'typical_usage': np.random.choice(['casual', 'running', 'formal']),
            'preferred_color': np.random.choice(['Black', 'White', 'Blue', 'Red'])
        })
    return pd.DataFrame(users)

def generate_shoes(n=100):
    shoes = []
    for i in range(n):
        shoes.append({
            'shoe_id': i + 1,
            'brand': fake.company(),
            'model': fake.word().capitalize() + ' ' + str(np.random.randint(100, 1000)),
            'type': np.random.choice(['sneaker', 'running shoe', 'dress shoe']),
            'color': np.random.choice(['Black', 'White', 'Blue', 'Red']),
            'material': np.random.choice(['Leather', 'Mesh', 'Synthetic'])
        })
    return pd.DataFrame(shoes)

def generate_interactions(n=500, n_users=50, n_shoes=100):
    interactions = []
    for _ in range(n):
        interactions.append({
            'user_id': np.random.randint(1, n_users + 1),
            'shoe_id': np.random.randint(1, n_shoes + 1),
            'interaction_type': np.random.choice(['view', 'purchase', 'wishlist']),
            'interaction_timestamp': (datetime.now() - timedelta(days=np.random.randint(1, 365))).strftime('%Y-%m-%d %H:%M:%S')
        })
    return pd.DataFrame(interactions)

def generate_care_history(n=250, n_users=50, n_shoes=100):
    care_history = []
    for _ in range(n):
        care_history.append({
            'user_id': np.random.randint(1, n_users + 1),
            'shoe_id': np.random.randint(1, n_shoes + 1),
            'care_type': np.random.choice(['clean', 'polish', 'repair']),
            'care_timestamp': (datetime.now() - timedelta(days=np.random.randint(1, 365))).strftime('%Y-%m-%d %H:%M:%S')
        })
    return pd.DataFrame(care_history)

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    users = generate_users()
    shoes = generate_shoes()
    interactions = generate_interactions()
    care_history = generate_care_history()
    users.to_csv('data/users.csv', index=False)
    shoes.to_csv('data/shoes.csv', index=False)
    interactions.to_csv('data/interactions.csv', index=False)
    care_history.to_csv('data/care_history.csv', index=False)
    print("Data generated successfully.")