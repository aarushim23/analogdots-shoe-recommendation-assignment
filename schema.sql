CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    age INTEGER,
    gender VARCHAR(10),
    typical_usage VARCHAR(20),
    preferred_color VARCHAR(20),
    CONSTRAINT unique_user_id UNIQUE (user_id)
);

CREATE TABLE shoe_catalog (
    shoe_id SERIAL PRIMARY KEY,
    brand VARCHAR(50),
    model VARCHAR(50),
    type VARCHAR(20),
    color VARCHAR(20),
    material VARCHAR(20),
    size VARCHAR(10),
    care_requirements TEXT,
    CONSTRAINT unique_shoe_id UNIQUE (shoe_id)
);

CREATE TABLE user_interactions (
    interaction_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id),
    shoe_id INTEGER REFERENCES shoe_catalog(shoe_id),
    interaction_type VARCHAR(20),
    interaction_timestamp TIMESTAMP,
    care_mode VARCHAR(20),
    care_frequency INTEGER,
    CONSTRAINT fk_user FOREIGN KEY (user_id) REFERENCES users(user_id),
    CONSTRAINT fk_shoe FOREIGN KEY (shoe_id) REFERENCES shoe_catalog(shoe_id)
);

CREATE TABLE recommendation_logs (
    log_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id),
    shoe_id INTEGER REFERENCES shoe_catalog(shoe_id),
    recommendation_timestamp TIMESTAMP,
    recommendation_score FLOAT,
    CONSTRAINT fk_user_log FOREIGN KEY (user_id) REFERENCES users(user_id),
    CONSTRAINT fk_shoe_log FOREIGN KEY (shoe_id) REFERENCES shoe_catalog(shoe_id)
);