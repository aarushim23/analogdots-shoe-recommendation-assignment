import streamlit as st
import pandas as pd
from recommendation import load_data, prepare_lightfm_data, train_model, get_recommendations, personalized_services, personalized_care_tips
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Shoe Recommendation System", layout="wide")
st.title("Shoe Recommendation System")

# Load data
try:
    data_dir = 'data'
    users, shoes, interactions, care_history = load_data(data_dir)
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    logger.error("Data loading failed: %s", str(e))
    st.stop()

# Prepare model
model = None
dataset = None
interactions_matrix = None
item_features = None
try:
    logger.info("Starting model preparation")
    dataset, interactions_matrix, item_features, shoes = prepare_lightfm_data(interactions, users, shoes)
    logger.info("Model preparation completed")
    try:
        logger.info("Starting model training")
        model = train_model(interactions_matrix, item_features)
        logger.info("Model training completed")
    except Exception as e:
        st.warning(f"Model training failed: {str(e)}. Recommendations will be unavailable, but other tabs may work.")
        logger.error("Model training failed: %s", str(e))
except Exception as e:
    st.error(f"Error preparing data: {str(e)}")
    logger.error("Data preparation failed: %s", str(e))
    st.stop()

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Home", "Data Overview", "Recommendations", "Personalized Services", "Care Tips", "Visualizations", "Recommendations with Specific Outfits/Events"])

with tab1:
    st.subheader("Welcome to the Shoe Recommendation System")
    st.write("""
    This system leverages machine learning to enhance your shoe shopping and care experience. It analyzes user interactions, shoe attributes, and care history to provide personalized recommendations and services. Below is an overview of the available tabs:

    - **Data Overview**: Explore sample data from users, shoes, interactions, and care history to understand the dataset powering the system.
    - **Recommendations**: Get tailored shoe suggestions based on your preferences (available if model training succeeds).
    - **Personalized Services**: Receive notifications for shoe care and replacement suggestions based on usage and weather conditions.
    - **Care Tips**: Access customized shoe care advice tailored to your lifestyle and current weather.
    - **Visualizations**: View interactive charts to analyze interaction patterns and user engagement trends.
    - **Recommendations with Specific Outfits/Events**: Receive shoe recommendations tailored to specific outfits or events like weddings or workouts.

    The system uses a LightFM model to process data and provide insights, with fallback options to ensure functionality even if model training fails. Explore each tab to maximize your experience!
    """)

with tab2:
    st.subheader("Sample Data")
    try:
        st.write("Users")
        st.dataframe(users.head(), use_container_width=True)
        st.write("Shoes")
        st.dataframe(shoes.head(), use_container_width=True)
        st.write("Interactions")
        st.dataframe(interactions.head(), use_container_width=True)
        st.write("Care History")
        st.dataframe(care_history.head(), use_container_width=True)
    except Exception as e:
        st.error(f"Error displaying data: {str(e)}")
        logger.error("Data display failed: %s", str(e))

with tab3:
    st.subheader("Get Recommendations")
    if model is None:
        st.error("Recommendations are unavailable due to model training failure.")
    else:
        user_id = st.selectbox("Select User ID", users['user_id'].unique(), key="rec_user")
        if st.button("Generate Recommendations", key="gen_rec_button"):
            try:
                recommendations = get_recommendations(model, dataset, user_id, shoes)
                st.write("Top 5 Recommended Shoes")
                st.dataframe(recommendations, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating recommendations: {str(e)}")
                logger.error("Recommendation generation failed: %s", str(e))

with tab4:
    st.subheader("Personalized Services")
    weather_condition = st.selectbox("Select Weather Condition", ["Sunny", "Rainy", "Humid", "Cold"], key="service_weather")
    if st.button("Generate Services", key="gen_service_button"):
        try:
            notifications, replacements = personalized_services(care_history, shoes, interactions, weather_condition)
            st.write("### Care Notifications")
            st.write("These notifications are generated based on your shoe usage frequency, last care date, and current weather conditions. Regular maintenance can extend your shoes' lifespan and performance.")
            for note in notifications:
                st.write(f"- {note}")
            st.write("### Replacement Suggestions")
            st.write("Recommendations for replacement are based on the shoe's usage duration and wear frequency, calculated from purchase dates and interaction data. Replacing worn shoes ensures safety and comfort.")
            for rep in replacements:
                st.write(f"- {rep}")
            st.write("### Additional Insights")
            st.write("Weather impacts shoe care needs: Humid or rainy conditions increase the need for cleaning, while cold weather may require storage adjustments. Check your care history regularly for optimal results.")
        except Exception as e:
            st.error(f"Error generating services: {str(e)}")
            logger.error("Service generation failed: %s", str(e))

with tab5:
    st.subheader("Personalized Shoe Care Tips")
    user_id = st.selectbox("Select User ID", users['user_id'].unique(), key="care_user")
    weather_condition = st.selectbox("Select Weather Condition", ["Sunny", "Rainy", "Humid", "Cold"], key="care_weather")
    if st.button("Generate Care Tips", key="gen_care_button"):
        try:
            care_tips = personalized_care_tips(users, care_history, shoes, weather_condition)
            if care_tips:
                st.write("### Personalized Care Tips")
                st.write("These tips are tailored to your lifestyle (e.g., running or formal use) and current weather, helping maintain shoe quality and durability.")
                for tip in care_tips:
                    st.write(f"- {tip}")
            else:
                st.write("No care tips needed at this time.")
        except Exception as e:
            st.error(f"Error generating care tips: {str(e)}")
            logger.error("Care tips generation failed: %s", str(e))

with tab6:
    st.subheader("Interaction Statistics")
    st.write("""
    This section provides interactive visualizations to analyze user engagement and interaction patterns, critical for evaluating the recommendation system and tailoring services.
    """)
    try:
        interaction_counts = interactions['interaction_type'].value_counts()
        st.bar_chart(interaction_counts, use_container_width=True)
        st.write("### Interaction Type Distribution")
        st.write("Bar chart above shows the frequency of View, Purchase, and Wishlist interactions, indicating browsing behavior, conversion rates, and user intent.")

        # Additional Visualization: Shoe Type Popularity
        shoe_type_counts = shoes['type'].value_counts()
        st.bar_chart(shoe_type_counts, use_container_width=True)
        st.write("### Shoe Type Popularity")
        st.write("This bar chart displays the distribution of shoe types (sneaker, running shoe, dress shoe), highlighting user preferences.")

        # Additional Visualization: Interaction by User
        user_interaction_counts = interactions['user_id'].value_counts().head(10)
        st.bar_chart(user_interaction_counts, use_container_width=True)
        st.write("### Top 10 Active Users")
        st.write("This chart shows the top 10 users by interaction count, offering insights into engagement levels.")

        # Additional Visualization: Care Frequency by Shoe
        care_per_shoe = care_history['shoe_id'].value_counts().head(10)
        st.bar_chart(care_per_shoe, use_container_width=True)
        st.write("### Top 10 Most Cared-for Shoes")
        st.write("This visualization tracks the frequency of care for the top 10 shoes, indicating maintenance needs.")
    except Exception as e:
        st.error(f"Error generating visualization: {str(e)}")
        logger.error("Visualization failed: %s", str(e))

with tab7:
    st.subheader("Recommendations with Specific Outfits/Events")
    if model is None:
        st.error("Recommendations are unavailable due to model training failure.")
    else:
        outfit_event = st.selectbox("Select Outfit/Event", ["Casual", "Formal", "Sports", "Wedding", "Workout", "Party"], key="outfit_event")
        user_id = st.selectbox("Select User ID", users['user_id'].unique(), key="outfit_user")
        if st.button("Generate Recommendations", key="gen_outfit_button"):
            try:
                # Filter shoes based on outfit/event type
                outfit_mapping = {
                    "Casual": ["sneaker"],
                    "Formal": ["dress shoe"],
                    "Sports": ["running shoe"],
                    "Wedding": ["dress shoe"],
                    "Workout": ["running shoe", "sneaker"],
                    "Party": ["dress shoe", "sneaker"]
                }
                filtered_shoes = shoes[shoes['type'].isin(outfit_mapping.get(outfit_event, []))]
                if filtered_shoes.empty:
                    st.write(f"No shoes available for {outfit_event}.")
                else:
                    recommendations = get_recommendations(model, dataset, user_id, filtered_shoes)
                    st.write(f"Top 5 Shoe Recommendations for {outfit_event}")
                    st.dataframe(recommendations, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating outfit/event recommendations: {str(e)}")
                logger.error("Outfit/event recommendation generation failed: %s", str(e))