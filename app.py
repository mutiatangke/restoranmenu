# app.py (Streamlit Application)
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load artifacts
model = joblib.load('best_model.pkl')

st.title('🍽️ Restaurant Menu Profitability Predictor')
st.write("Predict profitability of menu items using machine learning")

# Input form
st.sidebar.header("Input Menu Features")
price = st.sidebar.number_input("Price", min_value=0.0, max_value=100.0, value=15.0)
num_ingredients = st.sidebar.number_input("Number of Ingredients", min_value=1, max_value=20, value=4)
has_premium = st.sidebar.selectbox("Contains Premium Ingredients", ["No", "Yes"])
is_beverage = st.sidebar.selectbox("Is Beverage", ["No", "Yes"])
menu_category = st.sidebar.selectbox("Menu Category", ["Appetizers", "Beverages", "Desserts", "Main Course"])

# Transform inputs
has_premium = 1 if has_premium == "Yes" else 0
is_beverage = 1 if is_beverage == "Yes" else 0

# Create input data
input_data = pd.DataFrame({
    'Price': [price],
    'num_ingredients': [num_ingredients],
    'has_premium': [has_premium],
    'is_beverage': [is_beverage],
    'MenuCategory': [menu_category]
})

# Preprocessing
category_encoded = encoder.transform(input_data[['MenuCategory']])
encoded_cols = encoder.get_feature_names_out(['MenuCategory'])
category_df = pd.DataFrame(category_encoded, columns=encoded_cols)

num_features = input_data[['Price', 'num_ingredients']]
num_scaled = scaler.transform(num_features)
num_df = pd.DataFrame(num_scaled, columns=['Price', 'num_ingredients'])

features = pd.concat([
    num_df,
    input_data[['has_premium', 'is_beverage']].reset_index(drop=True),
    category_df
], axis=1)

# Prediction
if st.sidebar.button('Predict Profitability'):
    prediction = model.predict(features)
    probabilities = model.predict_proba(features)[0]
    
    profit_class = le.inverse_transform(prediction)[0]
    class_prob = probabilities.max()
    
    st.success(f"Predicted Profitability: **{profit_class}**")
    st.subheader("Probability Distribution")
    
    prob_df = pd.DataFrame({
        "Class": le.classes_,
        "Probability": probabilities
    })
    
    st.bar_chart(prob_df.set_index("Class"))
    
    st.write(f"Class Probabilities:")
    for i, cls in enumerate(le.classes_):
        st.write(f"- {cls}: {probabilities[i]:.4f}")
    
    # Show insights
    st.subheader("Business Insights")
    if profit_class == "High":
        st.success("✅ This menu item has high profitability potential")
    elif profit_class == "Medium":
        st.warning("⚠️ This menu item has medium profitability. Consider optimizing costs")
    else:
        st.error("❌ This menu item has low profitability. Consider re-engineering or removing")
    
    if has_premium and price < 20:
        st.info("💡 Tip: Premium ingredients with price < $20 may increase profit margin")
    elif is_beverage and price > 5:
        st.info("💡 Tip: Beverages priced > $5 typically have high profit margins")

st.sidebar.markdown("---")
st.sidebar.info("""
This model predicts menu profitability based on:
- Price
- Number of ingredients
- Premium ingredients
- Beverage status
- Menu category

""")
