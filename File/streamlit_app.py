import numpy as np
import pandas as pd
import seaborn as sns 
import streamlit as st
import sklearn
import joblib
import os

from streamlit.elements.form import FormData
from PIL import Image
from sklearn.model_selection import train_test_split


st.set_page_config(
    page_title="Predicting Discount Price For Your Product On Wish",
    page_icon="🧙‍♂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

"""
# Predicting Discount Price For Your Product On Wish
[![Star](https://img.shields.io/github/stars/arturlunardi/predict_rental_prices_streamlit?style=social)](https://github.com/SuzyNguyenn/Wish_ecommerce)
&nbsp[![Follow](https://img.shields.io/badge/Connect-follow?style=social&logo=linkedin)](https://www.linkedin.com/in/hayennguyen/)
"""
st.title('Discount Price Prediction')
st.sidebar.header('Product Data')
image = Image.open('Sell-On-Wish-Marketplace-1.jpeg')
st.image(image, '')

# ----------- Global Sidebar ---------------

condition = st.sidebar.selectbox(
    "Select the visualization",
    ("Introduction", "Model Prediction")
)

# FUNCTION
def user_report():
  prod_gen = st.sidebar.selectbox('Genders',('Man','Woman'))
  retail_price = st.sidebar.number_input('Retail Price')
  nb_cart_orders_approx = st.sidebar.number_input('Unit sold')
  rating = st.sidebar.number_input('Rating(0-5)')
  rating_count = st.sidebar.number_input('Rating count')
  product_color = st.sidebar.selectbox('Product Color',('black', 'white', 'blue', 'red', 'green', 'yellow', 'pink', 'grey', 'purple', 'orange', 'brown', 'beige'))
  product_variation_size_id = st.sidebar.selectbox('Product Size',('extra_small_size', 'xs', 's', 'm', 'l', 'xl','extra_big_size'))
  product_variation_inventory = st.sidebar.number_input('Inventory total')
  shipping_option_price = st.sidebar.number_input('Shipping Price')
  merchant_positive_percent = st.sidebar.number_input('Positive Merchant Percent (0-100)')
  merchant_rating = st.sidebar.number_input('Merchant Rating (0-5)')
  merchant_rating_count = st.sidebar.number_input('merchant rating count')

  user_report_data = {
      'prod_gen': prod_gen,
      'retail_price':retail_price,
      'nb_cart_orders_approx':nb_cart_orders_approx,
      'rating':rating,
      'rating_count':rating_count,
      'product_color':product_color,
      'product_variation_size_id':product_variation_size_id,
      'product_variation_inventory':product_variation_inventory,
      'shipping_option_price':shipping_option_price,
      'shipping_is_express':True,
      'inventory_total':50,
      'origin_country':'CN',
      'merchant_positive_percent':merchant_positive_percent,
      'merchant_rating_count': merchant_rating_count,
      'merchant_rating':merchant_rating,
      'isnan' : 0
  }
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data


my_model_loaded = joblib.load("my_model_lightGBM.pkl")


# ------------- Introduction ------------------------

if condition == 'Introduction':
    # st.image(os.path.join(os.path.abspath(''), 'data', 'dataset-cover.jpg'))
    st.subheader('About')
    
    ## FALTA O CHECK ON GITHUB
    st.write("""
    This application provides an overview of the Sales of summer clothes in E-commerce Wish dataset from Kaggle.
    The data were provided from this [source](https://www.kaggle.com/jmmvutu/summer-products-and-sales-in-ecommerce-wish/metadata). 
    You can check on the sidebar:
    - EDA (Exploratory Data Analysis)
    - Model Prediction
    The prediction are made regarding to the Discount Price amount utilizing pre trained machine learning model Light GBM.
    All the operations in the dataset were already done and stored as csv files inside the data directory. If you want to check the code, go through the notebook directory in the [github repository](https://github.com/SuzyNguyenn/Wish_ecommerce).
    """)

    st.subheader('Model Definition')
    st.write('After train on many base model')
    st.image('new_base_model.png')
    st.write("""
    The structure of the training it is to wrap the process around a scikit-learn Pipeline. 
    
    Model:
    - LightGBM - MAE = 1.53 & RMSE = 3.60
    - Our main accuracy metric is RMSE. To enhance our model definition, we utilized Random Search for hyperparameter tuning.
    """)
    st.image('model_selection_proof.png')
    st.write('Features Important')
    st.image('feature_important.png')

    

# ------------- Model prediction ------------------------

if condition == 'Model Prediction':
    def user_report():
        prod_gen = st.sidebar.selectbox('Genders',('Man','Woman'))
        retail_price = st.sidebar.number_input('Retail Price')
        nb_cart_orders_approx = st.sidebar.number_input('Unit sold')
        rating = st.sidebar.number_input('Rating(0-5)')
        rating_count = st.sidebar.number_input('Rating count')
        product_color = st.sidebar.selectbox('Product Color',('black', 'white', 'blue', 'red', 'green', 'yellow', 'pink', 'grey', 'purple', 'orange', 'brown', 'beige'))
        product_variation_size_id = st.sidebar.selectbox('Product Size',('extra_small_size', 'xs', 's', 'm', 'l', 'xl','extra_big_size'))
        product_variation_inventory = st.sidebar.number_input('total Variance Inventory')
        shipping_option_price = st.sidebar.number_input('Shipping Price')
        inventory_total	= st.sidebar.number_input('Total Inventory')
        merchant_positive_percent = st.sidebar.number_input('Positive Merchant Percent (0-100)')
        merchant_rating = st.sidebar.number_input('Merchant Rating (0-5)')
        merchant_rating_count = st.sidebar.number_input('merchant rating count')

        user_report_data = {
            'prod_gen': prod_gen,
            'retail_price':retail_price,
            'nb_cart_orders_approx':nb_cart_orders_approx,
            'rating':rating,
            'rating_count':rating_count,
            'product_color':product_color,
            'product_variation_size_id':product_variation_size_id,
            'product_variation_inventory':product_variation_inventory,
            'shipping_option_price':shipping_option_price,
            'shipping_is_express':True,
            'inventory_total':inventory_total,
            'origin_country':'CN',
            'merchant_positive_percent':merchant_positive_percent,
            'merchant_rating_count': merchant_rating_count,
            'merchant_rating':merchant_rating,
            'isnan' : 0
        }
        report_data = pd.DataFrame(user_report_data, index=[0])
        return report_data
        
    user_data = user_report()
    st.header('Product Data')
    st.write(user_data)
    prediction = my_model_loaded.predict(user_data)
    st.write('Your Product Discount Price approximately:',round(prediction[0],2))
    st.write('mean absolute error (MAE) = 1.53')
    

