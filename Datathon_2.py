import streamlit as st
from streamlit_option_menu import option_menu
from st_aggrid import AgGrid
import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import math as m
import xgboost as xg
from sklearn.model_selection import train_test_split
from sklearn import metrics



from PIL import Image
import streamlit as st
st.set_page_config(page_title='Immo', page_icon='ML', layout="centered", initial_sidebar_state="auto", menu_items=None)

import base64

df_final = pd.read_csv(r'C:\Users\anton\Downloads\Data pour le Streamlit\final_file.csv')
df_final.dropna(inplace=True)

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local(r'C:\Users\anton\Downloads\Data pour le Streamlit\background_2.jpg')

cola, colb, colc = st.columns([1,6,1])

with cola:
    st.write("")

with colb:
    st.image(r'C:\Users\anton\Downloads\Data pour le Streamlit\Logodatathon2.png')

with colc:
    st.write("")

#################### IMPORT DF ##################3
#df=pd.read_csv('final_file.csv')


#with open('style.css') as f:
    #st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
 # %% selection des filtres   
with st.container():
    st.markdown("<h1 style='text-align: center; font-family:RotisSemiSerif; color: black;'>Predicting the future price</h1>", unsafe_allow_html=True)
    st.markdown("""<hr style="height:1px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: left; font-family:RotisSemiSerif; color: black;'>Enter search criteria : </h2>", unsafe_allow_html=True)
    #col1,col2= st.columns(2)
    col1,col3= st.columns((2,2))
    with col1:
        st.markdown("<p style='text-align: center; text-decoration: underline;color: black;'>Local Type : </p>", unsafe_allow_html=True)
        select_local= st.selectbox(" ", options=set(df_final.type_local))
   
    with col3:
        st.markdown("<p style='text-align: center;text-decoration: underline; color: black;'>Department COde</p>", unsafe_allow_html=True)
        select_region = st.selectbox(" ", options=set(df_final.code_departement))
         
        
    st.markdown("""<hr style="height:2px;border:none;color:black;background-color:white;" /> """, unsafe_allow_html=True)



with st.container():
   
    col2,col3,col4= st.columns((1, 2, 1))
    
     
   
    with col2:
        st.markdown("<p style='text-align: center;text-decoration: underline; color: black;'>city code</p>", unsafe_allow_html=True)
        city= st.number_input(" ", key=2, value=75013)
    with col3:
        st.markdown("<p style='text-align: center; text-decoration: underline;color: black;'>Rooms quantity' : </p>", unsafe_allow_html=True)
        rooms= st.number_input(" ", key=3, value = 3)
   
    with col4:
        st.markdown("<p style='text-align: center;text-decoration: underline; color: black;'>Total Area</p>", unsafe_allow_html=True)
        total_area = st.number_input(" ", key=4, value = 65)
         
        
with st.container():

    #col1,col2= st.columns(2)
    col1,col2,col3= st.columns((1, 2, 1))
    with col1:
        st.markdown("<p style='text-align: center; text-decoration: underline;color: black;'> interest rate : </p>", unsafe_allow_html=True)
        interest_rate= st.number_input(" ", key=5, value =1.1 )
   
    with col2:
        st.markdown("<p style='text-align: center;text-decoration: underline; color: black;'>Rate_duration</p>", unsafe_allow_html=True)
        Rate_duration= st.number_input(" ", key=6, value = 300)
    with col3:
        st.markdown("<p style='text-align: center; text-decoration: underline;color: black;'>jobless rate : </p>", unsafe_allow_html=True)
        jobless_rate= st.number_input(" ", key=7, value = 8.9)
   
 
# ['valeur_fonciere','code_commune',
# 'nombre_pieces_principales','Total_surface','taux pret','duree pret','taux chômage'] 

df_final = df_final[(df_final.code_departement == select_region) | (df_final.type_local == select_local) ]
#st.table(df_final.head())#	type_loca
List_col = ['code_commune','nombre_pieces_principales',
            'Total_surface','taux pret','duree pret','taux chômage'] 
X, y = df_final[List_col], df_final['valeur_fonciere']
X['taux chômage'] = X['taux chômage'].apply(lambda x : float(x.replace(',','.')))

if st.button("price prediction"):

    xgb_r = xg.XGBRegressor(objective ='reg:linear', n_estimators = 1000, seed = 42,learning_rate=0.01,max_depth=7)# Fitting the model
    
    train_X, test_X, train_y, test_y = train_test_split(X,y,random_state=42,test_size =0.3)
    xgb_r.fit(train_X, train_y)# Predict the model
    pred_test = xgb_r.predict(test_X)
    pred_train = xgb_r.predict(train_X)
    # RMSE Computation
    st.write(metrics.r2_score(train_y,pred_train))
    st.write(metrics.r2_score(test_y,pred_test))
    st.write(metrics.mean_absolute_error(test_y,pred_test))
    st.write(m.sqrt(metrics.mean_squared_error(test_y,pred_test)))

#new_topred = [[]]