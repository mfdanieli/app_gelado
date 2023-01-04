#!/usr/bin/env python
# coding: utf-8

# Goal: using the model to predict Fe and Mn concentrations
# Author: Danieli M. F.
# Date: 04/01/23

# -------------------------   
# Bibliotecas
# -------------------------   

import pandas            as pd
import seaborn           as sns
import matplotlib.pyplot as plt
import numpy             as np
import plotly.express as px
import plotly.io as pio
import inflection
import streamlit as st

from IPython.display     import display, HTML
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from scipy.stats import pearsonr

from scipy.optimize import fsolve
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import r2_score

# -------------------------   
# Auxiliary functions
# -------------------------   

# calculates regression errors and shows it on a table
# y: observed, y_hat: predicted

def ml_error( model_name, y, yhat ):
    mae = metrics.mean_absolute_error( y, yhat )
    mse = metrics.mean_squared_error( y, yhat )
    mape = metrics.mean_absolute_percentage_error( y, yhat )
    r2 = r2_score( y, yhat )     
    rmse = np.sqrt( metrics.mean_squared_error( y, yhat ) )
    mpe = np.mean(((y - yhat) / y)) * 100
    
    return pd.DataFrame( { 'Model Name': model_name, 
                           'MAE': mae, 
                           'MSE': mse,
                           'MAPE': mape,
                           'R2': r2,
                           'RMSE': rmse,
                           'MPE': mpe,  
                         }, index=[0] )

# -------------------------   
# Leitura dados > preciso deixar mais inteligente!!!
# -------------------------

# LEITURA 
df = pd.read_excel("dados_editado.xlsx",
                   sheet_name = "NWC18",
                   skiprows = range(1, 55))
                   
df1 = df.T                               # transpoe 
df2 = df1.drop(df1.columns[[1]], axis=1) # tira col com nan
df3=df2.dropna(axis=0)                   # tira linhas com nanimport streamlit as st

tab_1 = df3.rename(columns=df3.iloc[0]).drop(df3.index[0])  # pega linha 1 e coloca como nome das colunas

df = pd.read_excel("dados_editado.xlsx",
                   sheet_name = "NWC21",
                   skiprows = range(1, 55))                
df1 = df.T                               
df2 = df1.drop(df1.columns[[1]], axis=1) 
df3=df2.dropna(axis=0)                 
tab_2 = df3.rename(columns=df3.iloc[0]).drop(df3.index[0])  

df = pd.read_excel("dados_editado.xlsx",
                   sheet_name = "RD04",
                   skiprows = range(1, 55))                
df1 = df.T                               
df2 = df1.drop(df1.columns[[1]], axis=1) 
df3=df2.dropna(axis=0)                 
tab_3 = df3.rename(columns=df3.iloc[0]).drop(df3.index[0])  

df = pd.read_excel("dados_editado.xlsx",
                   sheet_name = "RD05",
                   skiprows = range(1, 55))                
df1 = df.T                               
df2 = df1.drop(df1.columns[[1]], axis=1) 
df3=df2.dropna(axis=0)                 
tab_4 = df3.rename(columns=df3.iloc[0]).drop(df3.index[0])  

df = pd.read_excel("dados_editado.xlsx",
                   sheet_name = "RD06",
                   skiprows = range(1, 55))                
df1 = df.T                               
df2 = df1.drop(df1.columns[[1]], axis=1) 
df3=df2.dropna(axis=0)                 
tab_5 = df3.rename(columns=df3.iloc[0]).drop(df3.index[0])  

df = pd.read_excel("dados_editado.xlsx",
                   sheet_name = "RD07",
                   skiprows = range(1, 55))                
df1 = df.T                               
df2 = df1.drop(df1.columns[[1]], axis=1) 
df3=df2.dropna(axis=0)                 
tab_6 = df3.rename(columns=df3.iloc[0]).drop(df3.index[0])  

df = pd.read_excel("dados_editado.xlsx",
                   sheet_name = "NE07",
                   skiprows = range(1, 55))                
df1 = df.T                               
df2 = df1.drop(df1.columns[[1]], axis=1) 
df3=df2.dropna(axis=0)                 
tab_7 = df3.rename(columns=df3.iloc[0]).drop(df3.index[0])  

df = pd.read_excel("dados_editado.xlsx",
                   sheet_name = "NE09",
                   skiprows = range(1, 55))                
df1 = df.T                               
df2 = df1.drop(df1.columns[[1]], axis=1) 
df3=df2.dropna(axis=0)                 
tab_8 = df3.rename(columns=df3.iloc[0]).drop(df3.index[0])  

df = pd.read_excel("dados_editado.xlsx",
                   sheet_name = "RD08",
                   skiprows = range(1, 55))                
df1 = df.T                               
df2 = df1.drop(df1.columns[[1]], axis=1) 
df3=df2.dropna(axis=0)                 
tab_9 = df3.rename(columns=df3.iloc[0]).drop(df3.index[0])  

df = pd.read_excel("dados_editado.xlsx",
                   sheet_name = "NWC23",
                   skiprows = range(1, 55))                
df1 = df.T                               
df2 = df1.drop(df1.columns[[1]], axis=1) 
df3=df2.dropna(axis=0)                 
tab_10 = df3.rename(columns=df3.iloc[0]).drop(df3.index[0])  

df = pd.read_excel("dados_editado.xlsx",
                   sheet_name = "NWC22",
                   skiprows = range(1, 55))                
df1 = df.T                               
df2 = df1.drop(df1.columns[[1]], axis=1) 
df3=df2.dropna(axis=0)                 
tab_11 = df3.rename(columns=df3.iloc[0]).drop(df3.index[0])  

df = pd.read_excel("dados_editado.xlsx",
                   sheet_name = "RD09",
                   skiprows = range(1, 55))                
df1 = df.T                               
df2 = df1.drop(df1.columns[[1]], axis=1) 
df3=df2.dropna(axis=0)                 
tab_12 = df3.rename(columns=df3.iloc[0]).drop(df3.index[0])  

df = pd.read_excel("dados_editado.xlsx",
                   sheet_name = "RD10",
                   skiprows = range(1, 55))                
df1 = df.T                               
df2 = df1.drop(df1.columns[[1]], axis=1) 
df3=df2.dropna(axis=0)                 
tab_13 = df3.rename(columns=df3.iloc[0]).drop(df3.index[0])  

df = pd.read_excel("dados_editado.xlsx",
                   sheet_name = "T-344-J",
                   skiprows = range(1, 55))                
df1 = df.T                               
df2 = df1.drop(df1.columns[[1]], axis=1) 
df3=df2.dropna(axis=0)                 
tab_14 = df3.rename(columns=df3.iloc[0]).drop(df3.index[0]) 

df = pd.read_excel("dados_editado.xlsx",
                   sheet_name = "RD11",
                   skiprows = range(1, 55))                
df1 = df.T                               
df2 = df1.drop(df1.columns[[1]], axis=1) 
df3=df2.dropna(axis=0)                 
tab_15 = df3.rename(columns=df3.iloc[0]).drop(df3.index[0])  

# NWC18 (tab_1) e NE07 (tab_7) sao efluentes! -> tirar das analises
tabelas = [tab_2, tab_3, tab_4, tab_5, tab_6, tab_8, tab_9, tab_10, tab_11, tab_12, tab_13, tab_14, tab_15]

dataframe = pd.concat(tabelas,join='inner', ignore_index=True)

# correcting mistake in maganese symbol
dataframe.rename(columns={"Kd_Mg": "Kd_Mn"},inplace=True) # simbolo manganes (Mn) tava como magnesio (Mg)
dataframe.rename(columns={"Dissolved Mg": "Dissolved Mn"},inplace=True) 
dataframe.rename(columns={"Total Mg": "Total Mn"},inplace=True) 
# display(HTML(dataframe.to_html()))



# -------------------------   
# Data cleaning
# -------------------------

df1 = dataframe.copy()

# Dropping unwanted columns 

df1 = df1.drop(['Kd_Cu','Kd_Fe','Kd_Mn','Kd_mean','Dissolved Cu','ponto','Dissolved Fe','Dissolved Mn'], axis=1) 

# Fomatting column names

# spaces to underscore
df1.columns = df1.columns.str.replace(' ', '_')

# to lower case
df1= df1.rename(columns=str.lower)


# Check tipos de dados

df1.info()


# Dealing with Nan

# found some cells with empty spaces, so I replaced them with Nan
df1 = df1.replace(' ', np.nan) 

# checking where the nan are
df1.isnull().sum()

# selecting only the columns with numeric values

df1_numeric = df1.select_dtypes( include=['int64', 'float64'] )

# replacing with the median value

for cols in df1_numeric.columns:
    df1[cols] = df1[cols].fillna(df1[cols].median())
    
# verifying if it worked
df1.isnull().sum()

# -------------------------   
# MODEL PREPARATION
# -------------------------

df2 = df1.copy()

# Separação dataframe seca / chuvoso 

df2_seca = df2[df2['season']=='dry']
df2_chuvoso = df2[df2['season']=='wet']

# Separação dataframe seca / chuvoso - Fe / Mn -> 4 dataframes

df2_seca_Fe = df2_seca.drop(['total_mn'],axis=1)

df2_chuvoso_Fe = df2_chuvoso.drop(['total_mn'],axis=1)

df2_seca_Mn = df2_seca.drop(['total_fe'],axis=1)

df2_chuvoso_Mn = df2_chuvoso.drop(['total_fe'],axis=1)

# -------------------------   
# Random forest regressor 
# -------------------------

st.header('Modeling Fe and Mn during the dry and rainy')

# criar abas
tab1,tab2 = st.tabs(['How to use','Model application'])

with tab1: 
    with st.container(): 
        st.subheader('Bla')

        st.markdown('## ......')
        
        
        
        
        st.subheader('Model performance metrics: original observed data')

        # Fe: dry
        # -------------------------

        # dataframe of interest
        X = df2_seca_Fe.drop(['season','total_fe'],axis=1)
        y = df2_seca_Fe['total_fe']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

        # model

        rf_seca_Fe = RandomForestRegressor( n_estimators=100, n_jobs=-1, random_state=101).fit( X_train, y_train )

        # prediction
        yhat = rf_seca_Fe.predict(X_test)

        performance = ml_error('Random forest', y_test, yhat)

        st.write('Model performance for Fe during the dry season:')
        st.write(performance)

        # Fe: wet
        # -------------------------

        # dataframe of interest
        X = df2_chuvoso_Fe.drop(['season','total_fe'],axis=1)
        y = df2_chuvoso_Fe['total_fe']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

        # model
        rf_chuvoso_Fe = RandomForestRegressor( n_estimators=100, n_jobs=-1, random_state=101).fit( X_train, y_train )

        # prediction
        yhat = rf_chuvoso_Fe.predict(X_test)

        performance = ml_error('Random forest', y_test, yhat)
        st.write('Model performance for Fe during the rainy season:')
        st.write(performance)

        # Mn: dry
        # -------------------------

        # dataframe of interest 
        X = df2_seca_Mn.drop(['season','total_mn'],axis=1)
        y = df2_seca_Mn['total_mn']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

        # model
        rf_seca_Mn = RandomForestRegressor( n_estimators=100, n_jobs=-1, random_state=101).fit( X_train, y_train )

        # prediction
        yhat = rf_seca_Mn.predict(X_test)

        performance = ml_error('Random forest', y_test, yhat)
        st.write('Model performance for Mn during the dry season:')
        st.write(performance) 

        # Mn: wet
        # -------------------------

        # dataframe of interest
        X = df2_chuvoso_Mn.drop(['season','total_mn'],axis=1)
        y = df2_chuvoso_Mn['total_mn']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

        # model
        rf_chuvoso_Mn = RandomForestRegressor( n_estimators=100, n_jobs=-1, random_state=101).fit( X_train, y_train )

        # prediction
        yhat = rf_chuvoso_Mn.predict(X_test)

        performance = ml_error('Random forest', y_test, yhat)
        st.write('Model performance for Mn during the rainy season:')
        st.write(performance)


# ******************
# Barra lateral Streamlit
# ******************  

# image = Image.open('food-delivery-2.png')

# st.sidebar.image(image,width=120)

st.sidebar.markdown('# Fe and Mn in the Igarapé-Gelado')
st.sidebar.markdown('## Water quality in rivers')
st.sidebar.markdown("""---""")

st.sidebar.markdown('## Select water quality characteristics')

def get_user_data():
    ph = st.sidebar.slider('pH', 0,12,6)#df2['ph'].min(), df2['ph'].max(), df2['ph'].median())
    temperature = st.sidebar.slider('Temperature', df2['temperature'].min(), df2['temperature'].max(), df2['temperature'].median())
    do = st.sidebar.slider('Dissolved oxygen (mg/L)', df2['do'].min(), df2['do'].max(), df2['do'].median())
    turbidity = st.sidebar.slider('Turbidity', df2['turbidity'].min(), df2['turbidity'].max(), df2['turbidity'].median())
    conductivity = st.sidebar.slider('Conductivity', df2['conductivity'].min(), df2['conductivity'].max(), df2['conductivity'].median())
    suspended_solids = st.sidebar.slider('Suspended solids', df2['suspended_solids'].min(), df2['suspended_solids'].max(), df2['suspended_solids'].median())
    dissolved_solids = st.sidebar.slider('Dissolved solids', df2['dissolved_solids'].min(), df2['dissolved_solids'].max(), df2['dissolved_solids'].median())

    # um dicionário recebe as informações acima
    user_data = {'ph': ph,
                 'temperature': temperature,
                 'do': do,
                 'turbidity': turbidity,
                 'conductivity': conductivity,
                 'suspended_solids': suspended_solids,
                 'dissolved_solids': dissolved_solids
                 }
    features = pd.DataFrame(user_data,index=[0])
  
    return features

# form the new input dataset (X_test)
user_input_variables = get_user_data()
# New prediction
prediction_seca_Fe = rf_seca_Fe.predict(user_input_variables)
prediction_seca_Mn = rf_seca_Mn.predict(user_input_variables)
prediction_chuvoso_Fe = rf_chuvoso_Fe.predict(user_input_variables)
prediction_chuvoso_Mn = rf_chuvoso_Mn.predict(user_input_variables)
        
with tab2: 
    with st.container(): 

        st.header('Model application')

        st.write('Water characteristics defined by the user:')

        st.dataframe(user_input_variables)

        st.write('Predicted concentrations (mg/L):')
        
        col1,col2,col3,col4 = st.columns(4)
        with col1:
            col1.metric('Fe_rainy', np.round(prediction_chuvoso_Fe,2),delta=str(np.round(prediction_chuvoso_Fe-0.3,2)).replace('[','').replace(']',''),delta_color="inverse")
        with col2:
            col2.metric('Fe_dry', np.round(prediction_seca_Fe,2),delta=str(np.round(prediction_seca_Fe-0.3,2)).replace('[','').replace(']',''),delta_color="inverse")
        with col3:
            col3.metric('Mn_rainy', np.round(prediction_chuvoso_Mn,2),delta=str(np.round(prediction_chuvoso_Mn-0.1,2)).replace('[','').replace(']',''),delta_color="inverse")
        with col4:
            col4.metric('Mn_dry', np.round(prediction_seca_Mn,2),delta=str(np.round(prediction_seca_Mn-0.1,2)).replace('[','').replace(']',''),delta_color="inverse")
        
        st.markdown("""
> *The number accompanied by the arrow indicates the difference between the predicted and the limit concentration. 
> The red color indicates the mg/L by which the predicted concentration is larger than the limit; the green is the contrary.*
        """)

            
st.sidebar.markdown("""---""")

st.sidebar.markdown('### Powered by @mfdanieli')
