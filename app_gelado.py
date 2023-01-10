#!/usr/bin/env python
# coding: utf-8

# Goal: using the model to predict Fe and Mn concentrations
# Author: Danieli M. F.
# Date: 04/01/23
# Update: 10/01/23: perc land use as model feature

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
from pxmap import px_static

st.set_page_config(page_title='Water quality',page_icon='💦', layout='wide')

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

# calculates the risks
def health_risk(conc,rfd):
    IR = 2.2 # L/d
    ED = 70  # anos
    EF = 365 # dias
    BW = 70  # kg
    HQ = ((conc*IR*EF*ED) / (BW*365*ED))/rfd
    return HQ

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
# NEW: Adicionando uso do solo no dataframe
# LEITURA dos dataframes de pto x minibacia e de minibacia x porcentagens da área com cada tipo de uso
df_a = pd.read_excel("perc_solo_miniBac_pto_qualid.xlsx",
                   sheet_name = "gelado_ptos")
# df_a[['Ponto','FID_catchm']]
df_b = pd.read_excel("perc_solo_miniBac_pto_qualid.xlsx",
                   sheet_name = "gelado_uso")
# juntando os dataframes pelo codigo de minibacia
df_c = pd.merge(df_a[['Ponto','FID_catchm']],df_b[['FID_catchm','PercArea','UsoTipo']],how='inner',on='FID_catchm')
# pivot table p/ mostrar % por uso, pto e minibacia
df_d = pd.pivot_table(df_c,index=["Ponto","FID_catchm","UsoTipo"]).unstack().reset_index()
# arrumando nomes colunas
df_d.columns = ['ponto','fid_catchm','forest','mining','non_forest','pasture','urban_area','water']
# replacing Nan with 0
df_d = df_d.fillna(0)
# juntando com o dataframe de conc x caracteristicas agua
dataframe_new = pd.merge(df_d,dataframe,how='inner',on='ponto')


# -------------------------   
# Data cleaning
# -------------------------

df1 = dataframe_new.copy()

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

# st.header('Modeling Fe and Mn during the dry and rainy')

# criar abas
tab1,tab2,tab3 = st.tabs(['About','Model application','Data'])

with tab1: 
    with st.container(): 
        # st.subheader('About')

        st.subheader('Real-time estimations of Fe and Mn concentrations in the water column of rivers')
        
        st.markdown('##### How to use this app')
         
        st.write('In the sidebar, you can alter water characteristics.')
        st.write('In the next tab you will verify the resulting metal concentrations, during rainy and dry seasons.')
                
        st.markdown('##### Model information')
                    
        st.write('A regression technique is used to estimate total metal concentrations based on water characteristics. The random forest model was developed using monthly observed data during 2016 to 2018 in the Igarapé-Gelado basin, in the Pará State, Brazil. More details are found in ...')  
         
                    
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

        performance_seca_Fe = ml_error('Random forest', y_test, yhat)


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

        performance_chuvoso_Fe = ml_error('Random forest', y_test, yhat)

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

        performance_seca_Mn = ml_error('Random forest', y_test, yhat)

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

        performance_chuvoso_Mn = ml_error('Random forest', y_test, yhat)
        
        # merging the performance metrics into a dataframe
        performances = [performance_seca_Fe,performance_chuvoso_Fe,performance_seca_Mn,performance_chuvoso_Mn]

        performances_merg = pd.concat(performances,join='inner', ignore_index=True)
        performances_merg.index =['Fe: dry','Fe: rainy','Mn: dry','Mn: rainy']
        st.write('Model performance:')
        st.dataframe(performances_merg)

        st.markdown('##### Risks')
        
        st.markdown('###### Health quotient')

        st. write("""
        > HQ = CDI/RfD
        >
        > RfD: 0.7 mg/kg/day for iron and 0.14 mg/kg/day for manganese
        >
        > CDI = (C x IR x EF x ED) / (BW x AT)
        >
        > C is the iron or manganese concentration in water (mg/L); 
        > IR is the human water ingestion rate in L/day (2.2 L/day for adults);
        > ED is the exposure duration in years (70 years for adults); 
        > EF is the exposure frequency in days/year (365 days for adults); 
        > BW is the average body weight in kg (70 kg for adults); 
        > AT is the averaging time (AT = 365 × ED).
        >
        > The Health Index (HI) is the sum of HQ for each metal. A HI > suggests a possible risk.
        """)

# ******************
# Barra lateral Streamlit
# ******************  

st.sidebar.markdown('# Concentrations of Iron and Manganese in rivers')
# st.sidebar.markdown('## Water quality in rivers')
st.sidebar.image("hidro.gif")

st.sidebar.markdown("""---""")

st.sidebar.markdown('### Select water quality characteristics')

def get_user_data():
    # os valores sao min, max e med do dataset observado
    ph = st.sidebar.slider('pH', 5.15, 8.92, 7.04)
    temperature = st.sidebar.slider('Temperature', 7.63, 31.0, 26.7)
    do = st.sidebar.slider('Dissolved oxygen (mg/L)', 2.33, 9.58, 6.635)
    turbidity = st.sidebar.slider('Turbidity', 1.49, 89.8, 11.6)
    conductivity = st.sidebar.slider('Conductivity', 5.7, 395.0, 53.6)
    suspended_solids = st.sidebar.slider('Suspended solids', 11.0, 96.0, 11.0)
    dissolved_solids = st.sidebar.slider('Dissolved solids', 11.0, 232.0, 38.5)

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

        st.subheader('Model application')

        st.write('Input water characteristics:')

        # st.dataframe(user_input_variables)
        fig = px.bar(user_input_variables,barmode='group', labels={
                     "value": "Value",
                     "index": " ",
                 })
        st.plotly_chart(fig,use_container_width=True)

        st.subheader('Predicted concentrations (mg/L):')

        col1,col2,col3,col4 = st.columns(4)
        with col1:
            col1.metric('Fe: rainy', np.round(prediction_chuvoso_Fe,2),delta=str(np.round(prediction_chuvoso_Fe-0.3,2)).replace('[','').replace(']',''),delta_color="inverse")

        with col2:
            col2.metric('Fe: dry', np.round(prediction_seca_Fe,2),delta=str(np.round(prediction_seca_Fe-0.3,2)).replace('[','').replace(']',''),delta_color="inverse")

        with col3:
            col3.metric('Mn: rainy', np.round(prediction_chuvoso_Mn,2),delta=str(np.round(prediction_chuvoso_Mn-0.1,2)).replace('[','').replace(']',''),delta_color="inverse")
        with col4:
            col4.metric('Mn: dry', np.round(prediction_seca_Mn,2),delta=str(np.round(prediction_seca_Mn-0.1,2)).replace('[','').replace(']',''),delta_color="inverse")                            

        st.markdown("""
> *The number accompanied by the arrow indicates the difference between the predicted and the limit concentration. 
> The red color indicates the mg/L by which the predicted concentration is larger than the limit; the green is the contrary.*
        """)

    with st.container():
        health_index_seca = health_risk(prediction_seca_Fe,0.7) + health_risk(prediction_seca_Mn,0.14) 
        health_index_chuvoso = health_risk(prediction_chuvoso_Fe,0.7) + health_risk(prediction_chuvoso_Mn,0.14)
      
        col1,col2,col3 = st.columns(3)
        with col1:
            st.image("drink-water.png",width=50)
        with col2:
            col2.metric('HI: dry',np.round(health_index_seca,2))
        with col3:
            col3.metric('HI: rainy',np.round(health_index_chuvoso,2)) 
        
        # health = pd.DataFrame([health_index_seca,health_index_chuvoso])
        # health.index =['Dry','Rainy']
        # fig = px.bar(health,labels={
                 #     "value": "Health Index (HI)",
                 #     "index": " ",            
                 # })
        # fig.update_layout(showlegend=False)
        # st.plotly_chart(fig)
        st.markdown("""
> A HI > 1 suggests a possible risk""")
        
        # fig = px.choropleth(locationmode="USA-states", color=[1], scope="usa")
        # st.plotly_chart(fig,use_container_width=True)

with tab3:
    with st.container():
        st.write('map')
        # fig = px.scatter_mapbox(d3, lat=d3['Lat'], lon=d3['Long'],opacity=0.4)
        # fig.update_layout(mapbox_style='carto-positron')
        
st.sidebar.markdown("""---""")

st.sidebar.markdown('#### Contact: danieli@email.com')
