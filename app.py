# import all the app dependencies
import pandas as pd
import numpy as np
import sklearn
import streamlit as st
import joblib
import matplotlib
from IPython import get_ipython
from PIL import Image

# load the encoder and model object
model = joblib.load('rta_model_deploy3.joblib' )
encoder = joblib.load('ordinal_encoder2.joblib')

st.set_option('deprecation.showPyplotGlobalUse', False)

# 1: lesion seria, 2: Lesion leve, 0: Lesion Fatal 

st.set_page_config(page_title="Prediccion de Severidad de accidentes",
                page_icon="🚧", layout="wide")

# creando opciones para las listas desplegables 
options_day = ['Sunday', "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
options_age = ['18-30', '31-50', 'Over 51', 'Unknown', 'Under 18']

# number of vehicle involved: range of 1 to 7
# number of casualties: range of 1 to 8
# hour of the day: range of 0 to 23

options_types_collision = ['Vehicle with vehicle collision','Collision with roadside objects',
                           'Collision with pedestrians','Rollover','Collision with animals',
                           'Unknown','Collision with roadside-parked vehicles','Fall from vehicles',
                           'Other','With Train']

options_sex = ['Male','Female','Unknown']

options_education_level = ['Junior high school','Elementary school','High school',
                           'Unknown','Above high school','Writing & reading','Illiterate']

options_services_year = ['Unknown','2-5yrs','Above 10yr','5-10yrs','1-2yr','Below 1yr']

options_acc_area = ['Other', 'Office areas', 'Residential areas', ' Church areas',
       ' Industrial areas', 'School areas', '  Recreational areas',
       ' Outside rural areas', ' Hospital areas', '  Market areas',
       'Rural village areas', 'Unknown', 'Rural village areasOffice areas',
       'Recreational areas']

# features list
features = ['Number_of_vehicles_involved','Number_of_casualties','Hour_of_Day','Type_of_collision','Age_band_of_driver','Sex_of_driver',
       'Educational_level','Service_year_of_vehicle','Day_of_week','Area_accident_occured']

page_bg_css = """
<style> .stApp { background-color: #bdbdbd; }</style>"""
# Aplicar el CSS a la página
st.markdown(page_bg_css, unsafe_allow_html=True)
# Give a title to web app using html syntax
st.markdown("<h1 style='text-align: center; font-weight: bold; font-size: 50px;color: black;'> 🚧 Predicción de Severidad de accidentes 🚧</h1>"
    , unsafe_allow_html=True)

# define a main() function to take inputs from user in form based approach
def main():
       with st.form("Severidad de accidentes de trafico"):
              st.subheader("Por favor ingresa los siguientes datos :")
              
              No_vehicles = st.slider("Numero de vehiculos involucrados:",1,7, value=0, format="%d")
              No_casualties = st.slider("Numero de casualidades:",1,8, value=0, format="%d")
              Hour = st.slider("Hora del dia:", 0, 23, value=0, format="%d")
              collision = st.selectbox("Tipo de colision:",options=options_types_collision)
              Age_band = st.selectbox("Grupo de edad del conductor?:", options=options_age)
              Sex = st.selectbox("Sexo del conductor:", options=options_sex)
              Education = st.selectbox("Educación del conductor:",options=options_education_level)
              service_vehicle = st.selectbox("año del ultimo servicio:", options=options_services_year)
              Day_week = st.selectbox("Dia de la semana:", options=options_day)
              Accident_area = st.selectbox("Lugar del accidente:", options=options_acc_area)
              
              submit = st.form_submit_button("Predecir")

# encode using ordinal encoder and predict
       if submit:
              input_array = np.array([collision,
                                   Age_band,Sex,Education,service_vehicle,
                                   Day_week,Accident_area], ndmin=2)
              
              encoded_arr = list(encoder.transform(input_array).ravel())
              
              num_arr = [No_vehicles,No_casualties,Hour]
              pred_arr = np.array(num_arr + encoded_arr).reshape(1,-1)              
          
# predict the target from all the input features
              prediction = model.predict(pred_arr)
              
              if prediction == 0:
                     st.write(f"La predicción es: Lesiones fatales")
              elif prediction == 1:
                     st.write(f"La predicción es: Lesiones serias")
              else:
                     st.write(f"La predicción es: Lesiones ligeras")
               
              st.write("Elaborado por: Joshua Esquivel")
              st.markdown("""Reach out to me on: |
              [Linkedin](www.linkedin.com/in/jgiovannie) |
              [GitHub](https://github.com/JGIOVANNIE) 
              """)

a,b,c = st.columns([0.25,0.5,0.25])
with b:
  st.image(image="acc.jpeg", use_column_width=True)


# Descripcion del proyecto y del codigo             
st.markdown("<h2 style='text-align: center; font-weight: bold; font-size: 30px;color:black ;'> 🧾🧾Descripción:🧾🧾 </h2>",unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-weight: italic; font-size: 18px;color:black ;'> Este conjunto de datos se recopiló de los departamentos de policía de la Sub-ciudad de Addis Abeba para un trabajo de investigación de maestría. El conjunto de datos se ha preparado a partir de registros manuales de accidentes de tráfico del año 2017-20. Toda la información sensible se ha excluido durante la codificación de datos y finalmente tiene 32 características y 12316 instancias del accidente. Luego se preprocesa y se identifican las principales causas del accidente analizándolo utilizando diferentes algoritmos de clasificación de aprendizaje automático.</p> ",
             unsafe_allow_html=True)

st.markdown("Fuente del dataset: [Click Here](https://www.narcis.nl/dataset/RecordID/oai%3Aeasy.dans.knaw.nl%3Aeasy-dataset%3A191591)")

st.markdown("<h2 style='text-align:center; font-weight:bold; font-size: 30px; color:black ;'>🧭 Problema :🧭 </h2>",unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-weight: italic; font-size: 18px;color:black ;'> La característica objetivo es Accident_severity, que es una variable multiclase. La tarea es clasificar esta variable en base a las otras 31 características paso a paso, realizando cada tarea diaria. La métrica para la evaluación será el f1-score.</p> ",
              unsafe_allow_html=True)

st.markdown("<p style='font-size:14px; font-weight:italik;color:black'> Encuentra mi repositorio en: <a href='https://github.com/JGIOVANNIE/severidad_de_accidentes-STREAMLIT-' class='hyperlink'>Click Here</a> </p>",unsafe_allow_html=True) 

st.markdown("<p style='font-size:14px; font-weight:italik;color:black'> Siguiendo los pasos del perfil de <a href='https://www.kaggle.com/avikumart' class='hyperlink'>avikumart</a>, se logro este proyecto con éxito </p>",unsafe_allow_html=True) 

pie_html = """ <style content: '''Encuentrame en: | [Linkedin](www.linkedin.com/in/jgiovannie) | [GitHub](https://github.com/JGIOVANNIE)  [Correo](jg.esquivel@outlook.com) '''); display: block; position: relative; padding: 10px; top: 2px; color: withe; background-color: #0E1117 text-align: center; font-size: 12px;} </style>"""
st.markdown(pie_html, unsafe_allow_html=True)                
   
# run the main function               
if __name__ == '__main__':
   main()              