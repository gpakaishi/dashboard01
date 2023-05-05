import streamlit as st
import pandas as pd
import numpy as np

df_cand = pd.read_csv('candidatos.csv', sep=';', encoding='latin-1')
df_eleit = pd.read_csv('eleitos.csv', sep=';', encoding='latin-1')

st.header("Dashboard Eleições  ")
st.sidebar.text("Filtros")

print(df_cand.columns)

df_cand.rename(columns={"Ano de eleição":"ano", "Cargo":"cargo", "Cor/raça":"cor_raca","Detalhe da situação de candidatura":"det_sit_cand", "Estado Civil":"Estado Civil", "Faixa etária":"faixa_etaria", "Gênero":"genero", "Grau de instrução":"instrucao", "Município":"municipio",  "Nacionalidade":"nacionalidade" , "Ocupação":"ocupacao", "Sigla partido":"partido" , "Situação de candidatura":"sit_cand","Situação de totalização":"sit_total", "Região": "regiao"  }, inplace=True)

print(df_cand.columns)