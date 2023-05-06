import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df_cand = pd.read_csv('candidatos.csv', sep=';', encoding='latin-1')
df_gov_cand = pd.read_csv('candidatos_gov.csv', sep=';', encoding='latin-1')
df_pref_cand = pd.read_csv('candidatos_pref.csv', sep=';', encoding='latin-1')

df_pref_eleito = pd.read_csv('eleitos_pref.csv', sep=';', encoding='latin-1')
df_gov_eleito  = pd.read_csv('eleitos_gov.csv', sep=';', encoding='latin-1')
df_eleit = pd.read_csv('eleitos.csv', sep=';', encoding='latin-1')


st.header("Dashboard Eleições  ")
st.sidebar.text("Filtros")

#print(df_cand.columns)

df_gov_cand.rename(columns={"Ano de eleição":"ano", "Cargo":"cargo", "Cor/raça":"cor_raca","Detalhe da situação de candidatura":"det_sit_cand", "Estado Civil":"Estado Civil", "Faixa etária":"faixa_etaria", "Gênero":"genero", "Grau de instrução":"instrucao", "Município":"municipio",  "Nacionalidade":"nacionalidade" , "Ocupação":"ocupacao", "Sigla partido":"partido" , "Situação de candidatura":"sit_cand","Situação de totalização":"sit_total", "Região": "regiao"  }, inplace=True)
df_pref_cand.rename(columns={"Ano de eleição":"ano", "Cargo":"cargo", "Cor/raça":"cor_raca","Detalhe da situação de candidatura":"det_sit_cand", "Estado Civil":"Estado Civil", "Faixa etária":"faixa_etaria", "Gênero":"genero", "Grau de instrução":"instrucao", "Município":"municipio",  "Nacionalidade":"nacionalidade" , "Ocupação":"ocupacao", "Sigla partido":"partido" , "Situação de candidatura":"sit_cand","Situação de totalização":"sit_total", "Região": "regiao"  }, inplace=True)
df_gov_eleito.rename(columns={"Ano de eleição":"ano", "Cargo":"cargo", "Cor/raça":"cor_raca","Detalhe da situação de candidatura":"det_sit_cand", "Estado Civil":"Estado Civil", "Faixa etária":"faixa_etaria", "Gênero":"genero", "Grau de instrução":"instrucao", "Município":"municipio",  "Nacionalidade":"nacionalidade" , "Ocupação":"ocupacao", "Sigla partido":"partido" , "Situação de candidatura":"sit_cand","Situação de totalização":"sit_total", "Região": "regiao"  }, inplace=True)
df_pref_eleito.rename(columns={"Ano de eleição":"ano", "Cargo":"cargo", "Cor/raça":"cor_raca","Detalhe da situação de candidatura":"det_sit_cand", "Estado Civil":"Estado Civil", "Faixa etária":"faixa_etaria", "Gênero":"genero", "Grau de instrução":"instrucao", "Município":"municipio",  "Nacionalidade":"nacionalidade" , "Ocupação":"ocupacao", "Sigla partido":"partido" , "Situação de candidatura":"sit_cand","Situação de totalização":"sit_total", "Região": "regiao"  }, inplace=True)


seleciona_cargo =  st.sidebar.radio("Selecione o cargo", ('Governador', 'Prefeito'))
if seleciona_cargo == 'Governador':

    opcao = st.sidebar.radio("Selecione uma opção", ('Idade', 'Gênero','Cor/Raça', 'Grau de Instrução'))


    if opcao == 'Idade':
        x_label = df_gov_eleito['ano'].unique()
        y_label = df_gov_eleito['faixa_etaria'].unique()

        heatmap_sexo_eleitos = list()

        for y in y_label:
            values = list()
            for x in x_label:
                df1 = df_gov_eleito[df_gov_eleito['ano'] == x]
                values.append(df1[df1['faixa_etaria'] == y]['cargo'].count())
            heatmap_sexo_eleitos.append(values)
        data_set = np.transpose(heatmap_sexo_eleitos)

        sns.heatmap(heatmap_sexo_eleitos, annot=True, yticklabels=y_label, xticklabels=x_label, fmt=',d', linewidths=.3)
        plt.title(f"Distribuição de eleitos")
        plt.ylabel("Ano de Eleição")
#        plt.xticks(range(2014, 2023, 4))
        plt.xlabel("Faixa Etária dos Eleitos")
        st.pyplot(plt)


    elif opcao == 'Gênero':
        st.write("Gênero")

        df_gov_eleito_grouped = df_gov_eleito.groupby(['ano', 'genero']).size().reset_index(name='count')
        df_gov_eleito_pivot = df_gov_eleito_grouped.pivot(index='ano', columns='genero', values='count')
        df_gov_eleito_pivot['proporcao_masculino'] = df_gov_eleito_pivot['Masculino'] / (df_gov_eleito_pivot['Masculino'] + df_gov_eleito_pivot['Feminino'])
        df_gov_eleito_pivot['proporcao_mulher'] = df_gov_eleito_pivot['Feminino'] / (df_gov_eleito_pivot['Masculino'] + df_gov_eleito_pivot['Feminino'])

        plt.plot(df_gov_eleito_pivot['proporcao_masculino'], label='Homens')
        plt.plot(df_gov_eleito_pivot['proporcao_mulher'], label='Mulheres')
       # plt.savefig("grafico-background-05.png", dpi=300, transparent='True')
        plt.xticks(range(2014, 2023, 4))
        plt.legend()
        #plt.show()
        st.pyplot(plt)

    elif opcao == "Cor/Raça":

            x_label = df_gov_eleito['ano'].unique()
            y_label = df_gov_eleito['cor_raca'].unique()

            heatmap_sexo_eleitos = list()

            for y in y_label:
                values = list()
                for x in x_label:
                    df1 = df_gov_eleito[df_gov_eleito['ano'] == x]
                    values.append(df1[df1['cor_raca'] == y]['cargo'].count())
                heatmap_sexo_eleitos.append(values)
            data_set = np.transpose(heatmap_sexo_eleitos)

            sns.heatmap(heatmap_sexo_eleitos, annot=True, yticklabels=y_label, xticklabels=x_label, fmt=',d',
                        linewidths=.3)
            plt.title(f"Distribuição de eleitos")
            plt.ylabel("Ano de Eleição")
            #        plt.xticks(range(2014, 2023, 4))
            plt.xlabel("Distribuição de Cor e Raça dos eleitos")
            st.pyplot(plt)

    elif opcao == "Grau de Instrução":

        x_label = df_gov_eleito['ano'].unique()
        y_label = df_gov_eleito['instrucao'].unique()

        heatmap_sexo_eleitos = list()

        for y in y_label:
            values = list()
            for x in x_label:
                df1 = df_gov_eleito[df_gov_eleito['ano'] == x]
                values.append(df1[df1['instrucao'] == y]['cargo'].count())
            heatmap_sexo_eleitos.append(values)
        data_set = np.transpose(heatmap_sexo_eleitos)

        sns.heatmap(heatmap_sexo_eleitos, annot=True, yticklabels=y_label, xticklabels=x_label, fmt=',d',
                    linewidths=.3)
        plt.title(f"Distribuição de eleitos")
        plt.ylabel("Ano de Eleição")
        #        plt.xticks(range(2014, 2023, 4))
        plt.xlabel("Distribuição de Grau de Instrução")
        st.pyplot(plt)

#st.line_chart(df_eleit.loc[:,['genero']])

if seleciona_cargo == 'Prefeito':

    opcao = st.sidebar.radio("Selecione uma opção", ('Idade', 'Gênero', 'Cor/Raça', 'Grau de Instrução'))

    if opcao == 'Idade':
        x_label = df_pref_eleito['ano'].unique()
        y_label = df_pref_eleito['faixa_etaria'].unique()

        heatmap_sexo_eleitos = list()

        for y in y_label:
            values = list()
            for x in x_label:
                df1 = df_pref_eleito[df_pref_eleito['ano'] == x]
                values.append(df1[df1['faixa_etaria'] == y]['cargo'].count())
            heatmap_sexo_eleitos.append(values)
        data_set = np.transpose(heatmap_sexo_eleitos)

        sns.heatmap(heatmap_sexo_eleitos, annot=True, yticklabels=y_label, xticklabels=x_label, fmt=',d', linewidths=.3)
        plt.title(f"Distribuição de eleitos")
        plt.ylabel("Ano de Eleição")
        #        plt.xticks(range(2014, 2023, 4))
        plt.xlabel("Faixa Etária dos Eleitos")
        st.pyplot(plt)


    elif opcao == 'Gênero':
        st.write("Gênero")

        df_pref_eleito_grouped = df_pref_eleito.groupby(['ano', 'genero']).size().reset_index(name='count')
        df_pref_eleito_pivot = df_pref_eleito_grouped.pivot(index='ano', columns='genero', values='count')
        df_pref_eleito_pivot['proporcao_masculino'] = df_pref_eleito_pivot['Masculino'] / (
                    df_pref_eleito_pivot['Masculino'] + df_pref_eleito_pivot['Feminino'])
        df_pref_eleito_pivot['proporcao_mulher'] = df_pref_eleito_pivot['Feminino'] / (
                    df_pref_eleito_pivot['Masculino'] + df_pref_eleito_pivot['Feminino'])

        plt.plot(df_pref_eleito_pivot['proporcao_masculino'], label='Homens')
        plt.plot(df_pref_eleito_pivot['proporcao_mulher'], label='Mulheres')
        # plt.savefig("grafico-background-05.png", dpi=300, transparent='True')
        plt.xticks(range(2014, 2023, 4))
        plt.legend()
        # plt.show()
        st.pyplot(plt)

    elif opcao == "Cor/Raça":

        x_label = df_pref_eleito['ano'].unique()
        y_label = df_pref_eleito['cor_raca'].unique()

        heatmap_sexo_eleitos = list()

        for y in y_label:
            values = list()
            for x in x_label:
                df1 = df_pref_eleito[df_pref_eleito['ano'] == x]
                values.append(df1[df1['cor_raca'] == y]['cargo'].count())
            heatmap_sexo_eleitos.append(values)
        data_set = np.transpose(heatmap_sexo_eleitos)

        sns.heatmap(heatmap_sexo_eleitos, annot=True, yticklabels=y_label, xticklabels=x_label, fmt=',d',
                    linewidths=.3)
        plt.title(f"Distribuição de eleitos")
        plt.ylabel("Ano de Eleição")
        #        plt.xticks(range(2014, 2023, 4))
        plt.xlabel("Distribuição de Cor e Raça dos eleitos")
        st.pyplot(plt)

    elif opcao == "Grau de Instrução":

        x_label = df_pref_eleito['ano'].unique()
        y_label = df_pref_eleito['instrucao'].unique()

        heatmap_sexo_eleitos = list()

        for y in y_label:
            values = list()
            for x in x_label:
                df1 = df_pref_eleito[df_pref_eleito['ano'] == x]
                values.append(df1[df1['instrucao'] == y]['cargo'].count())
            heatmap_sexo_eleitos.append(values)
        data_set = np.transpose(heatmap_sexo_eleitos)

        sns.heatmap(heatmap_sexo_eleitos, annot=True, yticklabels=y_label, xticklabels=x_label, fmt=',d',
                    linewidths=.3)
        plt.title(f"Distribuição de eleitos")
        plt.ylabel("Ano de Eleição")
        #        plt.xticks(range(2014, 2023, 4))
        plt.xlabel("Distribuição de Grau de Instrução")
        st.pyplot(plt)

# st.line_chart(df_eleit.loc[:,['genero']])