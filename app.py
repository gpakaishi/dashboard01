import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier



df_cand = pd.read_csv('candidatos.csv', sep=';', encoding='latin-1')
df_gov_cand = pd.read_csv('candidatos_gov.csv', sep=';', encoding='latin-1')
df_pref_cand = pd.read_csv('candidatos_pref.csv', sep=';', encoding='latin-1')
df_predicao = pd.read_csv('candidatosml.csv', sep = ';', encoding='latin-1')
df_pref_eleito = pd.read_csv('eleitos_pref.csv', sep=';', encoding='latin-1')
df_gov_eleito  = pd.read_csv('eleitos_gov.csv', sep=';', encoding='latin-1')
df_eleit = pd.read_csv('eleitos.csv', sep=';', encoding='latin-1')


df_predicao.drop("Ocupação", axis='columns')

st.header("Dashboard Eleições  ")
st.sidebar.text("Filtros")

#print(df_cand.columns)

df_gov_cand.rename(columns={"Ano de eleição":"ano", "Cargo":"cargo", "Cor/raça":"cor_raca","Detalhe da situação de candidatura":"det_sit_cand", "Estado Civil":"estado_civil", "Faixa etária":"faixa_etaria", "Gênero":"genero", "Grau de instrução":"instrucao", "Município":"municipio",  "Nacionalidade":"nacionalidade" , "Ocupação":"ocupacao", "Sigla partido":"partido" , "Situação de candidatura":"sit_cand","Situação de totalização":"sit_total", "Região": "regiao"  }, inplace=True)
df_pref_cand.rename(columns={"Ano de eleição":"ano", "Cargo":"cargo", "Cor/raça":"cor_raca","Detalhe da situação de candidatura":"det_sit_cand", "Estado Civil":"estadoc_civil", "Faixa etária":"faixa_etaria", "Gênero":"genero", "Grau de instrução":"instrucao", "Município":"municipio",  "Nacionalidade":"nacionalidade" , "Ocupação":"ocupacao", "Sigla partido":"partido" , "Situação de candidatura":"sit_cand","Situação de totalização":"sit_total", "Região": "regiao"  }, inplace=True)
df_gov_eleito.rename(columns={"Ano de eleição":"ano", "Cargo":"cargo", "Cor/raça":"cor_raca","Detalhe da situação de candidatura":"det_sit_cand", "Estado Civil":"estado_civil", "Faixa etária":"faixa_etaria", "Gênero":"genero", "Grau de instrução":"instrucao", "Município":"municipio",  "Nacionalidade":"nacionalidade" , "Ocupação":"ocupacao", "Sigla partido":"partido" , "Situação de candidatura":"sit_cand","Situação de totalização":"sit_total", "Região": "regiao"  }, inplace=True)
df_pref_eleito.rename(columns={"Ano de eleição":"ano", "Cargo":"cargo", "Cor/raça":"cor_raca","Detalhe da situação de candidatura":"det_sit_cand", "Estado Civil":"estado_civil", "Faixa etária":"faixa_etaria", "Gênero":"genero", "Grau de instrução":"instrucao", "Município":"municipio",  "Nacionalidade":"nacionalidade" , "Ocupação":"ocupacao", "Sigla partido":"partido" , "Situação de candidatura":"sit_cand","Situação de totalização":"sit_total", "Região": "regiao"  }, inplace=True)
df_predicao.rename(columns={"Ano de eleição":"ano", "Cargo":"cargo", "Cor/raça":"cor_raca","Detalhe da situação de candidatura":"det_sit_cand", "Estado civil":"estado_civil", "Faixa etária":"faixa_etaria", "Gênero":"genero", "Grau de instrução":"instrucao" , "UF": "uf", 'Ocupação': "ocupacao"  }, inplace=True)


df_predicao = df_predicao.dropna()
df_predicao.drop("ocupacao",axis = 1, inplace=True)


seleciona_cargo =  st.sidebar.radio("Selecione o cargo", ('Governador', 'Prefeito', 'Análise de Tendências'))

if seleciona_cargo == 'Governador':

    opcao = st.sidebar.radio("Selecione uma opção", ('Idade', 'Gênero','Cor/Raça', 'Grau de Instrução'))

    if opcao == 'Idade':
        x_label = sorted(list(df_gov_eleito['ano'].unique()))
        y_label = sorted(list(df_gov_eleito['faixa_etaria'].unique()))
        #cem = y_label.pop(0)
        #y_label.insert(len(y_label) - 1, cem)

        heatmap_sexo_eleitos = list()

        for y in y_label:
            values = list()
            for x in x_label:
                df1 = df_gov_eleito[df_gov_eleito['ano'] == x]
                values.append(df1[df1['faixa_etaria'] == y]['cargo'].count())
            heatmap_sexo_eleitos.append(values)
        data_set = np.transpose(heatmap_sexo_eleitos)


        sns.heatmap(heatmap_sexo_eleitos, cmap='Spectral', annot=True, yticklabels=y_label, xticklabels=x_label )
        #fmt = ',d'
      #  custom_palette = sns.color_palette("RdGy", 11)
        sns.set(rc={ 'figure.facecolor': (0,0,0,0), 'axes.labelcolor': 'black',
             "axes.titlecolor": "black", "legend.labelcolor": "red"})

        xlabels = plt.gca().get_xticklabels()
        plt.setp(xlabels, color='black')

        ylabels = plt.gca().get_yticklabels()
        plt.setp(ylabels, color='black')

        plt.title(f"Distribuição de eleitos")
        plt.ylabel("Ano de Eleição")
        #plt.xticks(range(2013, 2023, 4))
        plt.xlabel("Faixa Etária dos Eleitos")
        st.pyplot(plt)


    elif opcao == 'Gênero':
        st.write("Gênero")

        df_gov_eleito_grouped = df_gov_eleito.groupby(['ano', 'genero']).size().reset_index(name='count')
        df_gov_eleito_pivot = df_gov_eleito_grouped.pivot(index='ano', columns='genero', values='count')
        df_gov_eleito_pivot['proporcao_masculino'] = df_gov_eleito_pivot['Masculino'] / (df_gov_eleito_pivot['Masculino'] + df_gov_eleito_pivot['Feminino'])
        df_gov_eleito_pivot['proporcao_mulher'] = df_gov_eleito_pivot['Feminino'] / (df_gov_eleito_pivot['Masculino'] + df_gov_eleito_pivot['Feminino'])
        plt.plot(df_gov_eleito_pivot['proporcao_masculino'], label='Masculino')
        plt.plot(df_gov_eleito_pivot['proporcao_mulher'], label='Feminino')
      #  plt.savefig("grafico-background-05.png", dpi=300, transparent='True')
        plt.xticks(range(2014, 2023, 4))
        plt.legend()
        st.pyplot(plt)

    elif opcao == "Cor/Raça":

            x_label = sorted(list(df_gov_eleito['ano'].unique()))
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
            plt.xlabel("Distribuição de Cor e Raça dos eleitos")
            st.pyplot(plt)

    elif opcao == "Grau de Instrução":

        x_label = sorted(list(df_gov_eleito['ano'].unique()))
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
        x_label = sorted(list(df_pref_eleito['ano'].unique()))
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

        x_label = sorted(list(df_pref_eleito['ano'].unique()))
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

        x_label = sorted(list(df_pref_eleito['ano'].unique()))
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

if seleciona_cargo == 'Análise de Tendências':
    st.text("Analise de Tendências")

    st.write("De acordo com o histórico de candidatos")


    df_predicao = pd.get_dummies(df_predicao, columns=['ano', 'cargo', 'cor_raca', 'det_sit_cand', 'estado_civil', 'faixa_etaria','genero', 'instrucao', 'uf'])

    # Define as variáveis de entrada (X) e a variável de saída (y)

    X = df_predicao.drop(['Situação de totalização'], axis=1)
    y = df_predicao['Situação de totalização']

    # Cria uma interface para que o usuário escolha as variáveis de entrada
    select_ano = st.sidebar.selectbox("Selecione o ano", ("2014", "2016", "2018", "2020", "2022"))
    select_cargo = st.sidebar.selectbox("Selecione o cargo", ("Presidente" ,"Prefeito ","Governador ","Vice-prefeito","Vice-governador ","Vice-presidente ",))
    select_raca = st.sidebar.selectbox("Selecione Cor/Raça", ("Amarela", "Branca", "Indígena", "Parda", "Preta"))
    select_estado_civil = st.sidebar.selectbox("Selecione o Estado Civil", ("Casado(a)" ,"Divorciado(a) ","Não divulgável","Separado(a) judicialmente ","Solteiro(a) ","Viúvo(a)"))
    select_faixa_etaria = st.sidebar.selectbox("Selecione a faixa etária", ("20 anos","21 a 24 anos","25 a 29 anos","30 a 34 anos","35 a 39 anos","40 a 44 anos","45 a 49 anos","50 a 54 anos","55 a 59 anos","60 a 64 anos ","65 a 69 anos","70 a 74 anos","75 a 79 anos","80 a 84 anos","85 a 89 anos","90 a 94 anos","95 a 99 anos","Não divulgável","100 anos ou mais"))
    select_genero = st.sidebar.selectbox("Selecione o gênero", ("Masculino", "Feminino"))
    select_instrucao = st.sidebar.selectbox("Selecione o grau de instrução", ("Analfabeto", "Lê e escreve", "Ensino Fundamental incompleto", "Ensino Fundamental completo","Ensino Médio incompleto", "Ensino Médio completo", "Superior incompleto", "Superior completo"))
    select_uf = st.sidebar.selectbox("Selecione a UF", ("AC" ,"AL","AM","AP","BA","CE","DF","ES","GO","MA","MG","MS","MT ","PA","PB","PE","PI ","PR","RJ","RN","RO","RR","RS ","SC","SE","SP","TO",))
    # Filtra os dados de acordo com as escolhas do usuário

    if st.sidebar.button("Calcular"):
        target = np.where(df_predicao['Situação de totalização'] == 'Eleito', 1, 0)


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)
        model = RandomForestClassifier(n_estimators=100, random_state=None)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]

        zero_df = pd.DataFrame(columns=df_predicao.columns)
        for col in df_predicao.columns:
            zero_df[col] = [0]

        zero_df = zero_df.drop('Situação de totalização', axis=1)

       # zero_df = zero_df.loc[(zero_df[0:0, 'ano_' + select_ano] = 1) & (zero_df[0:0,'raca_' + select_raca] = 1) & (zero_df[0:0,'genero_' + select_genero] = 1) & ( zero_df[0:0,'instrucao_' + select_instrucao] = 1)  & (zero_df[0:0,'cargo_' + select_cargo] = 1) & (zero_df[0:0,'estado_civil_' + select_estado_civil] = 1) & (0:0,zero_df['faixa_etaria_' + select_faixa_etaria] =1)  & (zero_df[0:0,'UF_' + select_uf] =1) & (zero_df[0:0,'det_sit_cand_deferido'] =1)]

        zero_df['ano_' + select_ano] = 1
        zero_df['cor_raca_' + select_raca] = 1
        zero_df['genero_' + select_genero] = 1
        zero_df['instrucao_' + select_instrucao] = 1
        zero_df['cargo_' + select_cargo] = 1
        zero_df['estado_civil_' + select_estado_civil] = 1
        zero_df['faixa_etaria_' + select_faixa_etaria] = 1
        zero_df['uf_' + select_uf] = 1
        zero_df['det_sit_cand_Deferido'] = 1

        # Divide os dados em conjunto de treino e teste
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        pessoa = pd.DataFrame(zero_df, index=[0])
        probabilidade = model.predict_proba(pessoa)[:, 1]
        chance = probabilidade*100
        chance = str(round(chance[0],1)).replace('.',',')+'%'


        st.write(f"A probabilidade de que o perfil selecionado seja eleito é de")
        st.markdown(f'<h1 style="color:#FFBF00;font-size:64px;text-align: center;">{chance}</h1>',
                    unsafe_allow_html=True)

# st.line_chart(df_eleit.loc[:,['genero']])