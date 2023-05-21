import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


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

#ajuste dos dados para a predição
df_predicao.rename(columns={"Ano de eleição":"ano", "Cargo":"cargo", "Cor/raça":"cor_raca","Detalhe da situação de candidatura":"det_sit_cand", "Estado civil":"estado_civil", "Faixa etária":"faixa_etaria", "Gênero":"genero", "Grau de instrução":"instrucao" , "UF": "uf", 'Ocupação': "ocupacao"  }, inplace=True)
df_predicao = df_predicao.dropna()
df_predicao.drop("ocupacao",axis = 1, inplace=True)

def mapacalor(parametro, df, titulo):
    x_label = sorted(list(df['Ano de eleição'].unique()))
    y_label = sorted(list(df[parametro].unique()))

    heatmap_sexo_eleitos = []

    for y in y_label:
        values = []
        for x in x_label:
            df1 = df[df['Ano de eleição'] == x]
            values.append(df1[df1[parametro] == y]['Cargo'].count())
        heatmap_sexo_eleitos.append(values)

    data_set = np.transpose(heatmap_sexo_eleitos)

    plt.style.use("dark_background")
    sns.heatmap(heatmap_sexo_eleitos, cmap='Spectral', annot=True, yticklabels=y_label, xticklabels=x_label, fmt=".0f")

    xlabels = plt.gca().get_xticklabels()
    plt.setp(xlabels, color='white')

    ylabels = plt.gca().get_yticklabels()
    plt.setp(ylabels, color='white')

    plt.title(titulo)
    plt.ylabel(parametro)
    plt.xlabel("Ano de Eleição")

    st.pyplot(plt)
    plt.show()

def mapacalor_porcentagem(parametro, df, titulo):
    x_label = sorted(list(df['Ano de eleição'].unique()))
    y_label = sorted(list(df[parametro].unique()))

    heatmap_sexo_eleitos = []

    for y in y_label:
        values = []
        for x in x_label:
            df1 = df[df['Ano de eleição'] == x]
            count = df1[df1[parametro] == y]['Cargo'].count()
            total = df1['Cargo'].count()
            percentage = (count / total) * 100
            values.append(percentage)
        heatmap_sexo_eleitos.append(values)

    data_set = np.transpose(heatmap_sexo_eleitos)

    plt.style.use("dark_background")
    ax = sns.heatmap(heatmap_sexo_eleitos, cmap='Spectral', annot=True, yticklabels=y_label, xticklabels=x_label, fmt=".1f")

    for t in ax.texts:
        t.set_text(t.get_text() + "%")

    xlabels = plt.gca().get_xticklabels()
    plt.setp(xlabels, color='white')

    ylabels = plt.gca().get_yticklabels()
    plt.setp(ylabels, color='white')

    plt.title(titulo)
    plt.ylabel(parametro)
    plt.xlabel("Ano de Eleição")

    st.pyplot(plt)
    plt.show()



seleciona_base =  st.sidebar.radio("Selecione a base que deseja visualizar informações", ('Candidatos', 'Eleitos', 'Análise de Tendências'))

if seleciona_base == 'Candidatos':

    seleciona_cargo = st.sidebar.radio("Selecione o cargo", ('Governador', 'Prefeito'))
    if seleciona_cargo == 'Governador':

        opcao = st.sidebar.radio("Selecione uma opção", ('Cor/Raça', 'Gênero','Idade', 'Estado Civil', 'Grau de Instrução'))

        if opcao == 'Idade':

            seletc_tipo_grafico = st.selectbox('Selecione o tipo de exibição do gráfico', ('Dados absolutos', 'Dados em porcentagem'))

            if seletc_tipo_grafico == 'Dados absolutos':
                mapacalor('Faixa etária', df_gov_cand, 'Distribuição dos Candidatos')
            if seletc_tipo_grafico == 'Dados em porcentagem':
                mapacalor_porcentagem('Faixa etária', df_gov_cand, 'Distribuição dos Candidatos' )
            st.write('Os valores mencionados no gráfico representam a soma de candidatos para os cargos de Governador e Vice Governador.')

        elif opcao == "Cor/Raça":

            seletc_tipo_grafico = st.selectbox('Selecione o tipo de exibição do gráfico',  ('Dados absolutos', 'Dados em porcentagem'))

            if seletc_tipo_grafico == 'Dados absolutos':
                mapacalor('Cor/raça', df_gov_cand, 'Distribuição dos Candidatos')
            if seletc_tipo_grafico == 'Dados em porcentagem':
                mapacalor_porcentagem('Cor/raça', df_gov_cand, 'Distribuição dos Candidatos')
            st.write('Os valores mencionados no gráfico representam a soma de candidatos para os cargos de Governador e Vice Governador.')

        elif opcao == "Grau de Instrução":

            seletc_tipo_grafico = st.selectbox('Selecione o tipo de exibição do gráfico',  ('Dados absolutos', 'Dados em porcentagem'))

            if seletc_tipo_grafico == 'Dados absolutos':
                mapacalor('Grau de instrução', df_gov_cand, 'Distribuição dos Candidatos')
            if seletc_tipo_grafico == 'Dados em porcentagem':
                mapacalor_porcentagem('Grau de instrução', df_gov_cand, 'Distribuição dos Candidatos')
            st.write('Os valores mencionados no gráfico representam a soma de candidatos para os cargos de Governador e Vice Governador.')

        elif opcao == "Gênero":

            seletc_tipo_grafico = st.selectbox('Selecione o tipo de exibição do gráfico', ('Dados absolutos', 'Dados em porcentagem'))

            if seletc_tipo_grafico == 'Dados absolutos':
               mapacalor('Gênero', df_gov_cand, 'Distribuição dos Candidatos')
            if seletc_tipo_grafico ==  'Dados em porcentagem':
                mapacalor_porcentagem('Gênero', df_gov_cand, 'Distribuição dos Candidatos')
            st.write('Os valores mencionados no gráfico representam a soma de candidatos para os cargos de Governador e Vice Governador.')

        elif opcao == 'Estado Civil':

            seletc_tipo_grafico = st.selectbox('Selecione o tipo de exibição do gráfico',  ('Dados absolutos', 'Dados em porcentagem'))

            if seletc_tipo_grafico == 'Dados absolutos':
                mapacalor('Estado civil', df_gov_cand, 'Distribuição dos Candidatos')
            if seletc_tipo_grafico ==  'Dados em porcentagem':
                mapacalor_porcentagem('Estado civil', df_gov_cand, 'Distribuição dos Candidatos')
            st.write('Os valores mencionados no gráfico representam a soma de candidatos para os cargos de Governador e Vice Governador.')


    if seleciona_cargo == 'Prefeito':

        opcao = st.sidebar.radio("Selecione uma opção", ('Cor/Raça', 'Gênero','Idade', 'Estado Civil', 'Grau de Instrução'))

        if opcao == 'Idade':

            seletc_tipo_grafico = st.selectbox('Selecione o tipo de exibição do gráfico', ('Dados absolutos', 'Dados em porcentagem'))

            if seletc_tipo_grafico == 'Dados absolutos':
                mapacalor('Faixa etária', df_pref_cand, 'Distribuição dos Candidatos')
            if seletc_tipo_grafico == 'Dados em porcentagem':
                mapacalor_porcentagem('Faixa etária', df_pref_cand, 'Distribuição dos Candidatos')

            st.write('Os valores mencionados no gráfico representam a soma de candidatos para os cargos de Prefeito e Vice Prefeito.')
            st.write('Foi encontrada uma inconsistência nos dados do TSE. O ano de nascimento do Prefeito eleito em Arambaré - RS em 2016, no registro de candidatura, consta como 1049')
            st.write('https://divulgacandcontas.tse.jus.br/divulga/#/candidato/2016/2/86320/210000006573')

        elif opcao == "Cor/Raça":

            seletc_tipo_grafico = st.selectbox('Selecione o tipo de exibição do gráfico',('Dados absolutos', 'Dados em porcentagem'))

            if seletc_tipo_grafico == 'Dados absolutos':
                mapacalor('Cor/raça', df_pref_cand, 'Distribuição dos Candidatos')
            if seletc_tipo_grafico == 'Dados em porcentagem':
                mapacalor_porcentagem('Cor/raça', df_pref_cand, 'Distribuição dos Candidatos')

            st.write('Os valores mencionados no gráfico representam a soma de candidatos para os cargos de Prefeito e Vice Prefeito.')

        elif opcao == "Grau de Instrução":

            seletc_tipo_grafico = st.selectbox('Selecione o tipo de exibição do gráfico',  ('Dados absolutos', 'Dados em porcentagem'))

            if seletc_tipo_grafico == 'Dados absolutos':
                mapacalor('Grau de instrução', df_pref_cand, 'Distribuição dos Candidatos')
            if seletc_tipo_grafico == 'Dados em porcentagem':
                mapacalor_porcentagem('Grau de instrução', df_pref_cand, 'Distribuição dos Candidatos')
            st.write('Os valores mencionados no gráfico representam a soma de candidatos para os cargos de Prefeito e Vice Prefeito.')

        elif opcao == "Gênero":

            seletc_tipo_grafico = st.selectbox('Selecione o tipo de exibição do gráfico',('Dados absolutos', 'Dados em porcentagem'))

            if seletc_tipo_grafico == 'Dados absolutos':
                mapacalor('Gênero', df_pref_cand, 'Distribuição dos Candidatos')
            if seletc_tipo_grafico == 'Dados em porcentagem':
                mapacalor_porcentagem('Gênero', df_pref_cand, 'Distribuição dos Candidatos')

            st.write('Os valores mencionados no gráfico representam a soma de candidatos para os cargos de Prefeito e Vice Prefeito.')

        elif opcao == 'Estado Civil':

            seletc_tipo_grafico = st.selectbox('Selecione o tipo de exibição do gráfico',  ('Dados absolutos', 'Dados em porcentagem'))

            if seletc_tipo_grafico == 'Dados absolutos':
                mapacalor('Estado civil', df_pref_cand, 'Distribuição dos Candidatos')
            if seletc_tipo_grafico ==  'Dados em porcentagem':
                mapacalor_porcentagem('Estado civil', df_pref_cand, 'Distribuição dos Candidatos')
            st.write('Os valores mencionados no gráfico representam a soma de candidatos para os cargos de Prefeito e Vice Prefeito.')

    st.write('Fonte: www.tse.gov.br')

if seleciona_base == 'Eleitos':

    seleciona_cargo =  st.sidebar.radio("Selecione o cargo", ('Governador', 'Prefeito'))

    if seleciona_cargo == 'Governador':

        opcao = st.sidebar.radio("Selecione uma opção", ('Cor/Raça', 'Gênero','Idade', 'Estado Civil', 'Grau de Instrução'))

        if opcao == 'Idade':

            seletc_tipo_grafico = st.selectbox('Selecione o tipo de exibição do gráfico', ('Dados absolutos', 'Dados em porcentagem'))

            if seletc_tipo_grafico == 'Dados absolutos':
                mapacalor('Faixa etária', df_gov_eleito, 'Distribuição dos Eleitos')
            if seletc_tipo_grafico == 'Dados em porcentagem':
                mapacalor_porcentagem('Faixa etária', df_gov_eleito, 'Distribuição dos Eleitos')
            st.write('Os valores mencionados no gráfico representam a soma de eleitos para os cargos de Governador e Vice Governador.')

        elif opcao == 'Gênero':

            seletc_tipo_grafico = st.selectbox('Selecione o tipo de exibição do gráfico', ('Dados absolutos', 'Dados em porcentagem'))

            if seletc_tipo_grafico == 'Dados absolutos':
                mapacalor('Gênero', df_gov_eleito, 'Distribuição dos Eleitos')
            if seletc_tipo_grafico == 'Dados em porcentagem':
                mapacalor_porcentagem('Gênero', df_gov_eleito, 'Distribuição dos Eleitos')
            st.write('Os valores mencionados no gráfico representam a soma de eleitos para os cargos de Governador e Vice Governador.')

        elif opcao == "Cor/Raça":

            seletc_tipo_grafico = st.selectbox('Selecione o tipo de exibição do gráfico',('Dados absolutos', 'Dados em porcentagem'))

            if seletc_tipo_grafico == 'Dados absolutos':
                mapacalor('Cor/raça', df_gov_eleito, 'Distribuição dos Eleitos')
            if seletc_tipo_grafico == 'Dados em porcentagem':
                mapacalor_porcentagem('Cor/raça', df_gov_eleito, 'Distribuição dos Eleitos')
            st.write('Os valores mencionados no gráfico representam a soma de eleitos para os cargos de Governador e Vice Governador.')


        elif opcao == "Grau de Instrução":

            seletc_tipo_grafico = st.selectbox('Selecione o tipo de exibição do gráfico', ('Dados absolutos', 'Dados em porcentagem'))

            if seletc_tipo_grafico == 'Dados absolutos':
                mapacalor('Grau de instrução', df_gov_eleito, 'Distribuição dos Eleitos')
            if seletc_tipo_grafico == 'Dados em porcentagem':
                mapacalor_porcentagem('Grau de instrução', df_gov_eleito, 'Distribuição dos Eleitos')
            st.write( 'Os valores mencionados no gráfico representam a soma de eleitos para os cargos de Governador e Vice Governador.')

        elif opcao == 'Estado Civil':

            seletc_tipo_grafico = st.selectbox('Selecione o tipo de exibição do gráfico',  ('Dados absolutos', 'Dados em porcentagem'))

            if seletc_tipo_grafico == 'Dados absolutos':
                mapacalor('Estado civil', df_gov_eleito, 'Distribuição dos Eleitos')
            if seletc_tipo_grafico ==  'Dados em porcentagem':
                mapacalor_porcentagem('Estado civil', df_gov_eleito, 'Distribuição dos Eleitos')
            st.write('Os valores mencionados no gráfico representam a soma de candidatos para os cargos de Governador e Vice Governador.')

    if seleciona_cargo == 'Prefeito':

        opcao = st.sidebar.radio("Selecione uma opção", ('Cor/Raça', 'Gênero','Idade', 'Estado Civil', 'Grau de Instrução'))

        if opcao == 'Idade':
            seletc_tipo_grafico = st.selectbox('Selecione o tipo de exibição do gráfico', ('Dados absolutos', 'Dados em porcentagem'))

            if seletc_tipo_grafico == 'Dados absolutos':
                mapacalor('Faixa etária', df_pref_eleito, 'Distribuição dos Eleitos')
            if seletc_tipo_grafico == 'Dados em porcentagem':
                mapacalor_porcentagem('Faixa etária', df_pref_eleito, 'Distribuição dos Eleitos')
            st.write('Os valores mencionados no gráfico representam a soma de eleitos para os cargos de Prefeito e Vice Prefeito.')
            st.write('Foi encontrada uma inconsistência nos dados do TSE. O ano de nascimento do Prefeito eleito em Arambaré - RS em 2016, no registro de candidatura, consta como 1049')
            st.write('https://divulgacandcontas.tse.jus.br/divulga/#/candidato/2016/2/86320/210000006573')

        elif opcao == 'Gênero':
            seletc_tipo_grafico = st.selectbox('Selecione o tipo de exibição do gráfico', ('Dados absolutos', 'Dados em porcentagem'))

            if seletc_tipo_grafico == 'Dados absolutos':
                mapacalor('Gênero', df_pref_eleito, 'Distribuição dos Eleitos')
            if seletc_tipo_grafico == 'Dados em porcentagem':
                mapacalor_porcentagem('Gênero', df_pref_eleito, 'Distribuição dos Eleitos')
            st.write('Os valores mencionados no gráfico representam a soma de eleitos para os cargos de Prefeito e Vice Prefeito.')


        elif opcao == "Cor/Raça":

            seletc_tipo_grafico = st.selectbox('Selecione o tipo de exibição do gráfico',('Dados absolutos', 'Dados em porcentagem'))

            if seletc_tipo_grafico == 'Dados absolutos':
                mapacalor('Cor/raça', df_pref_eleito, 'Distribuição dos Eleitos')
            if seletc_tipo_grafico == 'Dados em porcentagem':
                mapacalor_porcentagem('Cor/raça', df_pref_eleito, 'Distribuição dos Eleitos')
            st.write('Os valores mencionados no gráfico representam a soma de eleitos para os cargos de Prefeito e Vice Prefeito.')

        elif opcao == "Grau de Instrução":

            seletc_tipo_grafico = st.selectbox('Selecione o tipo de exibição do gráfico', ('Dados absolutos', 'Dados em porcentagem'))

            if seletc_tipo_grafico == 'Dados absolutos':
                mapacalor('Grau de instrução', df_pref_eleito, 'Distribuição dos Eleitos')
            if seletc_tipo_grafico == 'Dados em porcentagem':
                mapacalor_porcentagem('Grau de instrução', df_pref_eleito, 'Distribuição dos Eleitos')
            st.write( 'Os valores mencionados no gráfico representam a soma de eleitos para os cargos de Prefeito e Vice Prefeito.')

        elif opcao == 'Estado Civil':

            seletc_tipo_grafico = st.selectbox('Selecione o tipo de exibição do gráfico',  ('Dados absolutos', 'Dados em porcentagem'))

            if seletc_tipo_grafico == 'Dados absolutos':
                mapacalor('Estado civil', df_pref_eleito, 'Distribuição dos Eleitos')
            if seletc_tipo_grafico ==  'Dados em porcentagem':
                mapacalor_porcentagem('Estado civil', df_pref_eleito, 'Distribuição dos Eleitos')
            st.write('Os valores mencionados no gráfico representam a soma de candidatos para os cargos de Prefeito e Vice Prefeito.')

    st.write('Fonte: www.tse.gov.br')

if seleciona_base == 'Análise de Tendências':

    st.text("Analise de Tendências")

    df_predicao = pd.get_dummies(df_predicao,
                                 columns=['ano', 'cargo', 'cor_raca', 'det_sit_cand', 'estado_civil', 'faixa_etaria',
                                          'genero', 'instrucao', 'uf'])

    # Define as variáveis de entrada (X) e a variável de saída (y)

    X = df_predicao.drop(['Situação de totalização'], axis=1)
    y = df_predicao['Situação de totalização']

    # Cria uma interface para que o usuário escolha as variáveis de entrada
    select_cargo = st.sidebar.selectbox("Selecione o cargo", (  "Presidente", "Prefeito", "Governador", "Vice-prefeito", "Vice-governador", "Vice-presidente",))
    if select_cargo == 'Prefeito' or select_cargo ==  'Vice-prefeito':
        select_ano = st.sidebar.selectbox("Selecione o ano", ("2016", "2020"))
    else:
        select_ano = st.sidebar.selectbox("Selecione o ano", ("2014", "2018", "2022"))
    select_raca = st.sidebar.selectbox("Selecione Cor/Raça", ("Amarela", "Branca", "Indígena", "Parda", "Preta"))
    select_estado_civil = st.sidebar.selectbox("Selecione o Estado Civil", ( "Casado(a)", "Divorciado(a)", "Não divulgável", "Separado(a) judicialmente", "Solteiro(a)", "Viúvo(a)"))
    select_faixa_etaria = st.sidebar.selectbox("Selecione a faixa etária", ( "20 anos", "21 a 24 anos", "25 a 29 anos", "30 a 34 anos", "35 a 39 anos", "40 a 44 anos", "45 a 49 anos","50 a 54 anos", "55 a 59 anos", "60 a 64 anos", "65 a 69 anos", "70 a 74 anos", "75 a 79 anos", "80 a 84 anos","85 a 89 anos", "90 a 94 anos", "95 a 99 anos", "Não divulgável", "100 anos ou mais"))
    select_genero = st.sidebar.selectbox("Selecione o gênero", ("Masculino", "Feminino"))
    select_instrucao = st.sidebar.selectbox("Selecione o grau de instrução", ("Analfabeto", "Lê e escreve", "Ensino Fundamental incompleto", "Ensino Fundamental completo","Ensino Médio incompleto", "Ensino Médio completo", "Superior incompleto", "Superior completo"))
    select_uf = st.sidebar.selectbox("Selecione a UF", ("AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS", "MT", "PA", "PB", "PE", "PI", "PR", "RJ","RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO"))
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

        pessoa = pd.DataFrame(zero_df, index=[0])
        probabilidade = model.predict_proba(pessoa)[:, 1]
        chance = probabilidade * 100
        chance = str(round(chance[0], 1)).replace('.', ',') + '%'

        st.write( f"De acordo com o histórico de candidatos, a probabilidade de uma pessoa com o perfil selecionado seja eleito é de")
        st.markdown(f'<h1 style="color:#FFBF00;font-size:64px;text-align: center;">{chance}</h1>', unsafe_allow_html=True)

        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        acuracia = "{:.2%}".format(accuracy)
        st.write('A acurácia do modelo é:')
        st.markdown(f'<h1 style="color:#FFBF00;font-size:64px;text-align: center;">{acuracia}</h1>', unsafe_allow_html=True)



        st.write('Estes cálculos foram realizados a partir do modelo de machine learning "Random Forest" e por isso, podem apresentar pequenas variações nos resultados apresentados, uma vez que a cada nova interação, são utlizados novos dados de treinamento e teste e, portanto, novas árvores de decisões aleatórias são geradas.')

        st.write('Os cálculos não levam em consideração que, segundo a legislação vigente, a idade mínima para o cargo de governador é 30 anos, e para o cargo de presidente é de 35 anos. A idade mínima para o cargo de prefeito é de 21 anos.')
