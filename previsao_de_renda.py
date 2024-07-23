import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import requests

# Função para filtrar baseado na multiseleção de categorias
@st.cache_data
def multiselect_filter(relatorio, col, selecionados):
    if 'all' in selecionados:
        return relatorio
    else:
        return relatorio[relatorio[col].isin(selecionados)].reset_index(drop=True)

# Função principal da aplicação    
def main():
    sns.set(context='talk', style='ticks')

    st.set_page_config(
        page_title="Previsão de renda",
        page_icon="https://img.freepik.com/vetores-premium/icone-financeiro-de-crescimento-de-renda-de-grafico-de-dinheiro_71374-1953.jpg",
        layout="wide",
    )

    st.write('# Análise exploratória da previsão de renda')

    renda = pd.read_csv(r'https://raw.githubusercontent.com/DanielaDF13/Projeto_Previsao_de_renda/main/previsao_de_renda_csv.csv')
    
    renda.data_ref = pd.to_datetime(renda.data_ref)
    
    with st.sidebar.form(key='my_form'):

        #CALENDARIO
        min_data = renda.data_ref.min()
        max_data = renda.data_ref.max()

        data_inicial = st.date_input('Data inicial',
                                     value=min_data,
                                     min_value=min_data,
                                     max_value=max_data)
        data_final = st.date_input('Data final',
                                   value=max_data,
                                   min_value=min_data,
                                   max_value=max_data)

        renda = renda[(renda['data_ref'] <= pd.to_datetime(data_final)) & (renda['data_ref'] >= pd.to_datetime(data_inicial))]

        #SLIDER RENDA
        min_slider = renda.renda.min()
        max_slider = renda.renda.max()

        renda_original = renda['renda'].copy()
        renda_sort = renda['renda'].sort_values()

        min_renda_slider, max_renda_slider = st.select_slider('Selecione a amplitude da renda',
                                                              options=renda_sort,
                                                              value=(min_slider, max_slider))
        renda = renda[(renda['renda'] >= min_renda_slider) & (renda['renda'] <= max_renda_slider)]

        renda['renda'] = renda_original

        # POSSE DE IMÓVEL
        posse_de_imovel_list = renda.posse_de_imovel.unique().tolist()
        posse_de_imovel_list.append('all')
        posse_de_imovel_selected = st.multiselect("Posse de imóvel", posse_de_imovel_list, ['all'])

        # POSSE DE VEÍCULO
        posse_de_veiculo_list = renda.posse_de_veiculo.unique().tolist()
        posse_de_veiculo_list.append('all')
        posse_de_veiculo_selected = st.multiselect("Posse de veículo", posse_de_veiculo_list, ['all'])

        # QUANTIDADE DE FILHOS
        qtd_filhos_list = renda.qtd_filhos.unique().tolist()
        qtd_filhos_list.append('all')
        qtd_filhos_selected = st.multiselect("Quantide de filhos", qtd_filhos_list, ['all'])

        # TIPO DE RENDA
        tipo_renda_list = renda.tipo_renda.unique().tolist()
        tipo_renda_list.append('all')
        tipo_renda_selected = st.multiselect("Tipo de renda", tipo_renda_list, ['all'])

        # EDUCAÇÃO
        educacao_list = renda.educacao.unique().tolist()
        educacao_list.append('all')
        educacao_selected = st.multiselect("Educação", educacao_list, ['all'])

        # ESTADO CIVIL
        estado_civil_list = renda.estado_civil.unique().tolist()
        estado_civil_list.append('all')
        estado_civil_selected = st.multiselect("Estado civil", estado_civil_list, ['all'])

        # TIPO DE RESIDÊNCIA
        tipo_residencia_list = renda.tipo_residencia.unique().tolist()
        tipo_residencia_list.append('all')
        tipo_residencia_selected = st.multiselect("Tipo de residência", tipo_residencia_list, ['all'])

        renda = (renda
                    .pipe(multiselect_filter, 'posse_de_imovel', posse_de_imovel_selected)
                    .pipe(multiselect_filter, 'posse_de_veiculo', posse_de_veiculo_selected)
                    .pipe(multiselect_filter, 'qtd_filhos', qtd_filhos_selected)
                    .pipe(multiselect_filter, 'tipo_renda', tipo_renda_selected)
                    .pipe(multiselect_filter, 'educacao', educacao_selected)
                    .pipe(multiselect_filter, 'estado_civil', estado_civil_selected)
                    .pipe(multiselect_filter, 'tipo_residencia', tipo_residencia_selected)
                )
        
        submit_button = st.form_submit_button(label='Aplicar')

    # Plots
    st.write('## Gráficos ao longo do tempo')
    fig, ax = plt.subplots(8, 1, figsize=(20, 70))

    renda[['posse_de_imovel', 'renda']].plot(kind='hist', ax=ax[0])
    ax[0].set(xlabel='', ylabel='Frequência')

    sns.lineplot(x='data_ref', y='renda', hue='posse_de_imovel', data=renda, ax=ax[1])
    ax[1].tick_params(axis='x', rotation=45)
    ax[1].set(xlabel='', ylabel='Renda')

    sns.lineplot(x='data_ref', y='renda', hue='posse_de_veiculo', data=renda, ax=ax[2])
    ax[2].tick_params(axis='x', rotation=45)
    ax[2].set(xlabel='', ylabel='Renda')

    sns.lineplot(x='data_ref', y='renda', hue='qtd_filhos', data=renda, ax=ax[3])
    ax[3].tick_params(axis='x', rotation=45)
    ax[3].set(xlabel='', ylabel='Renda')

    sns.lineplot(x='data_ref', y='renda', hue='tipo_renda', data=renda, ax=ax[4])
    ax[4].tick_params(axis='x', rotation=45)
    ax[4].set(xlabel='', ylabel='Renda')

    sns.lineplot(x='data_ref', y='renda', hue='educacao', data=renda, ax=ax[5])
    ax[5].tick_params(axis='x', rotation=45)
    ax[5].set(xlabel='', ylabel='Renda')

    sns.lineplot(x='data_ref', y='renda', hue='estado_civil', data=renda, ax=ax[6])
    ax[6].tick_params(axis='x', rotation=45)
    ax[6].set(xlabel='', ylabel='Renda')

    sns.lineplot(x='data_ref', y='renda', hue='tipo_residencia', data=renda, ax=ax[7])
    ax[7].tick_params(axis='x', rotation=45)
    ax[7].set(xlabel='', ylabel='Renda')

    sns.despine()
    st.pyplot(fig)

    st.write('## Gráficos bivariada')
    fig, ax = plt.subplots(7, 1, figsize=(15, 50))

    sns.barplot(x='posse_de_imovel', y='renda', data=renda, ax=ax[0])
    ax[0].set(xlabel='Posse de imóvel', ylabel='Renda')

    sns.barplot(x='posse_de_veiculo', y='renda', data=renda, ax=ax[1])
    ax[1].set(xlabel='Posse de veículo', ylabel='Renda')

    sns.barplot(x='qtd_filhos', y='renda', data=renda, ax=ax[2])
    ax[2].set(xlabel='Quantidade de filhos', ylabel='Renda')

    sns.barplot(x='tipo_renda', y='renda', data=renda, ax=ax[3])
    ax[3].set(xlabel='Tipo de renda', ylabel='Renda')

    sns.barplot(x='educacao', y='renda', data=renda, ax=ax[4])
    ax[4].set(xlabel='Educação', ylabel='Renda')

    sns.barplot(x='estado_civil', y='renda', data=renda, ax=ax[5])
    ax[5].set(xlabel='Estado civil', ylabel='Renda')

    sns.barplot(x='tipo_residencia', y='renda', data=renda, ax=ax[6])
    ax[6].set(xlabel='Tipo de residência', ylabel='Renda')

    sns.despine()
    st.pyplot(fig)

    
    #REGREÇÃO LINEAR
    st.markdown('------')
    st.write('## Regreção linear')
    renda = (renda
                   .dropna(axis=0)
                   .drop(columns=['Unnamed: 0'])
                   .drop(columns=['id_cliente']))
       
    numeric_columns = renda.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_columns) > 1:

        st.sidebar.subheader('Seleção das Variáveis para regreção linear:')
        X_col = st.sidebar.selectbox('Selecione a variável independente (X):', options=numeric_columns)
        y_col = st.sidebar.selectbox('Selecione a variável dependente (y):', options=numeric_columns)

        X = renda[[X_col]]
        y = renda[y_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=88)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.subheader('Desempenho do Modelo:')
        st.write(f'Erro Quadrático Médio (MSE): {mse:.2f}')
        st.write(f'Coeficiente de Determinação (R²): {r2:.2f}')

        st.subheader('Visualização dos Resultados:')
        st.write('Gráfico de Dispersão com a Linha de Regressão:')

        plt.figure(figsize=(10, 6))
        plt.scatter(X_test, y_test, color='blue', label='Dados de Teste')
        plt.plot(X_test, y_pred, color='red', linewidth=2, label='Linha de Regressão')
        plt.xlabel(X_col)
        plt.ylabel(y_col)
        plt.legend()
        st.pyplot(plt)

    else:
        st.error('Não há colunas numéricas suficientes no arquivo CSV para realizar a regressão linear.')

    st.markdown('------')
    st.write('## Análise')
    st.markdown(' No caso de imóveis, as tendências de valores de renda não são claras, pois as linhas nbestão entrelaçadas.')
    st.markdown('No caso de veículos, o gráfico claramente mostra que os proprietários de carro têm, em sua maioria, rendas mais altas.')
    st.markdown('Quanto ao tipo de renda, os pensionistas e bolsistas tendem a ter uma renda menor em comparação com os outros tipos, sendo os servidores públicos aqueles que geralmente têm os maiores valores.')
    st.markdown('Na educação observa-se uma grande variação na pós-graduação. Até cerca da metade do gráfico, os dados são semelhantes, enquanto na outra metade, pessoas com ensino secundário completo e ensino superior completo apresentam faixas de renda superiores às demais.')
    st.markdown('No estado civil, as linhas estão mais separadas, com os viúvos geralmente tendo rendas menores e os casados as maiores.')
    st.markdown('A renda parece flutuar dentro de uma faixa de valores na variável de tipo de residência. Alguns tipos, como estúdio, governamental, aluguel e comunitário, têm picos específicos, o que não os torna bons indicadores de renda.')
    st.markdown('------')

if __name__ == '__main__':
    main()