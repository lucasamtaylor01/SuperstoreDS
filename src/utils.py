import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller

def clean_data(df):

    # Padronização de colunas string
    df.columns = df.columns.str.upper()
    df.columns = df.columns.str.replace(' ', '_')
    string_cols = df.select_dtypes(include=['object', 'string']).columns
    df[string_cols] = df[string_cols].apply(lambda x: x.astype(str).str.strip().str.upper())
    df = df.rename(columns={'SUB-CATEGORY': 'SUB_CATEGORY'})
    
    # Eliminação de colunas irrelevantes
    df = df.drop(columns=['CUSTOMER_NAME', 'POSTAL_CODE', 'ORDER_ID'])

    # Tratamento de datas
    df['ORDER_DATE'] = pd.to_datetime(df['ORDER_DATE'])
    df['SHIP_DATE'] = pd.to_datetime(df['SHIP_DATE'])

    # Eliminação de registros inconsistentes e duplicados
    df = df[df['SHIP_DATE'] >= df['ORDER_DATE']]

    # Tratamento de colunas de localização
    df = df.drop_duplicates(subset='ORDER_ID')
    state_map = {
        'ALABAMA': 'AL', 'ALASKA': 'AK', 'ARIZONA': 'AZ', 'ARKANSAS': 'AR',
        'CALIFORNIA': 'CA', 'COLORADO': 'CO', 'CONNECTICUT': 'CT',
        'DELAWARE': 'DE', 'FLORIDA': 'FL', 'GEORGIA': 'GA',
        'HAWAII': 'HI', 'IDAHO': 'ID', 'ILLINOIS': 'IL',
        'INDIANA': 'IN', 'IOWA': 'IA', 'KANSAS': 'KS',
        'KENTUCKY': 'KY', 'LOUISIANA': 'LA', 'MAINE': 'ME',
        'MARYLAND': 'MD', 'MASSACHUSETTS': 'MA', 'MICHIGAN': 'MI',
        'MINNESOTA': 'MN', 'MISSISSIPPI': 'MS', 'MISSOURI': 'MO',
        'MONTANA': 'MT', 'NEBRASKA': 'NE', 'NEVADA': 'NV',
        'NEW HAMPSHIRE': 'NH', 'NEW JERSEY': 'NJ', 'NEW MEXICO': 'NM',
        'NEW YORK': 'NY', 'NORTH CAROLINA': 'NC', 'NORTH DAKOTA': 'ND',
        'OHIO': 'OH', 'OKLAHOMA': 'OK', 'OREGON': 'OR',
        'PENNSYLVANIA': 'PA', 'RHODE ISLAND': 'RI',
        'SOUTH CAROLINA': 'SC', 'SOUTH DAKOTA': 'SD',
        'TENNESSEE': 'TN', 'TEXAS': 'TX', 'UTAH': 'UT',
        'VERMONT': 'VT', 'VIRGINIA': 'VA', 'WASHINGTON': 'WA',
        'WEST VIRGINIA': 'WV', 'WISCONSIN': 'WI', 'WYOMING': 'WY',
        'DISTRICT OF COLUMBIA': 'DC'
    }

    df['STATE'] = df['STATE'].map(state_map)
    df['COUNTRY'] = df['COUNTRY'].replace({'UNITED STATES': 'USA'})

    # Cálculo de vendas
    df['NET_SALES'] = df['SALES'] * df['QUANTITY'] * (1-df['DISCOUNT'])
    df = df.drop(columns=['SALES', 'QUANTITY', 'DISCOUNT'])

    return df


def treat_outliers(df):
    # Criação de coluna de lucro logarítmico para tratamento de outliers
    df['NET_SALES_LOG'] = np.log1p(df['NET_SALES'])

    # Definição de quantis para tratamento de outliers
    q_low = 0.15
    q_high = 0.85

    # Tratamento de outliers da coluna PROFIT e da coluna NET_SALES_LOG
    q_profit_low = df['PROFIT'].quantile(q_low)
    q_profit_high = df['PROFIT'].quantile(q_high)

    q_net_sales_low = df['NET_SALES_LOG'].quantile(q_low)
    q_net_sales_high = df['NET_SALES_LOG'].quantile(q_high)

    df = df[
        (df['PROFIT'].between(q_profit_low, q_profit_high)) &
        (df['NET_SALES_LOG'].between(q_net_sales_low, q_net_sales_high))
    ]

    df = df.reset_index(drop=True)
    df = df.drop(columns=['NET_SALES_LOG'])

    return df

def scale_data(df, cols):

    # Escalonamento de PROFIT e NET_SALES
    scaler = StandardScaler()
    df[[col + '_SCALED' for col in cols]] = scaler.fit_transform(df[cols])
    return df, scaler

def preprocess(df):

    # Pré-processamento para modelagem
    df = clean_data(df)
    df = scale_data(df, ['PROFIT', 'NET_SALES'])[0]
    df = treat_outliers(df)
    return df

def data_clustering(df):
    # Selecão de colunas relevantes para clustering e agregação por cliente
    df_clustering = df[['ORDER_DATE',
                        'SHIP_MODE',
                        'CUSTOMER_ID',
                        'SEGMENT',
                        'CATEGORY',
                        'REGION',
                        'NET_SALES',
                        'PROFIT',
                        'NET_SALES_SCALED',
                        'PROFIT_SCALED']].copy()
    
    # Agrupamento por cliente e agregação de colunas relevantes
    df_clustering = df_clustering.groupby('CUSTOMER_ID', as_index=False).agg({
        'ORDER_DATE': 'first',
        'SHIP_MODE': 'first',
        'CATEGORY': 'first',
        'REGION': 'first',
        'SEGMENT': 'first',
        'NET_SALES': 'sum',
        'PROFIT': 'sum',
        'NET_SALES_SCALED': 'sum',
        'PROFIT_SCALED': 'sum'
    })

    # Preparação dos dados para clustering (one-hot encoding e escalonamento)
    X_scaled = df_clustering.drop(columns=['CUSTOMER_ID', 'NET_SALES', 'PROFIT'])
    X_scaled['ORDER_DATE'] = pd.to_datetime(df_clustering['ORDER_DATE'], errors='coerce').dt.year
    cols_one_hot = ['SHIP_MODE', 'SEGMENT', 'REGION', 'CATEGORY', 'ORDER_DATE']

    X_scaled = pd.get_dummies(
        X_scaled,
        columns=cols_one_hot,
    )

    return X_scaled, df_clustering

def dict_analise_temporal(df_clustering):

    # Preparação dos dados para análise temporal por cluster
    df_clustering['ORDER_DATE'] = pd.to_datetime(df_clustering['ORDER_DATE'])
    
    # Agrupamento por mês e cluster, somando o lucro total
    df_temporal = df_clustering.groupby(
        [df_clustering['ORDER_DATE'].dt.to_period('M'), 'CLUSTER']
    )['PROFIT'].sum().reset_index()

    # Conversão da coluna ORDER_DATE para datetime e ordenação dos dados
    df_temporal['ORDER_DATE'] = df_temporal['ORDER_DATE'].dt.to_timestamp()
    df_temporal = df_temporal.sort_values('ORDER_DATE').reset_index(drop=True)

    # Separação dos dados por cluster
    df_temporal_cluster_0 = df_temporal[df_temporal['CLUSTER'] == 0].copy()
    df_temporal_cluster_1 = df_temporal[df_temporal['CLUSTER'] == 1].copy()
    df_temporal_cluster_2 = df_temporal[df_temporal['CLUSTER'] == 2].copy()

    # Criação de dicionário para armazenar os dados temporais por cluster
    df_temporal_dict = {
        0: df_temporal_cluster_0,
        1: df_temporal_cluster_1,
        2: df_temporal_cluster_2
    }

    # Filtragem dos dados para o período a partir de julho de 2020
    for i in df_temporal_dict:
        df_temporal_dict[i] = df_temporal_dict[i][
            df_temporal_dict[i]['ORDER_DATE'] >= '2020-07-01'
        ]

    # Teste de estacionariedade e diferenciação dos dados para cada cluster
    for i, data in df_temporal_dict.items():
        stationary = test_stationarity(data['PROFIT'])
        df_temporal_dict[i] = df_temporal_dict[i].loc[stationary.index].copy()
        df_temporal_dict[i]['PROFIT'] = stationary

    return df_temporal_dict


def test_stationarity(timeseries, diff_order=0):

    # Teste de estacionariedade usando o teste de Dickey-Fuller 
    result = adfuller(timeseries)
    if result[1] >= 0.05:
        return test_stationarity(timeseries.diff().dropna(), diff_order + 1)

    return timeseries