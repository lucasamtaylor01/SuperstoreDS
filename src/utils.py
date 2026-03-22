import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def clean_data(df):
    # PADRONIZAÇÃO DE CONTEÚDO STRING

    df.columns = df.columns.str.upper()
    df.columns = df.columns.str.replace(' ', '_')
    string_cols = df.select_dtypes(include=['object', 'string']).columns
    df[string_cols] = df[string_cols].apply(lambda x: x.astype(str).str.strip().str.upper())

    df = df.drop_duplicates(subset='ORDER_ID')

    # Eliminação de colunas desnecessárias
    df = df.drop(columns=['CUSTOMER_NAME', 'POSTAL_CODE', 'ORDER_ID'])

    # Padronização de dados temporais
    df['ORDER_DATE'] = pd.to_datetime(df['ORDER_DATE'])
    df['SHIP_DATE'] = pd.to_datetime(df['SHIP_DATE'])
    df = df[df['SHIP_DATE'] >= df['ORDER_DATE']]

    # Padronização de dados geográficos
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
    df['NET_SALES'] = df['SALES'] * df['QUANTITY'] * (1-df['DISCOUNT'])

    df = df.drop(columns=['SALES', 'QUANTITY', 'DISCOUNT'])
    df = df.rename(columns={'SUB-CATEGORY': 'SUB_CATEGORY'})

    return df

def feature_engineering(df):
    df['NET_SALES_LOG'] = np.log1p(df['NET_SALES'])

    return df


def treat_outliers(df):
    q_low = 0.15
    q_high = 0.85

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
    scaler = StandardScaler()
    df[[col + '_SCALED' for col in cols]] = scaler.fit_transform(df[cols])
    return df, scaler


def preprocess(df):
    df = clean_data(df)
    df = feature_engineering(df)
    df = scale_data(df, ['PROFIT', 'NET_SALES'])[0]
    df = treat_outliers(df)
    return df

def data_clustering(df):
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
    df_clustering = df_clustering.reset_index(drop=True)
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

    X_scaled = df_clustering.drop(columns=['CUSTOMER_ID', 'NET_SALES', 'PROFIT'])
    X_scaled['ORDER_DATE'] = pd.to_datetime(df_clustering['ORDER_DATE'], errors='coerce').dt.year
    cols_one_hot = ['SHIP_MODE', 'SEGMENT', 'REGION', 'CATEGORY', 'ORDER_DATE']

    X_scaled = pd.get_dummies(
        X_scaled,
        columns=cols_one_hot,
    )

    return X_scaled, df_clustering


def data_prediction(df_clustering):
    df_predict = df_clustering[['ORDER_DATE', 'SHIP_MODE', 'PROFIT_SCALED', 'NET_SALES_SCALED', 'REGION', 'CLUSTER']].copy()
    df_predict = pd.get_dummies(
        df_predict,
        columns=['REGION'],
    )

    state_map = {
        'STANDARD CLASS': '0',
        'SECOND CLASS': '1',
        'FIRST CLASS': '2',
        'SAME DAY': '3'
    }

    df_predict['SHIP_MODE'] = df_predict['SHIP_MODE'].map(state_map)
    df_predict['SHIP_MODE'] = df_predict['SHIP_MODE'].astype('Int64')

    df_predict['ORDER_DATE'] = pd.to_datetime(df_predict['ORDER_DATE'], errors='coerce')
    df_predict['ORDER_YEAR'] = df_predict['ORDER_DATE'].dt.year

    df_predict = df_predict.drop(columns=['ORDER_DATE'])

    df_predict = {
        cluster: df_predict[df_predict['CLUSTER'] == cluster]
        for cluster in df_predict['CLUSTER'].unique()
    }
    
    return df_predict