import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def clean_data(df):
    # PADRONIZAÇÃO DE CONTEÚDO STRING

    df.columns = df.columns.str.upper()
    df.columns = df.columns.str.replace(" ", "_")
    string_cols = df.select_dtypes(include=['string']).columns
    df[string_cols] = df[string_cols].apply(lambda x: x.str.upper())

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
    df['COUNTRY'] = df['COUNTRY'].map({'UNITED STATES': 'USA'})
    df['NET_SALES'] = df['SALES'] * df['QUANTITY'] * (1-df['DISCOUNT'])

    df = df.drop(columns=['SALES', 'QUANTITY', 'DISCOUNT'])
    df = df.rename(columns={"SUB-CATEGORY": "SUB_CATEGORY"})

    return df

def feature_engineering(df):
    df['NET_SALES_LOG'] = np.log1p(df['NET_SALES'])

    freq = df['STATE'].value_counts(normalize=True)
    df['STATE_FREQ'] = df['STATE'].map(freq)

    return df

def treat_outliers(df):
    q_low = 0.025
    q_high = 0.975

    q1 = df['PROFIT'].quantile(q_low)
    q99 = df['PROFIT'].quantile(q_high)

    df['PROFIT_CLIPPED'] = df['PROFIT'].clip(q1, q99)

    return df

def scale_data(df, cols):
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df, scaler

def preprocess(df):
    df = clean_data(df)
    df = feature_engineering(df)
    df = treat_outliers(df)
    return df