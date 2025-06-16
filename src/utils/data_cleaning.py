import pandas as pd
import re

def clean_quantity(df):
    """Standardize Quantity: remove non-numeric, convert to int"""
    df['Quantity'] = df['Quantity'].astype(str).str.replace(r'\D', '', regex=True)
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0).astype(int)
    return df

def clean_unit_price(df):
    """Standardize UnitPrice: remove non-numeric, convert to float"""
    df['UnitPrice'] = df['UnitPrice'].astype(str).str.replace(r'[^\d\.]', '', regex=True)
    df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce').fillna(0.0).astype(float)
    return df

def clean_stock_code(df):
    """Standardize StockCode: remove special characters and whitespace"""
    df['StockCode'] = (df['StockCode']
        .astype(str)
        .str.encode('ascii', 'ignore')
        .str.decode('ascii')
        .str.replace(r'[^a-zA-Z0-9]', '', regex=True)
        .str.strip()
    )
    return df

def clean_invoice_customer(df):
    """Standardize InvoiceNo and CustomerID: remove non-numeric, convert to string"""
    df['InvoiceNo'] = df['InvoiceNo'].astype(str).str.replace(r'\D', '', regex=True)
    df['CustomerID'] = df['CustomerID'].astype(str).str.replace(r'\D', '', regex=True)
    return df

def clean_country_description(df):
    """Standardize Country and Description"""
    df['Country'] = df['Country'].astype(str).str.replace(r'[^\w\s]', '', regex=True).str.replace(r'XxY', '', regex=True).str.strip()
    df['Description'] = df['Description'].astype(str).str.replace(r'[$]', '', regex=True).str.strip()
    return df

def handle_missing_values(df):
    """Handle missing values in the dataset"""
    essential_cols = ['StockCode', 'Description', 'UnitPrice']
    df = df.dropna(subset=essential_cols)
    
    # Fill missing values in specific columns with default values
    df['Quantity'] = df['Quantity'].fillna(1)
    df['InvoiceDate'] = df['InvoiceDate'].fillna('1970-01-01 00:00:00')
    df['Country'] = df['Country'].fillna('Unknown')
    return df

def clean_dataset(df):
    """Run all cleaning functions on the dataset"""
    df = handle_missing_values(df)
    df = clean_quantity(df)
    df = clean_unit_price(df)
    df = clean_stock_code(df)
    df = clean_invoice_customer(df)
    df = clean_country_description(df)
    return df 