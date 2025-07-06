import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.project_utils import setup_project_path
setup_project_path()
import pandas as pd
from src.utils import data_cleaning

def test_clean_quantity():
    df = pd.DataFrame({'Quantity': ['10', '5a', None, '']})
    cleaned = data_cleaning.clean_quantity(df.copy())
    assert all(isinstance(x, int) for x in cleaned['Quantity'])
    assert cleaned['Quantity'].tolist() == [10, 5, 0, 0]

def test_clean_unit_price():
    df = pd.DataFrame({'UnitPrice': ['12.5', '$10.0', 'abc', None]})
    cleaned = data_cleaning.clean_unit_price(df.copy())
    assert all(isinstance(x, float) for x in cleaned['UnitPrice'])
    assert cleaned['UnitPrice'].tolist() == [12.5, 10.0, 0.0, 0.0]

def test_clean_stock_code():
    df = pd.DataFrame({'StockCode': ['A 123', 'B@456', 'C-789', 'D_000']})
    cleaned = data_cleaning.clean_stock_code(df.copy())
    assert cleaned['StockCode'].tolist() == ['A123', 'B456', 'C789', 'D000']

def test_clean_invoice_customer():
    df = pd.DataFrame({'InvoiceNo': ['INV123', '456'], 'CustomerID': ['C789', '012']})
    cleaned = data_cleaning.clean_invoice_customer(df.copy())
    assert cleaned['InvoiceNo'].tolist() == ['123', '456']
    assert cleaned['CustomerID'].tolist() == ['789', '012']

def test_clean_country_description():
    df = pd.DataFrame({'Country': ['U$A', 'UKXxY', None], 'Description': ['desc$', '$item', None]})
    cleaned = data_cleaning.clean_country_description(df.fillna(''))
    assert cleaned['Country'].tolist() == ['UA', 'UK', '']
    assert cleaned['Description'].tolist() == ['desc', 'item', '']

def test_handle_missing_values():
    df = pd.DataFrame({
        'StockCode': ['A', None],
        'Description': ['desc', None],
        'UnitPrice': [1.0, None],
        'Quantity': [None, 2],
        'InvoiceDate': [None, '2020-01-01 00:00:00'],
        'Country': [None, 'UK']
    })
    cleaned = data_cleaning.handle_missing_values(df.copy())
    assert cleaned.shape[0] == 1  # Drops row with missing essential fields
    assert cleaned.iloc[0]['Quantity'] == 1
    assert cleaned.iloc[0]['InvoiceDate'] == '1970-01-01 00:00:00' or cleaned.iloc[0]['InvoiceDate'] == '2020-01-01 00:00:00'
    assert cleaned.iloc[0]['Country'] in ['Unknown', 'UK']

def test_clean_dataset():
    df = pd.DataFrame({
        'StockCode': ['A', 'B'],
        'Description': ['desc$', 'item'],
        'UnitPrice': ['10', 'abc'],
        'Quantity': ['5a', None],
        'InvoiceNo': ['INV1', '2'],
        'CustomerID': ['C1', '2'],
        'InvoiceDate': [None, '2020-01-01 00:00:00'],
        'Country': ['U$A', None]
    })
    cleaned = data_cleaning.clean_dataset(df.copy())
    assert 'desc' in cleaned['Description'].tolist()[0]
    assert all(isinstance(x, int) for x in cleaned['Quantity'])
    assert all(isinstance(x, float) for x in cleaned['UnitPrice']) 