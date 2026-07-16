import numpy as np
import pandas as pd
# from IPython.core.page import page
import tabulate as tb
from rapidfuzz import process, fuzz, utils
from glob import glob
import calendar
import os

def clean_status(val, outputs):
    if val in outputs:
        return val
    match = process.extractOne(val, outputs, scorer=fuzz.WRatio, score_cutoff=80, processor=utils.default_process)
    if match:
        return match[0]
    else:
        return f"Clarification Needed: {val}"

categorical = {
        'Status': [[['passed', 'pass', 'approved'], ['fail', 'failed', 'rejected'], ['pend', 'pended'], ['conditionally approved', 'conditional', 'conditionallyapproved']], ['Approved', 'Rejected', 'Pended', 'Conditionally Approved']],
        'Brand': [[], ['509', 'Klim', 'Polaris', 'Arctic Cat']],
        'Country': [[], ['Vietnam', 'Bangladash', 'China', 'Korea', 'El Salvador', 'Mexico', 'Jordan', 'Taiwan', 'Indonesia', 'India']],
        'Inspection Type': [[], ["2nd Final", "2nd Inline", "3rd Final", "3rd Inline", "FinalPPMInline", "Pre-final"]],
        'Factory': [[], ["ASG", "Anhui", "Arctic Cat", "Asmara", "Asmara(Pinnacle )", "Caliber", "Captain", "Citimex", "DBS", "DSK, Solo( Asmara)", "Dhakarea Ltd", "Dogree", "Dongguan Turbo", "ETP", "Eastman", "FAC", "Flexfit", "Gia Loc", "GMP", "Grand Union", "Great King", "HCH", "Haotex", "Hyunjin", "Hyunjin-Gia Loc", "Hyunjin-Nam Sach", "Independent Trading", "J-Long", "JM Tech, Asmara", "KKAWISTARA,Solo  (Asmara )", "Kadena", "Kido", "Kido-Ha Noi", "Kido-Jaja", "Kido-Mulia", "Kido-Vinh", "Kin Hing", "King Hamm", "Klim", "Kwong Lung", "Lazer", "Luckywool", "M&S", "M5 Nam Dinh", "Maxport", "Maxport 88", "Maxport M5", "Nam Sach", "Needlecraft", "Ningbo", "Ningbo Chisage", "Ningbo Zhongtian", "Ogk", "POIS", "PS Vina", "PS Vina - Sub", "Pinnacle, Semarang (Asmara)", "Polaris", "Poogshin - PS vina", "Poongshin", "Private Label", "PT Jech Tech", "PT SOLO KAWISTARA", "PT. JM TECH , Asmara", "Quang Anh Sub", "Scikio", "Shin", "Shin BVT", "Shin ETP", "Shin TM", "Shin TN", "Shinhwa", "Shinhwoo", "Signal", "Sportz Pro", "Srinivasa", "Strategic Sports", "Sun free", "Takashima", "Terrific", "TopkeyLazer", "Treksta", "U&K", "Vast", "Venture", "Victor Scout", "Vista", "Waiwah", "Winning", "X-20", "Yodu", "YoungJin", "Youngtech", "ZKG"]],
        'Inspector': [[], ["In-house", "Jason", "Mahmud", "Rosy", "Tony", "Van"]],
        'None': [[['NAN', 'NaN', 'Nan', ' ', 'nan', None, np.nan]], ['']]
        }

def categorical_check(df, col):
    df[col] = df[col].astype(str).str.strip().str.lower()
    conditions = [df[col].isin(x) for x in categorical[col][0] + categorical['None'][0]]
    df[col] = np.select(conditions, categorical[col][1] + categorical['None'][1], default=df[col]) if len(categorical[col][0]) > 0 else np.select(conditions, categorical['None'][1], default=df[col])
    df[col] = df[col].apply(clean_status, outputs=categorical[col][1] + categorical['None'][1])
    return df[col]

lists = glob(f'{os.getcwd()}/data/individual/*.csv')
print(lists)
inspectors = []
dfs = {}
for i in lists:
    name = i.split('\\')[-1].split('.')[0] 
    inspectors.append(name)
    data = pd.read_csv(i)
    data.columns = data.columns.str.lower().str.strip()
    ## For Old Data
    data.rename(columns={'date': 'Inspection Date', 'country': 'Country', 'factory': 'Factory', 'style #': 'Style', 'style description': 'Style Name', 'inspection type': 'Inspection Type', 'po': 'PO', 'order qty': 'Order Quantity', 'status': 'Status', 'exit factory date': 'Exit Date', 'brand': 'Brand', 'date report emailed to factory': 'Report Emailed Date', 'comments': 'Comments', 'qty of inspection date changes': 'unknown' }, inplace=True)
    data.drop(columns=data.columns[data.columns.str.contains('unknown')], inplace=True)
    data['Inspector'] = f'{name.upper()[0]}{name.lower()[1:]}'
    data['Style'] = 'Placeholder'
    data['Time In'] = '10:00'
    data['Time Out'] = '17:00'
    data['Season'] = None
    data['Inspected Quantity'] = None
    ## Actual Pipeline
    data['Order Quantity'] = pd.to_numeric(data['Order Quantity'].astype(str).str.replace(r'\.0$', '', regex=True).str.replace(r'\D+', '', regex=True), errors='coerce').fillna(0).astype(int)
    data['Inspection Date'] = data['Inspection Date'].ffill()
    data['Style Name'] = data['Style Name'].str.lower().str.strip()
    data['Scheduled'] = np.where((data['Country'].notnull()) & (data['Factory'].notnull()) & (data['Style'].notnull()), True, False)
    data['Inspected Quantity'] = data['Inspected Quantity'].fillna(data['Order Quantity'] * 0.1).astype(int)
    data['Report Emailed Date'] = (pd.to_datetime(data['Inspection Date'], errors='coerce') + pd.to_timedelta(2, unit='D')).dt.strftime('%m %b %Y')
    data['Exit Date'] = pd.to_datetime(data['Report Emailed Date'], errors='coerce').dt.strftime('%m %b %Y')
    for j in categorical.keys():
        if j != 'None':
            data[j] = categorical_check(data, col=j)
    for k in data.columns.difference(['Inspection Date', 'Inspector', 'Scheduled']):
        data.loc[data['Scheduled'] == False, k] = np.nan
    print(data.loc[(data['Inspection Date'].str.split(',').str[0] != 'Sunday') & ~((data.duplicated(subset=['Inspection Date'], keep=False)) & (data[data.columns.difference(['Inspection Date'])].isnull().all(axis=1))), :])
    dfs[name] = data.loc[(data['Inspection Date'].str.split(',').str[0] != 'Sunday') & ~((data.duplicated(subset=['Inspection Date'], keep=False)) & (data[data.columns.difference(['Inspection Date'])].isnull().all(axis=1))), :]
    print(name)
    print(data.columns)
print(dfs)
# page(tb.tabulate(dfs['tony'], dfs['tony'].columns, tablefmt='grid'))

big_df = None
for i, df in enumerate(dfs.keys()):
    if i == 0:
        big_df = dfs[df]
    else:
        big_df = pd.concat([big_df, dfs[df]], ignore_index=True)