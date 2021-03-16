import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import datetime as dt
from collections import Counter    
import re
from datetime import datetime


def result_inference(x):
    result=x['Result']
    if r'(' in result:
        index=result.index(r'(')
        interval=result[index:]
        result=result[:index]
    if '--' in result:
        return 'Very low'
    elif '-' in result:
        return 'Low'
    elif '++' in result:
        return 'Very high'
    elif '+' in result:
        return 'High'
    elif 'negativ' in result.lower():
        return 'Negative'
    elif 'positiv' in result.lower():
        return 'Positive'
    return 'Normal'

def interpret_tests(data):
    Y=data[data['TYP']=='Y'].dropna(subset=["TEXT"])
    Y=Y.drop_duplicates()

    # Splitting rows with tests seperated by ';' eg HKT=43.0 %; MCV=87.4 fl; MCH=27.6 pg
    Y['text'] = Y['TEXT']
    Y=Y.set_index(['PATIENT_HASH', 'ZENTRUM_ID','PAT_GESCHLECHT', 'PATIENT_ID','DATUM', 'TYP', 'text','ICD10',
           'SICHERHEIT'])
    Y=Y['TEXT'].str.split(';').explode().reset_index()
    Y['TEXT']=Y['TEXT'].str.replace(r'^\s+',r'')

    # Processing lab results with format LEUKO=4.5
    Y_1=Y[Y.TEXT.dropna().str.contains(r'[a-zA-Z0-9\s]+=[a-zA-Z0-9\s]+')]
    Y_1[['Lab_test','Result']]=Y_1['TEXT'].str.split('=',expand=True,n=1)
    Y_1['Inference']=Y_1.apply(result_inference,axis=1)
    Y_1['Inference'].value_counts() 
    Y_new=Y_1

    # Find tests that appear in the dataframe atleast 10 times
    counts=Y_new['Lab_test'].value_counts().reset_index()
    tests=list(set(counts[counts['Lab_test']>1]['index']))
    Y_new=Y_new[Y_new['Lab_test'].isin(tests)]

    # Group by Patient and Date and create two dictionaries of Lab_test:Inference, Lab_test:Result
    columns=['PATIENT_HASH', 'ZENTRUM_ID','PATIENT_ID','PAT_GESCHLECHT', 'DATUM','TYP','text']
    Y_new=Y_new.set_index('Lab_test').groupby(columns)[['Inference','Result']].apply(lambda x: x.to_dict()).reset_index(name='Lab_results')
    Y_new['Inference_dict']=Y_new['Lab_results'].apply(lambda x: x['Inference'])
    Y_new['Result_dict']=Y_new['Lab_results'].apply(lambda x: x['Result'])
    Y_new=Y_new.drop('Lab_results',axis=1)

    # Convert Inference dictionary to multiple columns, one for each test
    results_df=Y_new["Inference_dict"].apply(pd.Series)
    Y_new=pd.concat([Y_new,results_df],axis=1)

    Y_new = Y_new.drop(columns=['PAT_GESCHLECHT','Result_dict','Inference_dict']).rename(columns={"text":"TEXT"})

    return Y_new
    
    