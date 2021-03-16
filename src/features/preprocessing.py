import pandas as pd
import numpy as np
import re

class preprocessing:

    def DATUM_preprocessing(self,data):
        data['entry_year'] = data.DATUM.apply(lambda x: str(x)[6:])
        
        # dealing with typos in the Berlin dataset
        if data.iloc[0,1] == 'BER01':
            data['entry_year'] = data['entry_year'].replace('50','05').replace('90','09')
            
        def extract_year(year):
            if year<'21':
                    return '20'+year
            else:
                return '19'+year

        data['entry_year'] = data['entry_year'].apply(lambda x: extract_year(x))
        _ = data['DATUM'].apply(lambda x: x[:6]) + data['entry_year']
        data['entry_date'] = pd.to_datetime(_, format='%d.%m.%Y')
        
        return data
    
    def PAT_GEBDATUM_preprocessing(self,data):
        data = self.DATUM_preprocessing(data)
        data['birth_year'] = data.PAT_GEBDATUM.apply(lambda x: str(x)[6:])
        data['birth_entry'] = data['entry_year'].apply(lambda x: str(x)[2:]).astype(int) - data['birth_year'].astype(int)
        
        def extract_year(row):
            if row.birth_year > '20':
                return '19' + row.birth_year
            else:
                if row.entry_year < '2000':
                    return '19' + row.birth_year
                else :
                    if row.birth_entry < 0: 
                        return '19' + row.birth_year
                    else:
                        return '20' + row.birth_year
             
                                      
        data.loc[:, 'birth_year'] = data.apply(extract_year, axis = 1)  
        
        _ = data['PAT_GEBDATUM'].apply(lambda x: x[:6]) + data['birth_year']
        data['birth_date'] = pd.to_datetime(_, format='%d.%m.%Y')
                
        data = data.drop(['birth_year', 'entry_year', 'birth_entry'], axis=1)
        
        return data
    
    def age(self,data):
        data = self.PAT_GEBDATUM_preprocessing(data)
        _ = data['entry_date'] - data['birth_date']
        data['age'] = (_ / np.timedelta64(1, 'Y')).round(1)
        
        return data
    
    def ICD(self,row): 
        if (pd.notnull(row['ICD10'])) & (str(row.ICD10)[-1] == '-'):
            return str(row.ICD10)[:-1]
        if (row['TYP'] in ['*', 'D']) & (pd.isnull(row['ICD10'])):
            if row['is_E75.22']>0:
                return 'E75.22'
            elif row['is_E78.01']>0:
                return 'E78.01'
            elif row['is_E78.3']>0:
                return 'E78.3'
            elif row['is_E71.3']>0:
                return 'E71.3'
            elif re.search(r'[A-Za-z][0-9]{2}\.[0-9]+[A-Za-z]',str(row.TEXT)) != None:
                return re.search(r'[A-Za-z][0-9]{2}\.[0-9]+[A-Za-z]',str(row.TEXT))[0][:-1]
            else:
                return row['ICD10']
        elif (row['TYP'] in ['*', 'D']) & (row['ICD10']=='E75.2') & (row['is_E75.22']>0):
            return 'E75.22'

        else:
            return row['ICD10']

    def SICHERHEIT(self, row):
        if (row['TYP'] in ['*', 'D']) & (('usschlu') in str(row.TEXT)):
            return 'A'
        if (row['TYP'] in ['*', 'D']) & (('V.a.') in str(row.TEXT)):
            return 'V'
        if (row['TYP'] in ['*', 'D']) & (('Z.a.') in str(row.TEXT)):
            return 'Z'
        if (row['TYP'] in ['*', 'D']) & (pd.notnull(row.ICD10)) & (pd.isnull(row.SICHERHEIT)):
            return 'G'    
        if (pd.isnull(row.SICHERHEIT)) & ((str(row.ICD10)+'G') in str(row.TEXT)):
            return 'G'
        if (pd.isnull(row.SICHERHEIT)) & ((str(row.ICD10)+'A') in str(row.TEXT)):
            return 'A'
        if (pd.isnull(row.SICHERHEIT)) & ((str(row.ICD10)+'V') in str(row.TEXT)):
            return 'V'
        if (pd.isnull(row.SICHERHEIT)) & ((str(row.ICD10)+'Z') in str(row.TEXT)):
            return 'Z'
        else:
            return row.SICHERHEIT
        
    
    def diagnosis_preprocessing(self,data):
        data = self.age(data)
        data['ICD10'] = data.apply(self.ICD, axis=1)
        data['SICHERHEIT'] = data.apply(self.SICHERHEIT, axis=1)
        
        # drop diagnosis duplicates
        data_diagnosis = data[(data.TYP =='*') + (data.TYP =='D')].sort_values(
            by=['PATIENT_HASH','entry_date']).drop_duplicates(subset=['PATIENT_HASH','ZENTRUM_ID','TYP','ICD10','SICHERHEIT'])
        data = pd.concat([data[(data.TYP =='A') + (data.TYP =='Y')], data_diagnosis]).sort_values(by=['PATIENT_HASH','entry_date'])
        
        # drop uncessary columns
        data = data.drop(columns=['PAT_GEBDATUM','TYP_EXT','is_E75.22', 'is_E78.01', 'is_E78.3', 'is_E71.3',
                                 'is_disease','birth_date','entry_date'])
        
        # remove reimbursement data
        data = data[data.TEXT.apply(lambda x: 'AL:' not in str(x))]
        data = data[data.TEXT.apply(lambda x: '|END' not in str(x))]
        
        # create diagnosis feature        
        for icd in ['E78.01','E78.3','E75.22','E71.3']:
            excluded_patients = data[(data.ICD10 == icd) & (data.SICHERHEIT == 'A')].PATIENT_HASH.unique()
            confirmed_patients = data[(data.ICD10 == icd) & (data.TYP.isin(["*"])) & (data.SICHERHEIT.isin(["G","Z"])) ].PATIENT_HASH.unique()
            confirmed_patients = set(confirmed_patients) - set(excluded_patients)
            suspicious_patients = set(data[data.ICD10 == icd].PATIENT_HASH.unique()) - set(confirmed_patients) - set(excluded_patients)
            
            def diagnosis(row):
                if row.PATIENT_HASH in excluded_patients:
                    return "excluded"
                if row.PATIENT_HASH in confirmed_patients:
                    return "confirmed"
                if row.PATIENT_HASH in suspicious_patients:
                    return "suspicious"
            
            data[f'diagnosis_{icd}'] = data.apply(diagnosis, axis=1)
                    
        return data
    
    
def main():
    data = pd.read_csv('/home/jovyan/amedes_challenge/data/interim/data_extract.csv')
    preprocessed_data = preprocessing().diagnosis_preprocessing(data)
    preprocessed_data.to_csv('/home/jovyan/amedes_challenge/data/interim/data_extract_preprocessed.csv', index=False)
    
if __name__ == "__main__":
    main()
   