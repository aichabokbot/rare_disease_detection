import pandas as pd
import numpy as np
import random


def extract_patients(center):
    """
    Extract the data of patients who are suspected to have or are diagnosed of 
    one of the four diseases of interest.
    Our method to find these patients is to keep only the rows where the disease 
    is mentioned, either by its ICD code or by its name.

    Parameters
    ------------
    center : dataframe
        raw data of each medical center (center, Frankfurt, Hamburg, Stuttgart)

    Returns
    -------
    dataframe
    """
    
    # Gaucher Disease
    center["is_E75.22"] = (center['TEXT'].apply(lambda x: 'E75.22' in str(x))*1 + 
                           center['ICD10'].apply(lambda x: 'E75.22' in str(x))*1 + 
                           center['TEXT'].apply(lambda x: 'Gaucher' in str(x))*1 +
                           center['TEXT'].apply(lambda x: 'gaucher' in str(x))*1)
    
    # Familial Hypercholesterolemia
    center["is_E78.01"] = (center['TEXT'].apply(lambda x: 
                                                any(t in str(x) for t in ['amiliäre Hyperchol','amiliäre hyperchol',
                                                                          'amiliären Hyperchol','amiliären hyperchol']))*1)
    
    # Familial Chylomicronemia 
    center["is_E78.3"] = (center['TEXT'].apply(lambda x: 'E78.3' in str(x))*1 + 
                          center['ICD10'].apply(lambda x: 'E78.3' in str(x))*1 +
                          center['TEXT'].apply(lambda x: 'hylomikronämie' in str(x))*1)
    
    # Betaoxidation Disorder
    center["is_E71.3"] = (center['TEXT'].apply(lambda x: 'E71.3' in str(x)) *1 + 
                          center['ICD10'].apply(lambda x: 'E71.3' in str(x)) *1 +
                          center['TEXT'].apply(lambda x: 'Betaoxidationsdefekt' in str(x))*1)

    
    center['is_disease'] = center['is_E75.22'] + center['is_E78.01'] + center['is_E78.3'] + center['is_E71.3']    
    patients_of_interest = list(center[center['is_disease']!=0]['PATIENT_HASH'].unique())
    other_patiens = random.sample(list(center[center['is_disease']==0]['PATIENT_HASH'].unique()),len(patients_of_interest))
    center_extract = center[center.PATIENT_HASH.isin(patients_of_interest + other_patiens)]
    
    return center_extract
    
    
def main():
    Berlin = extract_patients(pd.read_csv('../../data/raw/BER01_MD_HEC.csv.gz', sep=';',encoding="latin1"))
    print("Berlin extract done")
    Frankfurt = extract_patients(pd.read_csv('../../data/raw/FRA01_MD_HEC_V2.csv.gz', sep=';',encoding="latin1"))
    print("Frankfurt extract done")
    Hamburg = extract_patients(pd.read_csv('../../data/raw/HAM08_MD_HEC.csv.gz', sep=';',encoding="latin1"))
    print("Hamburg extract done")
    Stuttgart = extract_patients(pd.read_csv('../../data/raw/STR01_MD_HEC.csv.gz', sep=';',encoding="latin1"))
    print("Stuttgart extract done")
   
    centers = [Berlin, Frankfurt, Hamburg, Stuttgart]
    pd.concat(centers).to_csv("../../data/interim/data_extract.csv", index=False)
    print('data extract done')
    

if __name__ == "__main__":
    main()
   