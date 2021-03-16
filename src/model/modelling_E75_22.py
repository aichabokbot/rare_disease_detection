import pandas as pd
import numpy as np
import datetime as dt
from collections import Counter    
import re
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import random
import xgboost as xgb
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from features.interpret_tests import interpret_tests



def modelling(data, k):
    patients = data[data[f"diagnosis_E75.22"].isin(["confirmed","suspicious"])].PATIENT_HASH.unique()

    subset_patients = np.concatenate((np.array(random.choices(data[~data.PATIENT_HASH.isin(patients)].PATIENT_HASH.unique(), k=k)),
                                      patients))
    subset = data[data.PATIENT_HASH.isin(subset_patients)].reset_index(drop=True)

    Y_new = interpret_tests(subset)

    subset = subset.merge(Y_new, how='left', on=['PATIENT_HASH','ZENTRUM_ID','DATUM','TYP','TEXT'])
    subset = subset.dropna(axis=1, how='all')
    
    # relevant tests
    x = subset[['PATIENT_HASH','TYP','ERY', 'HB', 'HKT', 'MCH', 'THRO', 'AP', 'GGT', 'FERR', 'NEUT',
           'FKAP', 'FKALAQ', 'CRP', 'INSU', 'OSTE', 'VD25', 'TFS', 'ALBUMA',
           'A1GLOA', 'A2GLOA', 'DPD']]
    x =  x[x.TYP == "Y"].fillna("") \
        .groupby('PATIENT_HASH')[np.array(x.columns[2:])].sum() \
        .dropna(how='all', axis=0).reset_index()

    def value(x):
        if x=="":
            return "no_test"
        else:
            return ['Low','High','Very low','Very high','Normal'][
                np.argmax([x.count('Low'),x.count('High'),x.count('Very low'),x.count('Very high'),x.count('Normal')])]

    test = pd.concat([x.PATIENT_HASH,x.drop('PATIENT_HASH', axis=1).applymap(value)], axis=1)

    from sklearn.preprocessing import OneHotEncoder

    enc = OneHotEncoder()
    enc.fit_transform(test.drop('PATIENT_HASH',axis=1))
    relevantTests = ['ERY', 'HB', 'HKT', 'MCH', 'THRO', 'AP', 'GGT', 'FERR', 'NEUT','FKAP', 'FKALAQ', 'CRP', 
                     'INSU', 'OSTE', 'VD25', 'TFS', 'ALBUMA','A1GLOA', 'A2GLOA', 'DPD']
    tests = pd.concat([test.PATIENT_HASH, pd.DataFrame(data = enc.fit_transform(test.drop('PATIENT_HASH',axis=1)).toarray(),
                columns = enc.get_feature_names(relevantTests))], axis = 1)

    # co-morbidity
    def ICD(row): 
        if (row['TYP'] in ['*','D']) & (pd.isnull(row['ICD10'])):
            if ('Hyperton' in row.TEXT) & ('art' in row.TEXT):
                return 'I10.90'
            if 'Hepatitis C' in row.TEXT:
                return 'B18.2'
            if 'Hypothyreose nach medizinischen Maßnahmen' in row.TEXT:
                return 'E89.0'
            if 'Sterilität beim Mann' in row.TEXT:
                return 'N46'
            if '3-Gefäß-KHK' in row.TEXT:
                return 'I25.13'
            if ('Vit' in row.TEXT) & ('D' in row.TEXT) & ('Mangel' in row.TEXT):
                return 'E55.9'
            if 'Anämie' in row.TEXT:
                return 'D64.9'
            if ('fatigue' in row.TEXT) & ('yndrom' in row.TEXT):
                return 'G93.3'
            if 'Obstruktive Bronchitis' in row.TEXT:
                return 'J44.89'
            else:
                return row['ICD10']
        else:
            return row['ICD10']
    subset['ICD10'] = subset.apply(ICD, axis=1)
    codes = ['E55.9','I10.90','D69.61','G93.3','R16.1']

    coMorbidity = pd.DataFrame(tests.PATIENT_HASH)
    for code in codes:
        coMorbidity[code] = pd.merge(pd.DataFrame(tests.PATIENT_HASH),
                                     subset[subset.ICD10 == code].groupby('PATIENT_HASH')['ICD10'].nunique().reset_index(),
                                     how="left", on ="PATIENT_HASH").fillna(0).drop("PATIENT_HASH", axis=1)

    # symptoms
    words = [["Fatigue", "müde","Fatique","Erschöpfung", "fatigue"], #fatigue
                ["Knochenschmerzen","Knochenstoffwechselstörung"], # bone pain
                ["Splenomegalie","Milzläsionen", "splenomegalie", "Splenektomie", "Hepatosplenomegalie"], # spenomagalie
                ["Thrombopenie","Thrombozytopenie"], ["Chololithiasis"],["Chitotriosidase"],
                ["Anämie"],["Leukopenie"],["Panzytopenie"],["Niereninsuffizienz"],["Nephrolithiasis"]]

    symptoms = pd.DataFrame(data = subset.PATIENT_HASH.unique(), columns = ['PATIENT_HASH'])

    for word in words:
        symptoms[word[0]] = pd.merge(pd.DataFrame(data = subset.PATIENT_HASH.unique(), columns = ['PATIENT_HASH']),
                                     subset[subset.TEXT.apply(lambda x: any(t in x for t in word))].groupby('PATIENT_HASH')['PATIENT_ID'].count().reset_index(),
                                     how="left", on ="PATIENT_HASH").fillna(0).drop("PATIENT_HASH", axis=1)

    # age
    age = subset.groupby('PATIENT_HASH')['age'].mean().reset_index()

    dataset = subset[['PATIENT_HASH','PAT_GESCHLECHT']].drop_duplicates().replace("W",1).replace("M",0).reset_index(drop=True)
    dataset['gaucher'] = dataset.PATIENT_HASH.isin(patients)*1
    dataset = pd.merge(dataset, age , how="left", on='PATIENT_HASH')
    dataset = pd.merge(dataset, tests , how="left", on='PATIENT_HASH')
    dataset = pd.merge(dataset, coMorbidity , how="left", on='PATIENT_HASH')
    dataset = pd.merge(dataset, symptoms , how="left", on='PATIENT_HASH')
    dataset = dataset.fillna(0).drop('PATIENT_HASH', axis=1)
    
    X = dataset.drop('gaucher', axis=1)
    y = dataset.gaucher

    names = ["Nearest Neighbors", "Logistic Regression", 
             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
             "Naive Bayes"]

    classifiers = [
        KNeighborsClassifier(3),
        LogisticRegression(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        MLPClassifier(),
        AdaBoostClassifier(),
        GaussianNB()]

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        scores = cross_val_score(clf, X, y, cv=10)
        y_pred = cross_val_predict(clf, X, y, cv=10)
        CM = confusion_matrix(y, y_pred)
        TN = CM[0][0] 
        FN = CM[1][0]
        TP = CM[1][1]
        FP = CM[0][1]
        print(name,np.round(np.mean(scores),3))
        print("      Positives: ", 100*round(FN/(TP+FN),2), "% misclassifed     ", FN, '/',TP+FN)
        print("      Negatives: ", 100*round(FP/(TN+FP),2), "% misclassifed     ",FP, '/',TN+FP)
        
    return X,y
                    
def main():
    data = pd.read_csv('/home/jovyan/amedes_challenge/data/interim/data_extract_preprocessed.csv')                    
    modelling(data, 1800)
    
if __name__ == "__main__":
    main()
   