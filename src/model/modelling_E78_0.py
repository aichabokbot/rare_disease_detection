import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import re
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
plt.style.use('ggplot')
from scipy.spatial.distance import cdist
nltk.download('punkt')
nltk.download('stopwords')

##lab results:
def result_inference(x):
    """
    Function to encode lab results
    """
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

def lab_processing(data):
    """
    Function to return final lab dataframe
    """
    Y=data[data['TYP']=='Y']
    # Splitting rows with tests seperated by ';' eg HKT=43.0 %; MCV=87.4 fl; MCH=27.6 pg
    Y=Y.set_index(['PATIENT_HASH', 'ZENTRUM_ID', 'PATIENT_ID', 'PAT_GEBDATUM',
           'PAT_GESCHLECHT', 'DATUM', 'TYP', 'TYP_EXT',  'ICD10',
           'SICHERHEIT'])
    Y=Y['TEXT'].str.split(';').explode().reset_index()
    Y['TEXT']=Y['TEXT'].str.replace(r'^\s+',r'')
    
     # Format eg. Eosinophils 10+ % (2 - 4)
    Y_1=Y[Y.TEXT.str.contains(r'\([.0-9\s]+-[.0-9\s]+\)')].copy()
    Y_1[['Lab_test','Result']]=Y_1['TEXT'].str.split('\s',expand=True,n=1)
    Y_1.dropna(inplace=True)
    Y_1['Inference']=Y_1['Result'].apply(result_inference,axis=1)
    
     # Format LEUKO=4.5
    Y_2=Y[Y.TEXT.str.contains(r'[a-zA-Z0-9\s]+=[a-zA-Z0-9\s]+')]
    # Seperating Lab tests to their test name and results by searching for a '='
    Y_2[['Lab_test','Result']]=Y_2['TEXT'].str.split('=',expand=True,n=1)
    Y_2['Inference']=Y_2.apply(result_inference,axis=1)
    
    # Merge the data frames of the two formats
    Y_new=pd.concat([Y_1,Y_2],axis=0)
        
    return Y_new

def page(data):
    """
    Processing date columns and add age columns
    """
    def DATUM_preprocessing(data):
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
    
    def PAT_GEBDATUM_preprocessing(data):
  
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
    def age( data):

        _ = data['entry_date'] - data['birth_date']
        data['age'] = (_ / np.timedelta64(1, 'Y')).round(1)
        
        return data   
    return data

#function to replace values with numbers(Can be used as ordinal values)
def Inference(row):
    """
    Lab results inference encoding
    """
    if row['Inference'] =='Low':
        return 1
    if row['Inference'] =='High':
        return 1
    if row['Inference'] =='Very high':
        return 2
    if row['Inference'] =='Very low':
        return 2
    if row['Inference'] =='Negative':
        return 2
    if row['Inference'] =='Normal':
        return 0
    return 'Other'

def piv(findata,Y_new):
    """
    Extracting and encoding test results most commonly found in patients suffering from Familiäre Hypercholesterinämie
    """
    #drop the text columns
    findata=findata.dropna(subset=['TEXT'])
    #make the final column "A"-absent in the presence of "usschl" in any row for any disease
    findata.loc[findata['TEXT'].str.contains('usschl'), ['SICHERHEIT']] = 'A'
    #make lab results columns instead of rows
    nolab=findata[findata != 'Y']
    #y_3 contains the desired rows
    Y_3=Y_new[["PATIENT_HASH",'DATUM','Lab_test','Result','Inference']]
    #group patients text to get the rows combined(later used for filtering)
    Y_text=nolab.groupby(['PATIENT_HASH'])['TEXT'].apply(','.join).reset_index()
    #merge the text to the lab results to filter based on diseases
    text_lab=Y_3.merge(Y_text,on='PATIENT_HASH',how='left')
    #filter the result finally
    text_lab=text_lab[(text_lab['Inference']!='Normal')&(text_lab.TEXT.str.contains('Familiäre Hypercholesterinämie'))]
    #selected only 20 top lab tests
    text_interest=pd.DataFrame(text_lab.Lab_test.value_counts())
    list=text_interest.reset_index().head(20).iloc[:, 0]
    pivot=Y_3[(Y_3.Lab_test.isin(list))]
    pivot['label'] = pivot.apply (lambda row: Inference(row), axis=1)
    merge=pd.pivot_table(pivot, values = 'label', index=['PATIENT_HASH','DATUM'], columns = 'Lab_test').reset_index()
    final_with_lab=nolab.merge(merge,on=['PATIENT_HASH','DATUM'],how='left')
    #replace ICD10 na with "none"
    final_with_lab.ICD10.fillna('none',inplace=True)
    #drop values where sich is nan
    final_with_lab.dropna(subset=['SICHERHEIT'],inplace=True)
    final_with_lab=final_with_lab.fillna(0)
    return final_with_lab


def Status(final_with_lab):
    """
    Adding the status 1 for presence & status 0 for absence of disease
    """
    l = ['amiliäre Hyperchol', 'amiliäre hyperchol', 'amiliären Hyperchol', 'amiliären hyperchol']  
    regstr = '|'.join(l)
    final_with_lab.loc[(final_with_lab['TEXT'].str.contains(regstr)),'ICD10']='E78.01'
    final_with_lab.loc[(final_with_lab['SICHERHEIT'].isin(['G','Z'])) & (final_with_lab['ICD10'].isin(['E78.01']))& (final_with_lab['TYP'].isin(['*'])), 'Status'] = 1
    final_with_lab.Status.fillna(0,inplace=True)
    return final_with_lab

def clus(clusters,final_with_lab):
    """
    Adding the clusters to data
    """
     #clusters.rename({'NEW_COLUMN':'clusters'},axis=1,inplace=True)
    clusters=clusters[['PATIENT_HASH','clusters']]
    final_with_lab=final_with_lab.merge(clusters,on='PATIENT_HASH',how='left')
    return final_with_lab

def encoding(final_with_lab):
    """
    Getting the train and test set data
    """
    model=final_with_lab.drop(columns=['TYP','SICHERHEIT'])
    #mean encode ICD10 as feature
    mean_encode=model.groupby('ICD10').size()/len(model)
    model.loc[:,'ICD_mean_encode']=model['ICD10'].map(mean_encode)
    model.drop(columns=['ICD10'],inplace=True)
    
    #grouping columns from the model to obtain individual id's
    #model1
    m1=model[['PATIENT_HASH','is_E75.22', 'is_E78.01',
       'is_E78.3', 'is_E71.3', 'is_disease', 'CHOL', 'CRP', 'ERY', 'FT4', 'GGT', 'GLU', 'GPT', 'HB', 'HDL',
       'HKT', 'HS', 'LDL', 'MCH', 'MCHC', 'MCV', 'TRI', 'TRIG', 'TSH','TPO',
       'VITD25','ICD_mean_encode','Status']]
    m1=m1.groupby(['PATIENT_HASH']).sum()
    #model 2
    m2=model[['PATIENT_HASH','clusters','PAT_GESCHLECHT','ZENTRUM_ID','PATIENT_ID']]
    m2=m2.groupby(['PATIENT_HASH']).last()
    #merge the two
    model1=m1.merge(m2,on=['PATIENT_HASH'],how='left')
    #fill the clusters
    model1.clusters=model1.clusters.fillna(100)
    #make non zero or one status one:
    model1.loc[model1['Status']==2, 'Status'] = 1
    #subset the columns
    model1=model1[['CHOL', 'CRP', 'ERY', 'FT4', 'GGT', 'GLU', 'GPT', 'HB', 'HDL',
       'HKT', 'HS', 'LDL', 'MCH', 'MCHC', 'MCV', 'TRI', 'TRIG', 'TSH','TPO',
       'VITD25','ICD_mean_encode','Status','PAT_GESCHLECHT','ZENTRUM_ID',
        'clusters','is_E71.3','is_E75.22', 'is_E78.01' ,'is_E78.3','is_disease']]
    #dummy encode zentrum and pad
    model1 = pd.get_dummies(model1, columns=['ZENTRUM_ID'], drop_first=True)
    model1= pd.get_dummies(model1, columns=['PAT_GESCHLECHT'], drop_first=True)
    #train&test split by sampling
    df1 = model1[model1.Status == 0].sample(1000)
    df2 = model1[~model1.index.isin(df1.index)]
    X=df2.loc[:, df2.columns != 'Status']
    y=df2.loc[:, df2.columns == 'Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=8675309)
    return X_train, X_test, y_train, y_test



def run_exps(X_train, X_test, y_train, y_test):
    '''
    Running all the models where the best performing model is RandomForest
    '''
    
    dfs = []
    models = [
              ('LogReg', LogisticRegression()), 
              ('RF', RandomForestClassifier()),
              ('KNN', KNeighborsClassifier()),
              ('SVM', SVC()), 
              ('GNB', GaussianNB()),
              ('XGB', XGBClassifier())
            ]
    results = []
    names = []
    scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']
    target_names = ['no_e78', 'yes_e78']
    for name, model in models:
            kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=90210)
            cv_results = model_selection.cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring)
            clf = model.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            print(name)
            print(classification_report(y_test, y_pred, target_names=target_names))
            print(confusion_matrix(y_test, y_pred,labels=[0,1]))
    results.append(cv_results)
    names.append(name)
    this_df = pd.DataFrame(cv_results)
    this_df['model'] = name
    dfs.append(this_df)
    final = pd.concat(dfs, ignore_index=True)
    return final

def best_model(X_train, X_test, y_train, y_test,final_with_lab):
    from sklearn.inspection import permutation_importance
    #import shap
    from matplotlib import pyplot as plt


    model = RandomForestClassifier(n_estimators=100)

    kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=90210)
    cv_results = model_selection.cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring)
    clf = rf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred, target_names=target_names))
    print(confusion_matrix(y_test, y_pred,labels=[0,1]))

    model.fit(X_train, y_train)
    # view the feature scores
    feature_scores = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    # Creating a seaborn bar plot

    f, ax = plt.subplots(figsize=(30, 24))
    ax = sns.barplot(x=feature_scores, y=feature_scores.index, data=final_with_lab)
    ax.set_title("Visualize feature scores of the features")
    ax.set_yticklabels(feature_scores.index)
    ax.set_xlabel("Feature importance score")
    ax.set_ylabel("Features")
    plt.show()
    score=accuracy_score(y_test,y_pred)
    
    return score

def first_data(data):
    """
    Get right data for text processing
    """
    #selecting only anamnestic data for clustering
    data=data[data.TYP=='A']
    data=data.groupby(['PATIENT_HASH'])['TEXT'].apply(','.join).reset_index()
    #adding word count for themedical texts
    data['notes_word_count']=data['TEXT'].apply(lambda x:len(x.strip().split()))
    return data

def clean_text(text, for_embedding=False):
    """
    - remove any html tags (< /br> often found)
    - Keep only ASCII + European Chars and whitespace, no digits
    - remove single letter chars
    - convert all whitespaces (tabs etc.) to single wspace
    if not for embedding (but e.g. tdf-idf):
    - all lowercase
    - remove stopwords, punctuation and stemm
    """

    stemmer = SnowballStemmer("german")
    stop_words = stopwords.words("german")
    wods = ['hypercholesterinaemi','hypercholesterinami','hypercholesterinamie','hypercholesterinami hypercholesterinami','hypercholesterinamie','hypercholesterinami hypercholesterinami']
    stop_words.extend(wods)

    RE_WSPACE = re.compile(r"\s+", re.IGNORECASE)
    RE_TAGS = re.compile(r"<[^>]+>")
    RE_ASCII = re.compile(r"[^A-Za-zÀ-ž ]", re.IGNORECASE)
    RE_SINGLECHAR = re.compile(r"\b[A-Za-zÀ-ž]\b", re.IGNORECASE)
    if for_embedding:
        # Keep punctuation
        RE_ASCII = re.compile(r"[^A-Za-zÀ-ž,.!? ]", re.IGNORECASE)
        RE_SINGLECHAR = re.compile(r"\b[A-Za-zÀ-ž,.!?]\b", re.IGNORECASE)

    text = re.sub(RE_TAGS, " ", text)
    text = re.sub(RE_SINGLECHAR, " ", text)
    text = re.sub(RE_WSPACE, " ", text)

    word_tokens = word_tokenize(text)
    words_tokens_lower = [word.lower() for word in word_tokens]

    if for_embedding:
        # no stemming, lowering and punctuation / stop words removal
        words_filtered = word_tokens
    else:
        words_filtered = [
            stemmer.stem(word) for word in words_tokens_lower if word not in stop_words
        ]

    text_clean = " ".join(words_filtered)

    return text_clean


def vec(df):
    """
    Applying the function and gettin the vectorized dataframe X
    """
    df["comment_clean"] = df["TEXT"].map(
        lambda x: clean_text(x, for_embedding=False) if isinstance(x, str) else x
    )
    df["comment_clean"]=df.comment_clean.str.split().map(lambda x:" ".join(s for s in x if len(s) > 8))

    """
    Compute unique word vector with frequencies
    exclude very uncommon (<10 obsv.) and common (>=30%) words
    use pairs of two words (ngram)
    """
    vectorizer = TfidfVectorizer(
        analyzer="word", max_df=0.3, min_df=10, ngram_range=(1, 2), norm="l2"
    )
    X=vectorizer.fit_transform(df["comment_clean"])
    return X

def get_cluster(X,df):
    """
    Get dataframe with clusters
    """
    number_of_clusters = 15

    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    model = KMeans(n_clusters=number_of_clusters, 
                   init='k-means++', 
                   max_iter=100, # Maximum number of iterations of the k-means algorithm for a single run.
                   n_init=1)  # Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.

    model.fit(X)
    df['clusters'] = pd.Series(model.predict(X), index=df.index)
    return df


def main():
    # Generate clusters
    df= pd.read_csv('/home/jovyan/amedes_challenge/data/interim/data_extract_preprocessed.csv')    
    df=first_data(df)
    X=vec(df)
    clusters=get_cluster(X,df)
    
    data = pd.read_csv('/home/jovyan/amedes_challenge/data/interim/data_extract_preprocessed.csv')    
    #reading the cluster files 
    Y_new=lab_processing(data)
    data=page(data)
    final_with_lab=piv(data,Y_new)
    final_with_lab=Status(final_with_lab)
    final_with_lab=clus(clusters,final_with_lab)
    X_train, X_test, y_train, y_test=encoding(final_with_lab)
    final=run_exps(X_train, X_test, y_train, y_test)
    score=best_model(X_train, X_test, y_train, y_test,final_with_lab)
    
    
if __name__ == "__main__":
    main()
