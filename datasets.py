from lifelines import datasets
from lifelines import CoxPHFitter
from lifelines.datasets import load_rossi, load_kidney_transplant
import pandas as pd 
import numpy as np 


def load_regression_dataset_lifelines():

    #load a censored dataset from lifelines
    cencored_data = datasets.load_regression_dataset()
    cph = CoxPHFitter(penalizer=0.1, l1_ratio=1.0) # sparse solutions,
    cph.fit(cencored_data, 'T', 'E')
    cph.print_summary()
    survival_probabilities_Cox_PH = cph.predict_survival_function(cencored_data)

    return survival_probabilities_Cox_PH, cencored_data



def load_rossi_data():

    rossi = load_rossi()
    cph = CoxPHFitter(penalizer=0.1, l1_ratio=1.0) # sparse solutions,
    cph.fit(rossi, 'week', 'arrest')
    cph.print_summary()
    survival_probabilities_Cox_PH = cph.predict_survival_function(rossi[:400])

    return survival_probabilities_Cox_PH, rossi[:400]


def load_transplant_data():

    panel_data = load_kidney_transplant()
    #panel_data = panel_data.drop(columns=['time'])
    cph = CoxPHFitter(penalizer=0.1, l1_ratio=1.0) # sparse solutions,
    cph.fit(panel_data, 'time', 'death')
    cph.print_summary()
    survival_probabilities_Cox_PH = cph.predict_survival_function(panel_data[:500])

    return survival_probabilities_Cox_PH, panel_data[:500]



def load_FLchain_data():

    data = pd.read_csv('D:\\DSS_Visual_Analytics_XAI\\K-medoids for censored data\\data\\flchain.csv')
    if 'Unnamed: 0' in list(data.columns):
        data = data.drop(columns=['Unnamed: 0'])
    data = data.drop(columns=['chapter'])
    new_sex = data['sex'].replace({'F':0,'M':1})
    data['sex'] = new_sex
    data = data[data['creatinine'].notna()]
    
    #Drop the age column because is highly correlated with the outcome (and that possibly affects the result / maybe)
    data = data.drop(columns=['age'])

    #Cox regression model
    cph = CoxPHFitter(penalizer=0.1, l1_ratio=1.0) # sparse solutions,
    cph.fit(data, 'futime', 'death')
    cph.print_summary()

    survival_probabilities_Cox_PH = cph.predict_survival_function(data[:500])

    return survival_probabilities_Cox_PH, data[:500]



def load_support2_data():

    data = pd.read_csv('D:\\DSS_Visual_Analytics_XAI\\K-medoids for censored data\\data\\support2.csv')
    if 'Unnamed: 0' in list(data.columns):
        data = data.drop(columns=['Unnamed: 0'])
    new_sex = data['sex'].replace({'male':0,'female':1})
    data['sex'] = new_sex
    new_dzgroup = data['dzgroup'].replace({'ARF/MOSF w/Sepsis':0,
                                                     'CHF':1,
                                                     'COPD':2,
                                                     'Cirrhosis':3,
                                                     'Colon Cancer':4,
                                                     'Coma':5,
                                                     'Lung Cancer':6,
                                                     'MOSF w/Malig':7})

    data['dzgroup'] = new_dzgroup

    new_dzclass = data['dzclass'].replace({'ARF/MOSF':0, 'COPD/CHF/Cirrhosis':1, 'Cancer':2, 'Coma':3})

    data['dzclass'] = new_dzclass

    new_income = data['income'].replace({'$11-$25k':0, 'under $11k':1,'$25-$50k':2,'>$50k':3 })

    data['income'] = new_income

    new_race = data['race'].replace({'other':0, 'white':1, 'black':2, 'hispanic':3, 'asian':4})

    data['race'] = new_race

    new_ca = data['ca'].replace({'metastatic':0, 'no':1, 'yes':2})

    data['ca'] = new_ca

    new_dnr = data['dnr'].replace({'no dnr':0, 'dnr after sadm':1, 'dnr before sadm':2 })

    data['dnr'] = new_dnr

    new_sfdm2 = data['sfdm2'].replace({'<2 mo. follow-up':0, 'no(M2 and SIP pres)':1,
                                 'SIP>=30':2, 'adl>=4 (>=5 if sur)':3, 'Coma or Intub':4 })

    data['sfdm2'] = new_sfdm2

    data['age'] = data['age'].apply(np.int64)

    #data = data.drop(columns=['adlp', 'urine', 'glucose', 'bun', 'totmcst', 'alb', 'income', 'adls', 'bili', 'pafi', 'ph', 'prg2m', 'edu', 'prg6m', 'sfdm2' ])

    #Oι εγγραφες που φευγουν απο τα δεδομένα είναι πάρα πολλές
    data = data.dropna()

    print(data.isnull().sum())

    return data



def load_friendster_data():
    data = pd.read_csv('D:\DSS_Visual_Analytics_XAI\K-medoids for censored data\data\friendsterData.csv')
    return data