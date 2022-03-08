from lifelines import CoxPHFitter


def run_Cox_PH_model(data, time_column, death_column):
    '''Run a Cox PH model'''

    cph = CoxPHFitter(penalizer=0.1, l1_ratio=1.0) 
    cph.fit(data, time_column, death_column)

    cph.print_summary()

    survival_probabilities_Cox_PH = cph.predict_survival_function(data)

    return survival_probabilities_Cox_PH