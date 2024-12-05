# ource activate env_incident
from pyhere import here
import pandas as pd
import statistics
from util import sigmoid
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import roc_auc_score

def format_input_data(dfname, ftsname, other_ftsname, embed_dim, embed_fname, key_code, nlp_fts=False,has_other_fts=False, normalize_count_fts=False, normalize_other_fts=True, max_visits = 12):

    # read csvs
    df = pd.read_csv(here(dfname))
    embed_df = pd.read_csv(embed_fname, index_col=0).T
    count_fts = pd.read_csv(here(ftsname))['x'].values
    if has_other_fts == True:
        other_fts = pd.read_csv(here(other_ftsname))['x'].values
    
    # transform silver into [0,1]
    silver = df["silver"]
    # silver = np.array(silver).argsort().argsort()
    # silver = silver/np.max(silver)
    # if nlp_fts == True:
    #     silver = sigmoid((np.log(1+df["Utilization_EHR"]) - alpha_silver * np.log(1+df["Utilization_NLP"])) / temp_silver)
    # else:
    #     silver = sigmoid(np.log(1+df["Utilization_EHR"]) / temp_silver)

    df[count_fts] = df[count_fts].apply(lambda x: np.log(1+x))

    # conditional standardization
    scaler = StandardScaler()
    if normalize_count_fts:
        df[count_fts] = scaler.fit_transform(df[count_fts])
    if has_other_fts == True:
        if normalize_other_fts == True:
            df[other_fts] = scaler.fit_transform(df[other_fts])

    max_T = max_visits
    num_patient = int(df["T"].shape[0] / max_T)
    X = df[count_fts].values.reshape((num_patient, max_T, len(count_fts)))
    y = df["Y"].values.reshape((num_patient, max_T,1))
    patient_num = df["ID"].values.reshape((num_patient, max_T))
    T = df["T"].values.reshape((num_patient, max_T))

    has_label_loc = ~np.isnan(y) # Find positions of NaN values
    weights = np.where(has_label_loc, 1, 0) # Create a new array with 1s at not NaN positions and 0s at NaN
    #y[has_label_loc & (y < 4.)] = 0
    #y[has_label_loc & (y >= 4.)] = 1
    y[np.isnan(y)] = -1

    has_data_loc = df["bool_data"].values.astype(int).reshape((num_patient, max_T,1))
    has_data_loc = has_data_loc * weights # ensure the consistency of missing data with missing labels

    embedding = embed_df[count_fts].values.T[:,:embed_dim] # shape: [num_fts,embed_dim]
    key_embedding = embed_df[key_code].values[:embed_dim]
    silver = silver.values.reshape((num_patient, max_T))

    #! evaluate silver value and create better silver values
    
    # comments: all patients should have at least one label:
        # (1) Scenario 1: the labels are all 0/missing => 0 (of course, this contains some false negatives)
        # (2) Scenario 2: the labels are all 1/missing => >=1
        # (3) Scenario 3: the labels are 0/1/missing => >=1

    y_dich = y[has_label_loc].reshape(-1)
    y_med = statistics.median(y_dich)
    Y_nlev = int(max(y_dich))
    if all(y_dich >= y_med): 
        y_dich = y_dich > y_med
        y_prevalence = np.sum((y>y_med) * weights, axis=1)
    else:
        y_dich = y_dich >= y_med
        y_prevalence = np.sum((y>=y_med) * weights, axis=1)

        
    y_prevalence = np.where(y_prevalence >= 1.0, 1.0, 0.0).astype(np.float32)
    AUC_incidence = roc_auc_score(y_true=y_dich , y_score=silver[has_label_loc[:,:,0]].reshape(-1))
    silver_prevalence = np.mean(silver, axis=1)
    AUC_prevalence = roc_auc_score(y_true=y_prevalence.reshape(-1), y_score=silver_prevalence.reshape(-1))

    print("AUC_incidence = ",AUC_incidence," and AUC_prevalence = ", AUC_prevalence)

    if has_other_fts == True:
        z = df[other_fts].values.reshape((num_patient, max_T, len(other_fts)))
        return Y_nlev,X,y,patient_num,T,weights,has_data_loc, embedding,key_embedding,silver,z
    else:
        return Y_nlev,X,y,patient_num,T,weights,has_data_loc, embedding,key_embedding,silver

# # Example usage:

# if __name__ == '__main__':
#     output = format_input_data(
#         dfname="Data/created_data/latte_df_lookbackforward1mo_KG_ONCE_train.csv",
#         ftsname="Data/created_data/latte_df_lookbackforward1mo_KG_ONCE_fts.csv",
#         other_ftsname="Data/created_data/latte_df_lookbackforward1mo_demo_fts.csv",
#         embed_dim = 200,
#         embed_fname="/n/data1/hsph/biostat/celehs/lab/xix636/KOMAP/Embedding/embedding_ONCE_dim1768.csv",
#         key_code="PheCode:335",
#         nlp_fts=True,
#         other_fts=True,
#         normalize_count_fts=False,
#         normalize_other_fts=False,
#         alpha_silver=0.2,
#         temp_silver=0.8
#     )

#     for array in output:
#         print(array.shape)