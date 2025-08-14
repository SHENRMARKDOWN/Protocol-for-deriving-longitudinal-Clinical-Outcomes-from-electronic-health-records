# -*- coding: utf-8 -*-
"""
TRAIN LAUNCHER
"""
import os
import argparse
import pandas as pd
import random
from pyhere import here


def parse_arguments():
    """Read user arguments"""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Required positional argument with default value from 'here' function
    parser.add_argument("--train_dfname", type=str, default=here("Data", "created_data", "latte_df_KG_train.csv"),
                        help="Path to the main CSV file")

    # Required positional argument with default value from 'here' function
    parser.add_argument("--unlabel_dfname", type=str, default=here("Data", "created_data", "latte_df_KG_unlabel.csv"),
                        help="Path to the main CSV file")

    # Required positional argument with default value from 'here' function
    parser.add_argument("--test_dfname", type=str, default=here("Data", "created_data", "latte_df_KG_test.csv"),
                        help="Path to the main CSV file")

    # Required positional argument with default value from 'here' function
    parser.add_argument("--ftsname", type=str, default=here("Data", "created_data", "latte_df_KG_fts.csv"),
                        help="Path to the feature set CSV file")

    # Required positional argument with default value from 'here' function
    parser.add_argument("--other_ftsname", type=str, default=here("Data", "created_data", "latte_df_demo_fts.csv"),
                        help="Path to the other feature set CSV file")
    
    # Required argument with default value
    parser.add_argument("--embed_dim", type=int, default=100,
                        help="Dimension of the embeddings to use")

    # Optional argument with default value
    parser.add_argument("--embed_fname", type=str, default="/n/data1/hsph/biostat/celehs/lab/xix636/KOMAP/Embedding/embedding_ONCE_dim1768.csv",
                        help="Path to the embedding file")

    # Optional argument with default value
    parser.add_argument("--key_code", type=str, default="PheCode:335", help="Key code (default: 'PheCode:335')")

    # Optional flag (True if present, False otherwise)
    parser.add_argument("--nlp_fts", action="store_true", help="Use NLP features")

    # Optional flag (True if present, False otherwise)
    parser.add_argument("--has_other_fts", action="store_true", help="Use other features")

    # Optional flag (True if present, False otherwise)
    parser.add_argument("--normalize_count_fts", action="store_true", help="Normalize count features")

    # Optional flag (True if present, False otherwise)
    parser.add_argument("--normalize_other_fts", action="store_true", help="Normalize other features")

    # # Optional argument with default value
    # parser.add_argument("--alpha_silver", type=float, default=0.8, help="Alpha value for silver (default: 0.2)")

    # # Optional argument with default value
    # parser.add_argument("--temp_silver", type=float, default=0.2, help="Temperature value for silver (default: 0.8)")

     # Directory to save the results
    parser.add_argument('--output_directory', type=str, default="/n/data1/hsph/biostat/celehs/lab/feh370/UPMC_MS/03PDDS_Relapse_Imputation_Latte/Result/", help='Directory to save the results')

    # Filename to save the result data
    parser.add_argument('--output_fname', type=str, default="latte_PDDS", help='Filename to save the result data')

    # Data beginning column index
    parser.add_argument('--columns_min', type=int, default=4, help='Data beginning column index')

    # Data end column index
    parser.add_argument('--columns_max', type=int, default=1000, help='Data end column index')

    # Training epochs
    parser.add_argument('--epochs', type=int, default=60, help='Total training epochs: silver + gold')

    # Max visits length
    parser.add_argument('--max_visits', type=int, default=50, help='Max visits length')

    # Flag to augment training data with different windows
    parser.add_argument('--flag_train_augment', type=int, default=1,
                        help='Flag_train_augment: 1 to augment training data with different windows, 0 otherwise')

    # Flag to perform evaluations on the test dataset or cross-validation using the training dataset
    parser.add_argument('--flag_cross_dataset', type=int, default=0,
                        help='Flag_cross_dataset: 1 to do evaluations on the test dataset, 0 to perform cross-validation using the training dataset')

    # Number of labels
    parser.add_argument('--number_labels', type=int, default=50, help='Number of labels')

    # Epochs of pre-training using silver labels
    parser.add_argument('--epoch_silver', type=int, default=10, help='Epochs of pre-training using silver labels')

    # Layers of GRUs for temporal modeling
    parser.add_argument('--layers_incident', type=str, default="80",
                        help='Layers of GRUs for temporal modeling, "80" for only one layer, "80,90" for two layers with 80 and 90 units')

    # Weights of EVER/NEVER prevalence learning
    parser.add_argument('--weight_prevalence', type=float, default=0.25, help='Weights of EVER/NEVER prevalence learning')

    # Weights of silver unlabeled data during training
    parser.add_argument('--weight_unlabel', type=float, default=0.3, help='Weights of silver unlabeled data during training')

    # Weights of contrastive representation learning
    parser.add_argument('--weight_contrastive', type=float, default=0.1,
                        help='Weights of contrastive representation learning')

    # Weight of learning temporally-smooth representation
    parser.add_argument('--weight_smooth', type=float, default=0.1,
                        help='Weight of learning temporally-smooth representation')

    # Weights of additional regularization for semi-supervised learning
    parser.add_argument('--weight_additional', type=float, default=0.1,
                        help='Weights of additional regularization for semi-supervised learning')

    # If save the learned weights and representation for visualization; 1: save, 0: do not save
    parser.add_argument('--flag_save_attention', type=int, default=1,
                        help='If save the learned weights and representation for visualization; 1: save, 0: do not save')

    # If reload the model; 1: reload the model, 0: do not reload
    parser.add_argument('--flag_load_model', type=int, default=0,
                        help='If reload the model; 1: reload the model, 0: do not reload')

    # If predict new data, where the labels are random and evaluation with the Y is invalid
    parser.add_argument('--flag_prediction', type=int, default=0,
                        help='If predict new data, where the labels are random and evaluation with the Y is invalid')

    # If MS relapse
    parser.add_argument('--flag_relapse', type=int, default=0,
                        help='If work on MS relapse')

    # Type of multi-category models
    parser.add_argument("--multi_model", type=str, default=here("base_logit"),
                        help="Specify multi-category model from base_logit (Base Category Logistic, default), cum_logit (Cumulative Logistic), or adj_logit (Adjacent Category Logistic)")

    # Method for calculating longitudinal ordinal scores
    parser.add_argument("--ordinal_score_method", type=str, default="weighted", choices=["cumulative", "weighted"],
                        help="Method for calculating longitudinal ordinal scores: 'cumulative' (cumulative probability) or 'weighted' (weighted probability)")

    args = parser.parse_args()



    return args