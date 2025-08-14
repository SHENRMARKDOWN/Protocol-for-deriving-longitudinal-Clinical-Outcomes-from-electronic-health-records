from b_input_args import parse_arguments
from a_semi_model import Model_prediction,train_model
from a_preprocess import format_input_data
import os
import tensorflow as tf
import numpy as np

# Define a function to print the head of a tensor
def print_tensor_head(name, tensor, head_size=5):
    print(f"Head of {name}: {tensor[:head_size]}")

if __name__ == '__main__':

    batch_size = 128

    # Parse command-line arguments
    args = parse_arguments()

    # Access the parsed arguments
    train_dfname = args.train_dfname
    # unlabel_dfname = args.unlabel_dfname
    test_dfname = args.test_dfname
    ftsname = args.ftsname
    other_ftsname = args.other_ftsname
    embed_dim = args.embed_dim
    embed_fname = args.embed_fname
    key_code = args.key_code
    nlp_fts = args.nlp_fts
    has_other_fts = args.has_other_fts
    normalize_count_fts = args.normalize_count_fts
    normalize_other_fts = args.normalize_other_fts
    # alpha_silver = args.alpha_silver
    # temp_silver = args.temp_silver
    output_directory = args.output_directory
    output_fname = args.output_fname
    # columns_min = args.columns_min
    # columns_max = args.columns_max
    epochs = args.epochs
    max_visits = args.max_visits
    flag_train_augment = args.flag_train_augment
    flag_cross_dataset = args.flag_cross_dataset
    number_labels = args.number_labels
    epoch_silver = args.epoch_silver
    layers_incident = args.layers_incident
    weight_prevalence = args.weight_prevalence
    weight_unlabel = args.weight_unlabel
    weight_additional = args.weight_additional
    flag_save_attention = args.flag_save_attention
    flag_load_model = args.flag_load_model
    flag_prediction = args.flag_prediction
    flag_relapse = args.flag_relapse
    multi_model = args.multi_model
    ordinal_score_method = args.ordinal_score_method

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    if flag_prediction > 0:
        print(
            "\n----predict novel data: instead of cross-validation or evaluation----\n")
    else:
        print(
            "\n----cross-validation with training data or on test data ----\n")

    # optimizer = tf.keras.optimizers.Adam()
    # optimizer_silver = tf.keras.optimizers.Adam()
    if has_other_fts != False:
        (Y_nlev, X_train, y_train, patient_num_train, date_train, weights_train, has_data_loc_train, data_embedding_train, 
            key_features_train, silver_train, X_other_fts_train) = format_input_data(train_dfname, ftsname, other_ftsname, embed_dim,
                                                                    embed_fname, key_code, nlp_fts, has_other_fts,
                                                                    normalize_count_fts, normalize_other_fts,
                                                                    max_visits)

        (X_unsuper, y_train_unlabel, patient_num_train_unlabel, date_train_unlabel, data_embedding_unsuper, 
            key_features_unsuper, silver_unsuper, X_other_fts_unsuper) = (X_train, y_train, patient_num_train, date_train, data_embedding_train, 
            key_features_train, silver_train, X_other_fts_train)

        weights_train_unsuper = tf.ones_like(weights_train)
        has_data_loc_train_unsuper = has_data_loc_train
        # weights_train_unsuper = weights_train
        
        (Y_nlev, X_test, y_test, patient_num_test, date_test, weights_test, has_data_loc_test, data_embedding_test,
            key_features_test, silver_test, X_other_fts_test) = format_input_data(test_dfname, ftsname, other_ftsname, embed_dim,
                                                                    embed_fname, key_code, nlp_fts, has_other_fts,
                                                                    normalize_count_fts, normalize_other_fts,
                                                                    max_visits)   

        # Print the head of each tensor
        # print_tensor_head("X_train", X_train)
        # print_tensor_head("y_train", y_train)
        # print_tensor_head("patient_num_train", patient_num_train)
        # print_tensor_head("date_train", date_train)
        # print_tensor_head("weights_train", weights_train)
        # print_tensor_head("data_embedding_train", data_embedding_train)
        # print_tensor_head("key_features_train", key_features_train)
        # print_tensor_head("silver_train", silver_train)
        # print_tensor_head("X_other_fts_train", X_other_fts_train) 
        # print_tensor_head("X_unsuper", X_unsuper)
        # print_tensor_head("y_train_unlabel", y_train_unlabel)
        # print_tensor_head("patient_num_train_unlabel", patient_num_train_unlabel)
        # print_tensor_head("date_train_unlabel", date_train_unlabel)
        # print_tensor_head("data_embedding_unsuper", data_embedding_unsuper)
        # print_tensor_head("key_features_unsuper", key_features_unsuper)
        # print_tensor_head("silver_unsuper", silver_unsuper)
        # print_tensor_head("X_other_fts_unsuper", X_other_fts_unsuper)
        # print(f"normalize_other_fts={normalize_other_fts}")


    else:
        (Y_nlev, X_train, y_train, patient_num_train, date_train, weights_train, has_data_loc_train, data_embedding_train, 
            key_features_train, silver_train) = format_input_data(train_dfname, ftsname, other_ftsname, embed_dim,
                                                                    embed_fname, key_code, nlp_fts, has_other_fts,
                                                                    normalize_count_fts, normalize_other_fts,
                                                                    max_visits)

        (X_unsuper, y_train_unlabel, patient_num_train_unlabel, date_train_unlabel, data_embedding_unsuper, 
            key_features_unsuper, silver_unsuper) = (X_train, y_train, patient_num_train, date_train, data_embedding_train, 
            key_features_train, silver_train)
        
        weights_train_unsuper = tf.ones_like(weights_train)
        has_data_loc_train_unsuper = has_data_loc_train

        (Y_nlev, X_test, y_test, patient_num_test, date_test, weights_test, has_data_loc_test, data_embedding_test,
            key_features_test, silver_test) = format_input_data(test_dfname, ftsname, other_ftsname, embed_dim,
                                                                    embed_fname, key_code, nlp_fts, has_other_fts,
                                                                    normalize_count_fts, normalize_other_fts,
                                                                    max_visits)   
    weights_test_unsuper = weights_test   
    has_data_loc_test_unsuper = has_data_loc_test
    # print_tensokr_head("weights_test_unsuper", weights_test_unsuper) 
    
    
    # (X_unsuper, y_train_unlabel, patient_num_train_unlabel,date_train_unlabel, data_embedding_unsuper,
    #     key_features_unsuper, silver_unsuper) = (X_train, y_train, patient_num_train, date_train, data_embedding_train, 
    #     key_features_train, silver_train)

    # weights_train_unsuper = np.ones_like(weights_train)

    # some shape parameters
    max_length = X_train.shape[1]
    max_length_unsuper = X_unsuper.shape[1]
    print(f"max_length={max_length}, max_length_unsuper={max_length_unsuper}")
    num_feature = data_embedding_train.shape[0]
    embedding_dim = data_embedding_train.shape[1]
    num_patient_train = X_train.shape[0]
    num_patient_unsuper = X_unsuper.shape[0]
    num_patient_test = X_test.shape[0]
    if has_other_fts == True:
        num_other_feature = X_other_fts_train.shape[2]


    # train_loss = tf.keras.metrics.Mean(name='train_loss')
    # train_loss_silver = tf.keras.metrics.Mean(name='train_loss_silver')
    # train_loss_classfication = tf.keras.metrics.Mean(name='train_smooth_loss_classfication')
    # train_smooth_loss = tf.keras.metrics.Mean(name='train_smooth_loss')
    # train_smooth_loss_unsuper = tf.keras.metrics.Mean(name='train_smooth_loss_unsuper')
    # train_keyfeature_loss = tf.keras.metrics.Mean(name='train_keyfeature_loss')
    # train_constrastive_loss = tf.keras.metrics.Mean(name='train_constrastive_loss')
    # train_constrastive_loss_MLP = tf.keras.metrics.Mean(name='train_constrastive_loss_MLP')
    # train_MLP_entropy_unsuper = tf.keras.metrics.Mean(name='train_MLP_entropy_unsuper')
    # train_MLP_consistency = tf.keras.metrics.Mean(name='train_MLP_consistency')
    # train_MLP_incident = tf.keras.metrics.Mean(name='train_MLP_incident')
    # train_metric = tf.keras.metrics.AUC(name='train_auc', )
    # valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    # valid_smooth_loss = tf.keras.metrics.Mean(name='valid_smooth_loss')
    # valid_metric = tf.keras.metrics.AUC(name='test_auc')

    # X_train = X_unsuper: [patient_num x T x num_fts]
    # y_train: [patient_num x T x 1]
    # patient_num_train, date_train, weights_train, silver_train = silver_unsuper: [patient_num x T]
    # data_embedding_train = data_embedding_unsuper: num_fts x fts_dim
    # key_features_train = key_features_unsuper:  
    if has_other_fts == True:
        data_embedding_train_reshaped = np.repeat(data_embedding_train[np.newaxis,:,:],num_patient_train,axis=0)
        key_features_train_reshaped = np.repeat(key_features_train[np.newaxis,np.newaxis,:],num_patient_train,axis=0)
        key_features_train_reshaped = np.repeat(key_features_train_reshaped,num_feature,axis=1)
        ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train,
                                                    patient_num_train, date_train, weights_train, has_data_loc_train,
                                                    data_embedding_train_reshaped, 
                                                    data_embedding_train_reshaped,
                                                    X_unsuper, weights_train_unsuper, has_data_loc_train_unsuper,
                                                    key_features_train_reshaped, 
                                                    key_features_train_reshaped, silver_train,
                                                    silver_unsuper,X_other_fts_train,X_other_fts_unsuper)) \
            .shuffle(buffer_size=500).batch(batch_size) \
            .prefetch(tf.data.experimental.AUTOTUNE).cache()

        data_embedding_test_reshaped = np.repeat(data_embedding_test[np.newaxis,:,:],num_patient_test,axis=0)
        key_features_test_reshaped = np.repeat(key_features_test[np.newaxis,np.newaxis,:],num_patient_test,axis=0)
        key_features_test_reshaped = np.repeat(key_features_test_reshaped,num_feature,axis=1)
        ds_test = tf.data.Dataset.from_tensor_slices((X_test, y_test,
                                                    patient_num_test, date_test, weights_test, has_data_loc_test,
                                                    data_embedding_test_reshaped, 
                                                    data_embedding_test_reshaped,
                                                    X_test, weights_test_unsuper, has_data_loc_test_unsuper,
                                                    key_features_test_reshaped, 
                                                    key_features_test_reshaped,X_other_fts_test,X_other_fts_test)) \
            .shuffle(buffer_size=500).batch(batch_size) \
            .prefetch(tf.data.experimental.AUTOTUNE).cache()
    else:
        data_embedding_train_reshaped = np.repeat(data_embedding_train[np.newaxis,:,:],num_patient_train,axis=0)
        key_features_train_reshaped = np.repeat(key_features_train[np.newaxis,np.newaxis,:],num_patient_train,axis=0)
        key_features_train_reshaped = np.repeat(key_features_train_reshaped,num_feature,axis=1)
        ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train,
                                                    patient_num_train, date_train, weights_train, has_data_loc_train,
                                                    data_embedding_train_reshaped, 
                                                    data_embedding_train_reshaped,
                                                    X_unsuper, weights_train_unsuper, has_data_loc_train_unsuper,
                                                    key_features_train_reshaped, 
                                                    key_features_train_reshaped, silver_train,
                                                    silver_unsuper)) \
            .shuffle(buffer_size=500).batch(batch_size) \
            .prefetch(tf.data.experimental.AUTOTUNE).cache()

        data_embedding_test_reshaped = np.repeat(data_embedding_test[np.newaxis,:,:],num_patient_test,axis=0)
        key_features_test_reshaped = np.repeat(key_features_test[np.newaxis,np.newaxis,:],num_patient_test,axis=0)
        key_features_test_reshaped = np.repeat(key_features_test_reshaped,num_feature,axis=1)
        ds_test = tf.data.Dataset.from_tensor_slices((X_test, y_test,
                                                    patient_num_test, date_test, weights_test, has_data_loc_test,
                                                    data_embedding_test_reshaped, 
                                                    data_embedding_test_reshaped,
                                                    X_test, weights_test_unsuper, has_data_loc_test_unsuper,
                                                    key_features_test_reshaped, 
                                                    key_features_test_reshaped)) \
            .shuffle(buffer_size=500).batch(batch_size) \
            .prefetch(tf.data.experimental.AUTOTUNE).cache()

    model = Model_prediction(Y_nlev, multi_model,max_length,max_length_unsuper,num_feature,num_other_feature,embedding_dim,layers_incident)
    savename_model = output_directory + output_fname + "_model"

    if os.path.exists(savename_model) and flag_load_model > 0:
        print("---------------------------------------------loadding saved model....................")
        model = tf.keras.models.load_model(savename_model)
    train_model(Y_nlev,model, ds_train, ds_test, weight_prevalence, weight_unlabel,
                                weight_additional, flag_save_attention, flag_prediction, flag_relapse, epochs=epochs, epoch_silver=epoch_silver, output_fname=output_fname,output_directory=output_directory,ordinal_score_method=ordinal_score_method)
    print("---------------------------------------------saving model....................")
    model.save(savename_model)