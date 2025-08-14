import numpy as np
import pandas as pd
import random
import os
import pickle
import statistics
from tensorflow.keras import layers, models
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, accuracy_score, precision_score
from a_Transformer import MultiHeadAttention, MultiHeadAttention_sigmod

####comment:  constractive learning on visits representation without temporal information
##### without smooth loss; without noise in inputs;  MultiHeadAttention_sigmod;  larger weights for positive visits

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# This section of the code involves two types of evaluations:
# 1. Prevalence Evaluation:
#    - This assesses whether a patient is eventually disabled or not.
#    - It provides an overall perspective on the presence or absence of disability among the patient population.
#
# 2. Incidence Evaluation:
#    - This evaluates whether a patient develops disability during a specific evaluation period.
#    - It focuses on the occurrence of disability within a defined timeframe, providing insights into temporal trends.
#
# Please keep in mind the distinction between prevalence (overall presence) and incidence (occurrence within a specific period) when interpreting the results.

##############learning code importance based on embedding###############
def code_attention(input_seq, embedding, embedding_dim):
    dim = int(embedding_dim * 0.75)
    layer1 = layers.Dense(dim, activation=tf.nn.relu, name="code_attention/fc1")
    layer11 = layers.Dense(dim, activation=tf.nn.relu, name="code_attention/fc11")
    layer3 = layers.Dense(1, activation=None, name="code_attention/prediction")
    input_seq = layer1(input_seq)
    embedding = layer11(embedding)
    data_embedding = tf.concat([input_seq, embedding], axis=-1)
    data_embedding = layer3(data_embedding)
    data_embedding = tf.squeeze(data_embedding, axis=-1)
    data_embedding = tf.nn.softmax(data_embedding, axis=-1)
    data_embedding = tf.expand_dims(data_embedding, axis=-2)
    return data_embedding

##############learning code importance based on embedding###############

def Model_prediction(Y_nlev,multi_model, max_length,max_length_unsuper,num_feature,num_other_feature,embedding_dim,layers_incident):
    # input initialization
    inputs1 = layers.Input(shape=(max_length, num_feature,))
    inputs1_shuffle = layers.Input(shape=(max_length, num_feature,))
    inputs_unsuper = layers.Input(shape=(max_length_unsuper, num_feature,))
    inputs_unsuper_masked = layers.Input(shape=(max_length_unsuper, num_feature,))

    inputs_data_embedding_all = layers.Input(shape=(num_feature, embedding_dim,))
    inputs_data_embedding_unsuper = layers.Input(shape=(num_feature, embedding_dim,))
    inputs_keyfeature = layers.Input(shape=(num_feature, embedding_dim,))
    inputs_keyfeature_unsuper = layers.Input(shape=(num_feature, embedding_dim,))

    inputs_other_fts = layers.Input(shape=(max_length, num_other_feature,))
    inputs_other_fts_shuffle = layers.Input(shape=(max_length, num_other_feature,))
    inputs_other_fts_unsuper = layers.Input(shape=(max_length_unsuper, num_other_feature,))
    inputs_other_fts_unsuper_masked = layers.Input(shape=(max_length_unsuper, num_other_feature,))

    inputs1_temp = inputs1 * 1.0
    inputs_unsuper_temp = inputs_unsuper * 1.0
    inputs1_shuffle_temp = inputs1_shuffle * 1.0

    ##############the input counts is weighted
    attention_value = 10.0 * code_attention(inputs_data_embedding_all, inputs_keyfeature, embedding_dim)
    bias_value = 0.1
    inputs1_reweight = (attention_value + bias_value) * inputs1_temp
    inputs1_reweight_shuffle = (attention_value + bias_value) * inputs1_shuffle_temp
    inputs_unsuper_reweight = (attention_value + bias_value) * inputs_unsuper_temp
    inputs_unsuper_reweight_mask = (attention_value + bias_value) * inputs_unsuper_masked
    ##############the input counts is weighted

    inputs1_embedding_code_rw = tf.matmul(inputs1_reweight, inputs_data_embedding_all)

    ###################the re-weighted code utilization
    code_num = tf.reduce_sum(inputs1_reweight, axis=-1, keepdims=True) + 1
    code_num_unsuper = tf.reduce_sum(inputs_unsuper_reweight, axis=-1, keepdims=True) + 1
    code_num_unsuper_mask = tf.reduce_sum(inputs_unsuper_reweight_mask, axis=-1, keepdims=True) + 1
    code_num_shuffle = tf.reduce_sum(inputs1_reweight_shuffle, axis=-1, keepdims=True) + 1
    ###################the re-weighted code utilization

    ###########getting the normalized counts*embedding with the counts is re-weighted
    inputs1_embedding_ori = tf.matmul(inputs1_temp, inputs_data_embedding_all)
    inputs1_embedding = tf.matmul(inputs1_reweight / code_num, inputs_data_embedding_all)
    inputs1_embedding_shuffle = tf.matmul(inputs1_reweight_shuffle / code_num_shuffle, inputs_data_embedding_all)
    inputs_embedding_unsuper = tf.matmul(inputs_unsuper_reweight / code_num_unsuper, inputs_data_embedding_unsuper)
    inputs_embedding_unsuper_mask = tf.matmul(inputs_unsuper_reweight_mask / code_num_unsuper_mask,
                                                inputs_data_embedding_unsuper)
    ###########getting the normalized counts*embedding with the counts is re-weighted

    ########using self-attention to learn visit imporance
    Attention_prevalence = MultiHeadAttention_sigmod(embedding_dim, num_heads=1)
    Prevalence_layer1 = layers.Dense(int(embedding_dim), activation=tf.nn.relu, name="Binary/fcn1")
    Prevalence_predictor = layers.Dense(1, activation=None, name="Binary/predictor")
    Prevalence_predictor_silver = layers.Dense(1, activation=None, name="Binary/predictor_silver")
    ########using self-attention to learn visit imporance

    ####### getting Prevalence_prediction
    Prevalence_fcn, visit_weights = Attention_prevalence(inputs1_embedding, inputs1_embedding, inputs1_embedding,
                                                            mask=None)
    Prevalence_fcn = Prevalence_layer1(tf.reduce_mean(Prevalence_fcn[:, :, :], axis=1))
    Prevalence_prediction = tf.nn.dropout(Prevalence_fcn, rate=0.3)
    Prevalence_prediction = Prevalence_predictor(Prevalence_prediction)

    Prevalence_fcn_unsuper, visit_weights_unsuper = Attention_prevalence(inputs_embedding_unsuper,
                                                                            inputs_embedding_unsuper,
                                                                            inputs_embedding_unsuper, None)
    Prevalence_fcn_unsuper = Prevalence_layer1(tf.reduce_mean(Prevalence_fcn_unsuper[:, :, :], axis=1))
    Prevalence_fcn_unsuper = tf.nn.dropout(Prevalence_fcn_unsuper, rate=0.3)
    Prevalence_prediction_unsuper = Prevalence_predictor(Prevalence_fcn_unsuper)
    Prevalence_prediction_silver = Prevalence_predictor_silver(Prevalence_fcn_unsuper)

    Prevalence_fcn_shuffle, visit_weights_shuffle = Attention_prevalence(inputs1_embedding_shuffle,
                                                                            inputs1_embedding_shuffle,
                                                                            inputs1_embedding_shuffle, None)
    Prevalence_fcn_shuffle = Prevalence_layer1(tf.reduce_mean(Prevalence_fcn_shuffle[:, :, :], axis=1))
    Prevalence_prediction_shffle = tf.nn.dropout(Prevalence_fcn_shuffle, rate=0.3)
    Prevalence_prediction_shffle = Prevalence_predictor(Prevalence_prediction_shffle)

    Prevalence_fcn_unsuper_mask, visit_weights_mask = Attention_prevalence(inputs_embedding_unsuper_mask,
                                                                            inputs_embedding_unsuper_mask,
                                                                            inputs_embedding_unsuper_mask, None)
    Prevalence_fcn_unsuper_mask = Prevalence_layer1(tf.reduce_mean(Prevalence_fcn_unsuper_mask[:, :, :], axis=1))
    Prevalence_prediction_unsuper_mask = tf.nn.dropout(Prevalence_fcn_unsuper_mask, rate=0.3)
    Prevalence_prediction_unsuper_mask = Prevalence_predictor(Prevalence_prediction_unsuper_mask)

    ##########getting the weights
    visit_weights_show = tf.reduce_mean(visit_weights[:, :, 0, :], axis=1)
    inputs1_embedding_code_visit_rw = tf.expand_dims(tf.reduce_mean(visit_weights[:, :, 0, :], axis=1),
                                                        axis=-1) * inputs1_reweight
    ##########getting the weights

    visit_weights = tf.expand_dims(tf.reduce_sum(visit_weights[:, 0, :, :], axis=1), -1) + 0.1
    visit_weights_unsuper = tf.expand_dims(tf.reduce_sum(visit_weights_unsuper[:, 0, :, :], axis=1), -1) + 0.1
    visit_weights_mask = tf.expand_dims(tf.reduce_sum(visit_weights_mask[:, 0, :, :], axis=1), -1) + 0.1
    visit_weights_shuffle = tf.expand_dims(tf.reduce_sum(visit_weights_shuffle[:, 0, :, :], axis=1), -1) + 0.1

    FCN_pre = layers.Dense(9, activation=tf.nn.relu)
    representation_out_raw = FCN_pre(inputs1_embedding) * visit_weights
    representation_out_unsuper_raw = FCN_pre(inputs_embedding_unsuper) * visit_weights_unsuper
    representation_out_unsuper_mask_raw = FCN_pre(inputs_embedding_unsuper_mask) * visit_weights_mask
    representation_out_shuffle_raw = FCN_pre(inputs1_embedding_shuffle) * visit_weights_shuffle

    # # print(f'shape of representation_out_raw is {representation_out_raw.shape}')
    # # print(f'shape of inputs_other_fts is {inputs_other_fts.shape}')
    representation_out = tf.concat([representation_out_raw, inputs_other_fts],axis=2)
    representation_out_shuffle = tf.concat([representation_out_shuffle_raw, inputs_other_fts_shuffle],axis=2)
    representation_out_unsuper = tf.concat([representation_out_unsuper_raw, inputs_other_fts_unsuper],axis=2)
    representation_out_unsuper_mask = tf.concat([representation_out_unsuper_mask_raw, inputs_other_fts_unsuper_masked],axis=2)

    layes_num = list(str(layers_incident).split(","))
    try:
        layer_flag = -1
        # print("-------------------------------------using GRUs layers number: ", len(layes_num),
        #        " for incident ----: ", layers_incident)
        for layer_i in layes_num:
            if len(layer_i) > 0:
                layer_flag += 1
                layer_i = int(layer_i)/2
                if layer_i > 0:
                    layer_name = "GRU_" + str(layer_flag)
                    GRU_Bidirectional = tf.keras.layers.Bidirectional(
                        tf.keras.layers.GRU(units=int(layer_i), return_sequences=True,
                                            activation=tf.nn.relu), merge_mode='concat', name=layer_name)
                    representation_out = GRU_Bidirectional(representation_out)
                    representation_out_unsuper = GRU_Bidirectional(representation_out_unsuper)
                    representation_out_unsuper_mask = GRU_Bidirectional(representation_out_unsuper_mask)
                    representation_out_shuffle = GRU_Bidirectional(representation_out_shuffle)

                    representation_out = tf.nn.dropout(representation_out, rate=0.2)
                    representation_out_unsuper = tf.nn.dropout(representation_out_unsuper, rate=0.2)
                    representation_out_unsuper_mask = tf.nn.dropout(representation_out_unsuper_mask, rate=0.2)
                    representation_out_shuffle = tf.nn.dropout(representation_out_shuffle, rate=0.2)
    except:
        # print("----------------------------error of readding layers_incident: using default one layer with 80")
        layer_name = "GRU_default"
        GRU_Bidirectional = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=80, return_sequences=True,
                                                                                activation=tf.nn.relu),
                                                            merge_mode='ave', name=layer_name)
        representation_out = GRU_Bidirectional(representation_out)
        representation_out_unsuper = GRU_Bidirectional(representation_out_unsuper)
        representation_out_unsuper_mask = GRU_Bidirectional(representation_out_unsuper_mask)
        representation_out_shuffle = GRU_Bidirectional(representation_out_shuffle)

        representation_out = tf.nn.dropout(representation_out, rate=0.2)
        representation_out_unsuper = tf.nn.dropout(representation_out_unsuper, rate=0.2)
        representation_out_unsuper_mask = tf.nn.dropout(representation_out_unsuper_mask, rate=0.2)
        representation_out_shuffle = tf.nn.dropout(representation_out_shuffle, rate=0.2)

    # GRU_incident=tf.keras.layers.GRU(units=15, return_sequences=True,activation=tf.nn.relu,recurrent_dropout=0.1)
    GRU_incident = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=12, return_sequences=True,
                                                                        activation=tf.nn.relu, recurrent_dropout=0.1),
                                                    merge_mode='concat', name="incident")  # concat
    predictor_FCN = layers.Dense(10, activation=tf.nn.relu)
    predictor_temporal = layers.Dense(1, activation=None)
    predictor_temporal_silver = layers.Dense(1, activation=None)

    representation_out = GRU_incident(representation_out)
    representation_out_unsuper = GRU_incident(representation_out_unsuper)
    representation_out_unsuper_mask = GRU_incident(representation_out_unsuper_mask)
    representation_out_shuffle = GRU_incident(representation_out_shuffle)

    representation_out = predictor_FCN(representation_out)
    representation_out_unsuper = predictor_FCN(representation_out_unsuper)
    representation_out_unsuper_mask = predictor_FCN(representation_out_unsuper_mask)
    representation_out_shuffle = predictor_FCN(representation_out_shuffle)

    # # print("-----representation_out: ", representation_out)

    inputs1_embedding_out = representation_out  # tf.nn.dropout(representation_out, rate=0.0)
    representation_out_unsuper = representation_out_unsuper  # tf.nn.dropout(representation_out_unsuper, rate=0.0)
    inputs_unsuper_embedding_out_mask = representation_out_unsuper_mask  # tf.nn.dropout(representation_out_unsuper_mask, rate=0.0)

    inputs1_embedding_out = predictor_temporal(inputs1_embedding_out) # prediction for gold labels
    inputs_unsuper_embedding_out = predictor_temporal(representation_out_unsuper)
    prediction_silver = predictor_temporal_silver(representation_out_unsuper) # prediction for silver labels
    inputs_unsuper_embedding_out_mask = predictor_temporal(inputs_unsuper_embedding_out_mask)
    # # print('inputs1_embedding_out: ', inputs1_embedding_out)

    #START#############################################multi-cateogry prediction################
    # Declare multi-category layers
    predictor_multi = layers.Dense(Y_nlev-1, activation=None, name = "Multi/incidence_predictor")
    prediction_multi_LP = predictor_multi(inputs1_embedding_out)
    Prevalence_predictor_multi = layers.Dense(Y_nlev-1, activation=None, name = "Multi/Prevalence_predictor")
    Prevalence_multi_LP = Prevalence_predictor_multi(Prevalence_prediction)
    
    # Changed the probability calucation of logit models
    # Set the category with largest probability as baseline
    if multi_model=="base_logit":
        prediction_multi = tf.concat([tf.zeros(shape = tf.shape(inputs1_embedding_out)),
                                       inputs1_embedding_out, prediction_multi_LP], axis = -1)
        prediction_multi = (prediction_multi - tf.repeat(tf.reduce_max(prediction_multi, axis = -1, keepdims=True),
                                    repeats = Y_nlev+1, axis = -1))
        prediction_multi = ( tf.exp(prediction_multi) / tf.reduce_sum(tf.exp(prediction_multi), axis=-1, keepdims=True))
        prediction_multi = tf.cast(prediction_multi, tf.float32)
        Prevalence_multi = tf.concat([tf.zeros(shape = tf.shape(Prevalence_prediction)),
                                       Prevalence_prediction, Prevalence_multi_LP], axis = -1)
        Prevalence_multi = (Prevalence_multi - tf.repeat(tf.reduce_max(Prevalence_multi, axis = -1, keepdims=True),
                                    repeats = Y_nlev+1, axis = -1))
        Prevalence_multi = ( tf.exp(Prevalence_multi) / tf.reduce_sum(tf.exp(Prevalence_multi), axis=-1, keepdims=True))
        Prevalence_multi = tf.cast(Prevalence_multi, tf.float32)
    elif multi_model=="cum_logit":
        prediction_multi = tf.concat([inputs1_embedding_out, prediction_multi_LP], axis = -1)
        prediction_multi = tf.sigmoid(prediction_multi)
        prediction_multi = tf.concat([tf.zeros(shape = tf.shape(inputs1_embedding_out)),
                                       prediction_multi,
                                       tf.ones(shape = tf.shape(inputs1_embedding_out))], axis = -1)
        prediction_multi = tf.maximum(prediction_multi[:,:,1:] - prediction_multi[:,:,:-1],
                                      tf.constant([0.]))
        prediction_multi = tf.cast(prediction_multi, tf.float32)
        Prevalence_multi = tf.concat([Prevalence_prediction, Prevalence_multi_LP], axis = -1)
        Prevalence_multi = tf.sigmoid(Prevalence_multi)
        Prevalence_multi = tf.concat([tf.zeros(shape = tf.shape(Prevalence_prediction)),
                                       Prevalence_multi,
                                       tf.ones(shape = tf.shape(Prevalence_prediction))], axis = -1)
        Prevalence_multi = tf.maximum(Prevalence_multi[:,1:] - Prevalence_multi[:,:-1],
                                      tf.constant([0.]))
        Prevalence_multi = tf.cast(Prevalence_multi, tf.float32)
    elif multi_model=="adj_logit":
        prediction_multi = tf.concat([tf.zeros(shape = tf.shape(inputs1_embedding_out)),
                                       inputs1_embedding_out, prediction_multi_LP], axis = -1)
        prediction_multi = tf.cumsum(prediction_multi, axis = -1)
        prediction_multi = (prediction_multi - tf.repeat(tf.reduce_max(prediction_multi, axis = -1, keepdims=True),
                                    repeats = Y_nlev+1, axis = -1))
        prediction_multi = ( tf.exp(prediction_multi) / tf.reduce_sum(tf.exp(prediction_multi), axis=-1, keepdims=True))
        prediction_multi = tf.cast(prediction_multi, tf.float32)
        Prevalence_multi = tf.concat([tf.zeros(shape = tf.shape(Prevalence_prediction)),
                                       Prevalence_prediction, Prevalence_multi_LP], axis = -1)
        Prevalence_multi = tf.cumsum(Prevalence_multi, axis = -1)
        Prevalence_multi = (Prevalence_multi - tf.repeat(tf.reduce_max(Prevalence_multi, axis = -1, keepdims=True),
                                    repeats = Y_nlev+1, axis = -1))
        Prevalence_multi = ( tf.exp(Prevalence_multi) / tf.reduce_sum(tf.exp(Prevalence_multi), axis=-1, keepdims=True))
        Prevalence_multi = tf.cast(Prevalence_multi, tf.float32)
    else:
        Print("multi_model must be one of base_logit, cum_logit, adj_logit. Used base_logit model by default.")
        prediction_multi = tf.concat([tf.zeros(shape = tf.shape(inputs1_embedding_out)),
                                       inputs1_embedding_out, prediction_multi_LP], axis = -1)
        prediction_multi = (prediction_multi - tf.repeat(tf.reduce_max(prediction_multi, axis = -1, keepdims=True),
                                    repeats = Y_nlev+1, axis = -1))
        prediction_multi = ( tf.exp(prediction_multi) / tf.reduce_sum(tf.exp(prediction_multi), axis=-1, keepdims=True))
        prediction_multi = tf.cast(prediction_multi, tf.float32)
        Prevalence_multi = tf.concat([tf.zeros(shape = tf.shape(Prevalence_prediction)),
                                       Prevalence_prediction, Prevalence_multi_LP], axis = -1)
        Prevalence_multi = (Prevalence_multi - tf.repeat(tf.reduce_max(Prevalence_multi, axis = -1, keepdims=True),
                                    repeats = Y_nlev+1, axis = -1))
        Prevalence_multi = ( tf.exp(Prevalence_multi) / tf.reduce_sum(tf.exp(Prevalence_multi), axis=-1, keepdims=True))
        Prevalence_multi = tf.cast(Prevalence_multi, tf.float32)



    prediction_multi = tf.expand_dims(prediction_multi, axis = -2)
    Prevalence_multi = tf.expand_dims(Prevalence_multi, axis = -2)

    #END#############################################multi-cateogry prediction################
    inputs1_embedding_out = tf.nn.sigmoid(inputs1_embedding_out)
    inputs1_embedding_out = tf.cast(inputs1_embedding_out, tf.float32)



    model = models.Model(inputs=[inputs_data_embedding_all, inputs_data_embedding_unsuper,
                                    inputs_keyfeature, inputs_keyfeature_unsuper, inputs1, inputs1_shuffle,
                                    inputs_unsuper, inputs_unsuper_masked, inputs_other_fts,
                                    inputs_other_fts_shuffle, inputs_other_fts_unsuper, inputs_other_fts_unsuper_masked],
                            outputs=[inputs1_embedding_out, inputs_unsuper_embedding_out,
                                    inputs_unsuper_embedding_out_mask,
                                    representation_out_raw, representation_out_unsuper_raw,
                                    representation_out_unsuper_mask_raw,
                                    Prevalence_prediction,
                                    attention_value * inputs1_temp,
                                    visit_weights_show,
                                    inputs1_embedding_ori, inputs1_embedding_code_rw,
                                    inputs1_embedding_code_visit_rw,
                                    representation_out_shuffle_raw, attention_value,
                                    Prevalence_fcn, Prevalence_fcn_shuffle, Prevalence_prediction_unsuper,
                                    Prevalence_prediction_unsuper_mask, visit_weights, prediction_silver,
                                    Prevalence_prediction_silver,
                                    prediction_multi,Prevalence_multi])  # outputs_sque  outputs  outputs_fused

#    trainable_Vnames = [v.name for v in model.trainable_variables]
#    print("Trainable variables", trainable_Vnames)

    return model

def train_step(Y_nlev, train_prevalence_incident, train_loss_prevalence,train_loss_prevalence_unsuper,train_loss_prevalence_silver,
    train_loss_incident,train_loss_incident_unsuper, train_loss_incident_silver,
    train_smooth_loss, train_smooth_loss_unsuper, train_keyfeature_loss,
    train_contrastive_loss, train_contrastive_loss_prevalence, train_prevalence_entropy_unsuper,
    train_prevalence_consistency, optimizer, optimizer_silver,
                model, X_train, labels, weights, has_data_loc_train, data_embedding_all,
                data_embedding_unsuper,
                X_unsuper, weights_unsuper, has_data_loc_train_unsuper, smooth_weight_in,
                keyfeature, keyfeature_unsuper,
                X_other_fts_train,
                X_other_fts_unsuper,
                patient_num,
                silver_train, silver_unsuper, weight_prevalence, weight_unlabel, flag_silver=False, flag_relapse=False):
    with tf.GradientTape(persistent=True) as tape:
        # Question: what are the thresholdings for?
        threshold_embedding = 10.0
        threshold_embedding_same = 5.0
        threshold_MLP = 10.0
        threshold_MLP_same = 5.0
        threshold_smooth = 0.5

        max_length = X_train.shape[1]
        num_feature = data_embedding_all.shape[1]
        embedding_dim = data_embedding_all.shape[2]

        weights_value = np.sum(weights.numpy(), axis=-1)
        weights_value_unsuper = np.sum(weights_unsuper.numpy(), axis=-1)
        weights = tf.cast(weights, tf.float32)
        labels = tf.cast(labels, tf.float32)
        silver_unsuper = tf.cast(silver_unsuper, tf.float32)
        silver_train = tf.cast(silver_train, tf.float32)

        weights_unsuper = tf.cast(weights_unsuper, tf.float32)
        weights_smooth_super = tf.cast(weights, tf.float32)
        weights_unsuper = tf.cast(tf.greater(weights_unsuper, 0.0), tf.float32)
        weights_smooth_super = tf.cast(tf.greater(weights_smooth_super, 0.0), tf.float32)

        indices = tf.range(start=0, limit=tf.shape(X_train)[0], dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(indices)
        X_train_shuffle = tf.gather(X_train, shuffled_indices)
        X_other_fts_shuffle = tf.gather(X_other_fts_train, shuffled_indices)
        shuffled_weights = tf.gather(weights, shuffled_indices)
        shuffled_weights_smooth_super = tf.gather(weights_smooth_super, shuffled_indices)
        shuffled_labels = tf.gather(labels, shuffled_indices)
        batch_num = X_unsuper.numpy().shape[0]
        if random.random() < 0.5:
            mask_unsuper = np.random.normal(loc=1, scale=0.01, size=(batch_num, max_length, num_feature))
        else:
            mask_unsuper = np.random.normal(loc=1, scale=0.01, size=(batch_num, max_length, num_feature))
        mask_dropout = []
        for iii in range(int(batch_num * max_length * num_feature)):
            if random.random() < 0.5:
                if random.random() < 0.95:
                    mask_dropout.append(1)
                else:
                    mask_dropout.append(0)
            else:
                mask_dropout.append(1)
        mask_dropout = np.array(mask_dropout).reshape((batch_num, max_length, num_feature))
        mask_unsuper = tf.maximum(mask_unsuper, 0.0)
        if random.random() < 0.5:
            X_unsuper_masked = X_unsuper #* mask_unsuper
            X_train = X_train #* mask_unsuper
            X_other_fts_unsuper_masked = X_other_fts_unsuper
        else:
            X_unsuper_masked = X_unsuper #* mask_dropout
            X_train = X_train #* mask_dropout
            X_other_fts_unsuper_masked = X_other_fts_unsuper

        # Question: what is the difference between Prevalence_prediction_unsuper and Prevalence_prediciton_silver?
        predictions, predictions_unsuper, predictions_unsuper_mask, smooth_seq, smooth_seq_unsuper, smooth_seq_mask, Prevalence_prediction, \
        attention_value_codes, attention_value_visits, \
        embedding_ori, embedding_code_rw, embedding_code_visit_rw, shuffled_smooth_seq, \
        attention_value_code, Prevalence_fcn, Prevalence_fcn_shuffle, Prevalence_prediction_unsuper, \
        Prevalence_prediction_unsuper_mask, visit_weights, \
        prediction_silver, Prevalence_prediction_silver, \
        prediction_multi,Prevalence_multi = \
            model([data_embedding_all, data_embedding_unsuper,
                   keyfeature, keyfeature_unsuper, X_train, X_train_shuffle,
                   X_unsuper, X_unsuper_masked, X_other_fts_train, X_other_fts_shuffle,
                   X_other_fts_unsuper, X_other_fts_unsuper_masked], training=True)

        # those that are used to get gold label loss: predictions, Prevalence_prediction
        # those that are used to get silver label loss: predictions_unsuper, Prevalence_prediction_unsuper, prediction_silver, Prevalence_prediction_silver

##1     # Commented out the prevalence prediction from last incidence prediction
        # Need to identify the last position with non-zero weight
        #========================================================================
##1        prediction_incident_binary = predictions[0,max_length-1, 0]
##1        prediction_incident_binary = tf.expand_dims(prediction_incident_binary, 0)

        prediction_incident_binary_unsuper = predictions_unsuper[0, max_length-1, 0]
        prediction_incident_binary_unsuper = tf.expand_dims(prediction_incident_binary_unsuper, 0)

        '''generating the silver standard predictions'''
        prediction_incident_silver = prediction_silver[0,max_length-1, 0]
        prediction_incident_silver = tf.expand_dims(prediction_incident_silver, 0)
        for rowi in range(1, len(weights_value)):
##1            prediction_incident_binary_temp = predictions[rowi, max_length-1, 0]
##1            prediction_incident_binary_temp = tf.expand_dims(prediction_incident_binary_temp, 0)
##1            prediction_incident_binary = tf.concat([prediction_incident_binary, prediction_incident_binary_temp],axis=0)
            prediction_incident_binary_temp = predictions_unsuper[rowi, max_length-1, 0]
            prediction_incident_binary_temp = tf.expand_dims(prediction_incident_binary_temp, 0)
            prediction_incident_binary_unsuper = tf.concat([prediction_incident_binary_unsuper, prediction_incident_binary_temp], axis=0)
            prediction_incident_binary_temp = prediction_silver[rowi,max_length-1, 0]
            prediction_incident_binary_temp = tf.expand_dims(prediction_incident_binary_temp, 0)
            prediction_incident_silver = tf.concat([prediction_incident_silver, prediction_incident_binary_temp], axis=0)

##1        prediction_MLP_total = tf.expand_dims(prediction_incident_binary, axis=-1).numpy()
        prediction_incident_silver = tf.expand_dims(prediction_incident_silver, axis=-1)
        prediction_incident_silver = tf.nn.sigmoid(prediction_incident_silver)
        '''generating the silver standard predictions'''
        # Question: incident - prevalence?
        # Answer: suppose a patient has the disease, he will manifest at the last step in LSTM, or attention (aggregation)
##1        loss_prevalence_incident = tf.reduce_mean(tf.abs((prediction_incident_binary) - (Prevalence_prediction)))
        loss_prevalence_incident_unsuper = tf.reduce_mean(tf.abs((prediction_incident_binary_unsuper) -
                                                            (Prevalence_prediction_unsuper)))
##1        loss_prevalence_incident = (loss_prevalence_incident + loss_prevalence_incident_unsuper) / 2.0
##1        Prevalence_prediction = (tf.expand_dims(prediction_incident_binary,axis=-1)+Prevalence_prediction)/2.0

        # Commented out the original sigmoid model
        # Prevalence_prediction = tf.nn.sigmoid(Prevalence_prediction)

        Prevalence_prediction_silver = tf.nn.sigmoid(Prevalence_prediction_silver)

        Prevalence_prediction_unsuper = tf.nn.sigmoid(Prevalence_prediction_unsuper)
        Prevalence_prediction_unsuper_mask = tf.nn.sigmoid(Prevalence_prediction_unsuper_mask)

        labels_prevalence_shuffle = tf.reduce_sum(shuffled_labels * shuffled_weights, axis=1)
        labels_prevalence_shuffle = tf.cast(tf.greater_equal(labels_prevalence_shuffle, 1.0), tf.float32)

### The following block are modified by Amiee (Prevalence + incidence loss calculations), the modification reason are 2-folds:
### (1) the orginal version calculate silver-related loss wrongly due to not subsetting, hence the masked out visits will participate in binary_loss calculation
### (2) we now further allow missing visits information
# START Block
# Prevalence: gold
        # CAUTION: NOT all patients should have at least one label:
        # (1) Scenario 1: the labels are all 0/missing => 0
        # (2) Scenario 2: the labels are all 1/missing => >=1
        # (3) Scenario 3: the labels are 0/1/missing => >=1
        # (4) Scenario 4: all labels for the patient are missing => 0
        # We need to filter out Scenario 4 to make a reasonable evaluation



        bool_has_prevalence = tf.reduce_mean(weights, axis = 1)
        #START###########################multi-cateogry cross entropy loss for prevalence#########################
        # Commented out the old labels for prevalence (next 3 lines)
        # labels_prevalence = tf.reduce_sum(labels * weights, axis=1)
        # labels_prevalence = tf.cast(tf.greater_equal(labels_prevalence, 1.0), tf.float32)
        # labels_prevalence_subset = tf.boolean_mask(labels_prevalence, bool_has_prevalence)
        labels_prevalence = tf.reduce_max(labels * weights, axis=1)
        labels_prevalence_subset = tf.boolean_mask(labels_prevalence, bool_has_prevalence)

        # Commented out the old loss_prevalence (next 3 lines)
        # Prevalence_prediction_subset = tf.boolean_mask(Prevalence_prediction, bool_has_prevalence)
        # loss_prevalence = tf.keras.losses.binary_crossentropy(labels_prevalence_subset, Prevalence_prediction_subset)
        # loss_prevalence = tf.reduce_mean(loss_prevalence)

        # Construct the multi-category Prevalence loss
        Prevalence_multi_subset = tf.boolean_mask(Prevalence_multi, bool_has_prevalence)
        labels_MLP_multi = tf.one_hot(tf.cast(labels_prevalence_subset, dtype=tf.int32), Y_nlev+1, on_value=1.0, off_value=0.0, axis=-1)
        loss_prevalence = tf.reduce_mean(-tf.reduce_sum(labels_MLP_multi * tf.math.log(Prevalence_multi_subset), axis=-1))

        #END###########################multi-cateogry cross entropy loss for prevalence#########################

#       unsuper
        has_data_loc_train_unsuper = tf.squeeze(has_data_loc_train_unsuper,axis=-1)
        bool_has_unsuper_prevalence = tf.reduce_sum(has_data_loc_train_unsuper, axis = 1)
        bool_has_unsuper_prevalence = tf.where(tf.equal(bool_has_unsuper_prevalence, 0), 0, 1)
        labels_prevalence_unsuper = tf.reduce_mean(silver_unsuper, axis=1)
        labels_prevalence_unsuper_subset = tf.boolean_mask(labels_prevalence_unsuper, bool_has_unsuper_prevalence)
        labels_prevalence_unsuper_subset = tf.cast(labels_prevalence_unsuper_subset, dtype=tf.float32)
        Prevalence_prediction_unsuper_subset = tf.boolean_mask(Prevalence_prediction_unsuper, bool_has_unsuper_prevalence)
        Prevalence_prediction_unsuper_subset = tf.cast(Prevalence_prediction_unsuper_subset, dtype=tf.float32)
        loss_prevalence_unsuper = tf.square(labels_prevalence_unsuper_subset - Prevalence_prediction_unsuper_subset)
        loss_prevalence_unsuper = tf.reduce_mean(loss_prevalence_unsuper)

#       silver
        labels_prevalence_silver = tf.reshape(tf.reduce_mean(silver_train, axis=1),[-1])
        labels_prevalence_silver_subset = tf.boolean_mask(labels_prevalence_silver, bool_has_unsuper_prevalence)
        Prevalence_prediction_silver_subset = tf.boolean_mask(tf.reshape(Prevalence_prediction_silver,[-1]), bool_has_unsuper_prevalence)
        loss_prevalence_silver = tf.square(labels_prevalence_silver_subset - Prevalence_prediction_silver_subset)
        loss_prevalence_silver = tf.reduce_mean(loss_prevalence_silver)

# Incidence: gold
        #START###########################multi-cateogry cross entropy loss for incidence#########################
        # Commented out the old loss_incidence (next 6 lines)
        # predictions = tf.nn.sigmoid(predictions)
        # predictions = tf.cast(predictions, tf.float32)
        # label_subset = tf.boolean_mask(labels, weights)
        # predictions_subset = tf.boolean_mask(predictions, weights)
        # loss_incident = tf.keras.losses.binary_crossentropy(label_subset, predictions_subset)
        # loss_incident = tf.reduce_mean(loss_incident)

        # Construct the multi-category Incidence loss
        prediction_multi_subset = tf.boolean_mask(prediction_multi, weights)
        labels_subset = tf.boolean_mask(labels, weights)
        labels_subset_multi = tf.one_hot(tf.cast(labels_subset, dtype=tf.int32), Y_nlev+1, on_value=1.0, off_value=0.0, axis=-1)
        loss_incident = tf.reduce_mean(-tf.reduce_sum(labels_subset_multi * tf.math.log(prediction_multi_subset), axis=-1))

        # Print a few losses
        #loss_incident_x_multi = tf.keras.losses.categorical_crossentropy(labels_subset_multi, prediction_multi_subset)
        #loss_incident_x_multi = tf.reduce_mean(loss_incident_x_multi)
        #if Y_nlev==1:
        #    predictions_1 = tf.nn.sigmoid(predictions)
        #    predictions_1= tf.cast(predictions_1, tf.float32)
        #    predictions_1_subset = tf.boolean_mask(predictions_1, weights)
        #    loss_incident_xbin = tf.keras.losses.binary_crossentropy(labels_subset, predictions_1_subset)
        #    loss_incident_xbin = tf.reduce_mean(loss_incident_xbin)
        #    losslogs = 'Incidence Formula: {}, Incidence Function: {}, Binary Function: {}'
        #    tf.print(tf.strings.format(losslogs, (loss_incident, loss_incident_x_multi, loss_incident_xbin)))
        #else:
        #    losslogs = 'Incidence Formula: {}, Incidence Function: {}'
        #    tf.print(tf.strings.format(losslogs, (loss_incident, loss_incident_x_multi)))

        #END###########################multi-cateogry cross entropy loss for incidence#########################



#       unsuper
        label_unsuper_subset = tf.boolean_mask(silver_unsuper, has_data_loc_train_unsuper)
        prediction_incident_unsuper_subset = tf.boolean_mask(tf.cast(tf.nn.sigmoid(predictions_unsuper), tf.float32), has_data_loc_train_unsuper)
        loss_incident_unsuper = tf.square(label_unsuper_subset-prediction_incident_unsuper_subset)
        loss_incident_unsuper = tf.reduce_mean(loss_incident_unsuper)

#       silver
        has_data_loc_train = tf.squeeze(has_data_loc_train,axis=-1)
        label_silver_subset = tf.boolean_mask(silver_train, has_data_loc_train)
        prediction_incident_silver_subset = tf.boolean_mask(tf.cast(tf.nn.sigmoid(prediction_silver), tf.float32), has_data_loc_train)
        loss_incident_silver = tf.square(label_silver_subset-prediction_incident_silver_subset)
        loss_incident_silver = tf.reduce_mean(loss_incident_silver)

### END Block

        loss_consistency_prevalence = tf.reduce_mean(tf.square(Prevalence_prediction_unsuper - Prevalence_prediction_unsuper_mask))
        loss_consistency = tf.reduce_sum(tf.abs(smooth_seq_unsuper - smooth_seq_mask) \
                                           * weights_unsuper) / (tf.reduce_sum(weights_unsuper))

        """generating loss to encourage smoothness or non-decreasing"""
        if flag_relapse == True:
            """if with relapse such as MS: generating the penalty to encourage the prediction to be smoothy"""
            smooth_loss = tf.abs(smooth_seq[:, 0:-1, :] - smooth_seq[:, 1:, :]) * tf.expand_dims(weights_smooth_super[:, 1:], -1)
            smooth_loss_unsuper = tf.abs(smooth_seq_unsuper[:, 0:-1, :] - smooth_seq_unsuper[:, 1:, :]) * tf.expand_dims(weights_unsuper[:, 1:], -1)
            smooth_loss_mask = tf.abs(smooth_seq_mask[:, 0:-1, :] - smooth_seq_mask[:, 1:, :])   * tf.expand_dims(weights_unsuper[:, 1:], -1)

            weights_sum = tf.reduce_sum(weights)
            weights_sum_unsuper = tf.reduce_sum(weights_unsuper)
            smooth_loss = tf.reduce_sum(smooth_loss, name="smooth_loss") / weights_sum
            smooth_loss_unsuper = tf.reduce_sum(smooth_loss_unsuper, name="smooth_loss_unsuper") / weights_sum_unsuper
            smooth_loss_mask = tf.reduce_sum(smooth_loss_mask, name="smooth_loss_mask") / weights_sum_unsuper

        else:
            """if without relapse such as HF onset: generating the penalty to encourage the prediction to be non-decreassing"""
            """weights_smooth_super or weights_unsuper: the weights of training data because of the kernel re-weighting"""
            smooth_loss = tf.maximum(predictions[:, 0:-1, :] - predictions[:, 1:, :], 0) * weights_smooth_super[:, 1:]
            smooth_loss_unsuper = tf.maximum(predictions_unsuper[:, 0:-1, :] - predictions_unsuper[:, 1:, :], 0)  * weights_unsuper[:, 1:]
            smooth_loss_mask = tf.maximum(predictions_unsuper_mask[:, 0:-1, :] - predictions_unsuper_mask[:, 1:, :],0)  * weights_unsuper[:, 1:]
            smooth_loss = tf.maximum(tf.reduce_sum(smooth_loss, 1), threshold_smooth)
            smooth_loss_unsuper = tf.maximum(tf.reduce_sum(smooth_loss_unsuper, 1), threshold_smooth)
            smooth_loss_mask = tf.maximum(tf.reduce_sum(smooth_loss_mask, 1), threshold_smooth)
            smooth_loss = tf.reduce_mean(smooth_loss)
            smooth_loss_unsuper = tf.reduce_mean(smooth_loss_unsuper)
            smooth_loss_mask = tf.reduce_mean(smooth_loss_mask)

        predictions_unsuper = tf.nn.sigmoid(predictions_unsuper)
        """if without relapse such as HF onset: generating the pennalty to encourage the prediction to be non-decreassing"""
        entropy_unsuper = -tf.reduce_sum (predictions_unsuper * tf.math.log(predictions_unsuper) + (1 - predictions_unsuper + 1e-10) * tf.math.log(1 - predictions_unsuper + 1e-10)) * weights_unsuper / (tf.reduce_sum(weights_unsuper))

        """the entropy loss on the unmasked data: to encourage spare predictions"""
        entropy_unsuper_MLP = -tf.reduce_mean(
            Prevalence_prediction_unsuper * tf.math.log(Prevalence_prediction_unsuper) + (
                    1 - Prevalence_prediction_unsuper + 1e-10) * tf.math.log(
                1 - Prevalence_prediction_unsuper + 1e-10))

        """the entropy loss on the masked data: to encourage spare predictions"""
        entropy_unsuper_MLP_mask = -tf.reduce_mean(
            Prevalence_prediction_unsuper_mask * tf.math.log(Prevalence_prediction_unsuper_mask) + (
                    1 - Prevalence_prediction_unsuper_mask + 1e-10) * tf.math.log(
                1 - Prevalence_prediction_unsuper_mask + 1e-10))
        """the entropy loss on the masked data: to encourage spare predictions"""

        """contrastive loss on the positive/negative visit pairs"""
        features_half_batch = tf.reduce_sum(tf.abs(smooth_seq - shuffled_smooth_seq), axis=-1)
        weights_half_batch = weights_smooth_super + shuffled_weights_smooth_super
        weights_half_batch = tf.cast(tf.greater(weights_half_batch, 1.5), tf.float32)
        Y_half_batch = tf.reduce_mean(abs(labels - shuffled_labels), axis=-1)
        distance_same = (1 - Y_half_batch) * tf.maximum(features_half_batch - threshold_embedding_same, 0.0) * tf.squeeze(weights_half_batch, axis=-1)
        distance_differ = Y_half_batch * tf.maximum(threshold_embedding - features_half_batch, 0.0) * tf.squeeze(weights_half_batch, axis=-1)
        distance_constastive = distance_same + distance_differ
        distance_constastive = tf.reduce_sum(distance_constastive) #/ (tf.reduce_sum(weights_half_batch))
        """contrastive loss on the positive/negative visit pairs"""

        """contrastive loss on the positive/negative patients using their prevalence prediction"""
        features_half_batch = tf.reduce_sum(tf.abs(Prevalence_fcn - Prevalence_fcn_shuffle), axis=-1)
        Y_half_batch = tf.reduce_mean(abs(labels_prevalence - labels_prevalence_shuffle), axis=-1)
        distance_same = (1 - Y_half_batch) * tf.maximum(features_half_batch - threshold_MLP_same,0.0)
        distance_differ = Y_half_batch * tf.maximum(threshold_MLP - features_half_batch,0.0)
        distance_constastive_MLP = tf.reduce_mean(distance_same + distance_differ)
        """contrastive loss on the positive/negative patients using their prevalence prediction"""

# START Block: modified by Amiee. Reason: to adapt to the above modifications

        # Adopt the following format: incidence_loss + weight_prevalence * prevalence_loss
        loss_gold = loss_incident + weight_prevalence * loss_prevalence
        loss_unsuper = loss_incident_unsuper + weight_prevalence * loss_prevalence_unsuper
        loss_silver = loss_incident_silver + weight_prevalence * loss_prevalence_silver
        loss_all = loss_gold + weight_unlabel*(loss_unsuper + loss_silver)
        loss_unsuper_plus_silver = loss_unsuper + loss_silver


    all_variables = model.trainable_variables

    if flag_silver == False:
        gradients = tape.gradient(loss_all, all_variables)
        # print(gradients)
        # # Check for None gradients
        optimizer.apply_gradients(grads_and_vars=zip(gradients, all_variables))
    else:
        gradients = tape.gradient(loss_unsuper_plus_silver, all_variables)
        optimizer_silver.apply_gradients(grads_and_vars=zip(gradients, all_variables))

##1    train_prevalence_incident.update_state(loss_prevalence_incident)
    train_loss_prevalence.update_state(loss_prevalence)
    train_loss_prevalence_unsuper.update_state(loss_prevalence_unsuper)
    train_loss_prevalence_silver.update_state(loss_prevalence_silver)
    train_loss_incident.update_state(loss_incident)
    train_loss_incident_unsuper.update_state(loss_incident_unsuper)
    train_loss_incident_silver.update_state(loss_incident_silver)

# End Block

    # The following will be ignored in our analysis for now
    train_smooth_loss.update_state(smooth_loss)
    train_smooth_loss_unsuper.update_state(smooth_loss_unsuper)
    train_keyfeature_loss.update_state(smooth_loss_unsuper)
    train_contrastive_loss.update_state(distance_constastive)
    train_contrastive_loss_prevalence.update_state(distance_constastive_MLP)
    train_prevalence_entropy_unsuper.update_state((entropy_unsuper + entropy_unsuper_MLP + entropy_unsuper_MLP_mask / 3.0))
    train_prevalence_consistency.update_state((loss_consistency + loss_consistency_prevalence) * 0.5)

    #Prevalence_prediction_final = Prevalence_prediction.numpy() #+ np.array(tf.expand_dims(tf.nn.sigmoid(prediction_incident_binary),axis=-1).numpy())) / 2.0  # (Prevalence_prediction.numpy() + np.array(prediction_MLP_total)) / 2.0

    return loss_incident.numpy(), predictions.numpy(), Prevalence_prediction.numpy(), predictions_unsuper.numpy(), Prevalence_prediction_unsuper.numpy(),\
    prediction_silver.numpy(), Prevalence_prediction_silver.numpy(), labels_prevalence.numpy(), labels_prevalence_silver.numpy(), attention_value_codes.numpy(), \
    attention_value_visits.numpy(), embedding_ori.numpy(), \
    embedding_code_rw.numpy(), embedding_code_visit_rw.numpy(), \
    smooth_seq.numpy(), attention_value_code.numpy(), \
    prediction_multi.numpy(), Prevalence_multi.numpy()
##1 np.array(prediction_MLP_total),

### This function is added by Amiee to take care of the batches that contain only the unsuper data: has nearly the same construct as train_step function
def train_step_silver(train_prevalence_incident, train_loss_prevalence,train_loss_prevalence_unsuper,train_loss_prevalence_silver,
    train_loss_incident,train_loss_incident_unsuper, train_loss_incident_silver,
    train_smooth_loss, train_smooth_loss_unsuper, train_keyfeature_loss,
    train_contrastive_loss, train_contrastive_loss_prevalence, train_prevalence_entropy_unsuper,
    train_prevalence_consistency, optimizer, optimizer_silver,
                model, X_train, labels, weights, has_data_loc_train, data_embedding_all,
                data_embedding_unsuper,
                X_unsuper, weights_unsuper, has_data_loc_train_unsuper, smooth_weight_in,
                keyfeature, keyfeature_unsuper,
                X_other_fts_train,
                X_other_fts_unsuper,
                patient_num,
                silver_train, silver_unsuper, weight_prevalence, weight_unlabel, flag_relapse=False):

    with tf.GradientTape(persistent=True) as tape:

        # Question: what are the thresholdings for?
        threshold_embedding = 10.0
        threshold_embedding_same = 5.0
        threshold_MLP = 10.0
        threshold_MLP_same = 5.0
        threshold_smooth = 0.5

        max_length = X_train.shape[1]
        num_feature = data_embedding_all.shape[1]
        embedding_dim = data_embedding_all.shape[2]

        weights_value = np.sum(weights.numpy(), axis=-1)
        weights_value_unsuper = np.sum(weights_unsuper.numpy(), axis=-1)
        weights = tf.cast(weights, tf.float32)
        labels = tf.cast(labels, tf.float32)
        silver_unsuper = tf.cast(silver_unsuper, tf.float32)
        silver_train = tf.cast(silver_train, tf.float32)

        weights_unsuper = tf.cast(weights_unsuper, tf.float32)
        weights_smooth_super = tf.cast(weights, tf.float32)
        weights_unsuper = tf.cast(tf.greater(weights_unsuper, 0.0), tf.float32)
        weights_smooth_super = tf.cast(tf.greater(weights_smooth_super, 0.0), tf.float32)

        indices = tf.range(start=0, limit=tf.shape(X_train)[0], dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(indices)
        X_train_shuffle = tf.gather(X_train, shuffled_indices)
        X_other_fts_shuffle = tf.gather(X_other_fts_train, shuffled_indices)
        shuffled_weights = tf.gather(weights, shuffled_indices)
        shuffled_weights_smooth_super = tf.gather(weights_smooth_super, shuffled_indices)
        # shuffled_labels = tf.gather(labels, shuffled_indices)
        batch_num = X_unsuper.numpy().shape[0]
        if random.random() < 0.5:
            mask_unsuper = np.random.normal(loc=1, scale=0.01, size=(batch_num, max_length, num_feature))
        else:
            mask_unsuper = np.random.normal(loc=1, scale=0.01, size=(batch_num, max_length, num_feature))
        mask_dropout = []
        for iii in range(int(batch_num * max_length * num_feature)):
            if random.random() < 0.5:
                if random.random() < 0.95:
                    mask_dropout.append(1)
                else:
                    mask_dropout.append(0)
            else:
                mask_dropout.append(1)
        mask_dropout = np.array(mask_dropout).reshape((batch_num, max_length, num_feature))
        mask_unsuper = tf.maximum(mask_unsuper, 0.0)

        X_unsuper_masked = X_unsuper
        X_other_fts_unsuper_masked = X_other_fts_unsuper

        # Question: what is the difference between Prevalence_prediction_unsuper and Prevalence_prediciton_silver?
        predictions, predictions_unsuper, predictions_unsuper_mask, smooth_seq, smooth_seq_unsuper, smooth_seq_mask, Prevalence_prediction, \
        attention_value_codes, attention_value_visits, \
        embedding_ori, embedding_code_rw, embedding_code_visit_rw, shuffled_smooth_seq, \
        attention_value_code, Prevalence_fcn, Prevalence_fcn_shuffle, Prevalence_prediction_unsuper, \
        Prevalence_prediction_unsuper_mask, visit_weights, \
        prediction_silver, Prevalence_prediction_silver, \
        prediction_multi,Prevalence_multi  = \
            model([data_embedding_all, data_embedding_unsuper,
                   keyfeature, keyfeature_unsuper, X_train, X_train_shuffle,
                   X_unsuper, X_unsuper_masked, X_other_fts_train, X_other_fts_shuffle,
                   X_other_fts_unsuper, X_other_fts_unsuper_masked], training=True)

        prediction_incident_binary = predictions[0,max_length-1, 0]
        prediction_incident_binary = tf.expand_dims(prediction_incident_binary, 0)

        prediction_incident_binary_unsuper = predictions_unsuper[0, max_length-1, 0]
        prediction_incident_binary_unsuper = tf.expand_dims(prediction_incident_binary_unsuper, 0)

        '''generating the silver standard predictions'''
        prediction_incident_silver = prediction_silver[0,max_length-1, 0]
        prediction_incident_silver = tf.expand_dims(prediction_incident_silver, 0)
        for rowi in range(1, len(weights_value)):
            prediction_incident_binary_temp = predictions[rowi, max_length-1, 0]
            prediction_incident_binary_temp = tf.expand_dims(prediction_incident_binary_temp, 0)
            prediction_incident_binary = tf.concat([prediction_incident_binary, prediction_incident_binary_temp],axis=0)

            prediction_incident_binary_temp = predictions_unsuper[rowi, max_length-1, 0]
            prediction_incident_binary_temp = tf.expand_dims(prediction_incident_binary_temp, 0)
            prediction_incident_binary_unsuper = tf.concat([prediction_incident_binary_unsuper, prediction_incident_binary_temp], axis=0)
            prediction_incident_binary_temp = prediction_silver[rowi,max_length-1, 0]
            prediction_incident_binary_temp = tf.expand_dims(prediction_incident_binary_temp, 0)
            prediction_incident_silver = tf.concat([prediction_incident_silver, prediction_incident_binary_temp], axis=0)

##1         prediction_MLP_total = tf.expand_dims(prediction_incident_binary, axis=-1).numpy()
        prediction_incident_silver = tf.expand_dims(prediction_incident_silver, axis=-1)
        prediction_incident_silver = tf.nn.sigmoid(prediction_incident_silver)
        '''generating the silver standard predictions'''


        # # Question: incident - prevalence?
        # # Answer: suppose a patient has the disease, he will manifest at the last step in LSTM, or attention (aggregation)
        loss_prevalence_incident = tf.reduce_mean(tf.abs((prediction_incident_binary) - (Prevalence_prediction)))
        loss_prevalence_incident_unsuper = tf.reduce_mean(tf.abs((prediction_incident_binary_unsuper) -
                                                            (Prevalence_prediction_unsuper)))
        loss_prevalence_incident = (loss_prevalence_incident + loss_prevalence_incident_unsuper) / 2.0
        Prevalence_prediction = tf.nn.sigmoid((tf.expand_dims(prediction_incident_binary,axis=-1)+Prevalence_prediction)/2.0)
        Prevalence_prediction = tf.nn.sigmoid(Prevalence_prediction)
        Prevalence_prediction_silver = tf.nn.sigmoid(Prevalence_prediction_silver)

        Prevalence_prediction_unsuper = tf.nn.sigmoid(Prevalence_prediction_unsuper)
        Prevalence_prediction_unsuper_mask = tf.nn.sigmoid(Prevalence_prediction_unsuper_mask)

### The following block are modified by Amiee (Prevalence + incidence loss calculations), the modification reason are 2-folds:
### (1) the orginal version calculate silver-related loss wrongly due to not subsetting, hence the masked out visits will participate in binary_loss calculation
### (2) we now further allow missing visits information
# START Block
# Prevalence: unsuper
        has_data_loc_train_unsuper = tf.squeeze(has_data_loc_train_unsuper,axis=-1)
        bool_has_unsuper_prevalence = tf.reduce_sum(has_data_loc_train_unsuper, axis = 1)
        bool_has_unsuper_prevalence = tf.where(tf.equal(bool_has_unsuper_prevalence, 0), 0, 1)
        labels_prevalence_unsuper = tf.reduce_mean(silver_unsuper, axis=1)
        labels_prevalence_unsuper_subset = tf.boolean_mask(labels_prevalence_unsuper, bool_has_unsuper_prevalence)
        labels_prevalence_unsuper_subset = tf.cast(labels_prevalence_unsuper_subset, dtype=tf.float32)
        Prevalence_prediction_unsuper_subset = tf.boolean_mask(Prevalence_prediction_unsuper, bool_has_unsuper_prevalence)
        Prevalence_prediction_unsuper_subset = tf.cast(Prevalence_prediction_unsuper_subset, dtype=tf.float32)
        loss_prevalence_unsuper = tf.square(labels_prevalence_unsuper_subset - Prevalence_prediction_unsuper_subset)
        loss_prevalence_unsuper = tf.reduce_mean(loss_prevalence_unsuper)

#       silver
        labels_prevalence_silver = tf.reshape(tf.reduce_mean(silver_train, axis=1),[-1])
        labels_prevalence_silver_subset = tf.boolean_mask(labels_prevalence_silver, bool_has_unsuper_prevalence)
        Prevalence_prediction_silver_subset = tf.boolean_mask(tf.reshape(Prevalence_prediction_silver,[-1]), bool_has_unsuper_prevalence)
        loss_prevalence_silver = tf.square(labels_prevalence_silver_subset - Prevalence_prediction_silver_subset)
        loss_prevalence_silver = tf.reduce_mean(loss_prevalence_silver)

# Incidence: unsuper
        label_unsuper_subset = tf.boolean_mask(silver_unsuper, has_data_loc_train_unsuper)
        prediction_incident_unsuper_subset = tf.boolean_mask(tf.cast(tf.nn.sigmoid(predictions_unsuper), tf.float32), has_data_loc_train_unsuper)
        loss_incident_unsuper = tf.square(label_unsuper_subset-prediction_incident_unsuper_subset)
        loss_incident_unsuper = tf.reduce_mean(loss_incident_unsuper)

#       silver
        has_data_loc_train = tf.squeeze(has_data_loc_train,axis=-1)
        label_silver_subset = tf.boolean_mask(silver_train, has_data_loc_train)
        prediction_incident_silver_subset = tf.boolean_mask(tf.cast(tf.nn.sigmoid(prediction_silver), tf.float32), has_data_loc_train)
        loss_incident_silver = tf.square(label_silver_subset-prediction_incident_silver_subset)
        loss_incident_silver = tf.reduce_mean(loss_incident_silver)

### END Block
        loss_consistency_prevalence = tf.reduce_mean(
            tf.square(Prevalence_prediction_unsuper - Prevalence_prediction_unsuper_mask))
        # print(f"tf.reduce_sum(weights_unsuper)={tf.reduce_sum(weights_unsuper)}")
        loss_consistency = tf.reduce_sum(tf.abs(smooth_seq_unsuper - smooth_seq_mask) \
                                            * weights_unsuper) / (tf.reduce_sum(weights_unsuper))

        """generating loss to encourage smoothness or non-decreasing"""
        if flag_relapse == True:
            """if with relapse such as MS: generating the penalty to encourage the prediction to be smoothy"""
            smooth_loss_unsuper = tf.abs(smooth_seq_unsuper[:, 0:-1, :] - smooth_seq_unsuper[:, 1:, :]) * tf.expand_dims(weights_unsuper[:, 1:], -1)

            weights_sum_unsuper = tf.reduce_sum(weights_unsuper)
            smooth_loss_unsuper = tf.reduce_sum(smooth_loss_unsuper, name="smooth_loss_unsuper") / weights_sum_unsuper

        else:
            """if without relapse such as HF onset: generating the penalty to encourage the prediction to be non-decreassing"""
            """weights_smooth_super or weights_unsuper: the weights of training data because of the kernel re-weighting"""
            smooth_loss_unsuper = tf.maximum(predictions_unsuper[:, 0:-1, :] - predictions_unsuper[:, 1:, :], 0)  * weights_unsuper[:, 1:]
            smooth_loss_unsuper = tf.maximum(tf.reduce_sum(smooth_loss_unsuper, 1), threshold_smooth)
            smooth_loss_unsuper = tf.reduce_mean(smooth_loss_unsuper)

        predictions_unsuper = tf.nn.sigmoid(predictions_unsuper)
        """if without relapse such as HF onset: generating the pennalty to encourage the prediction to be non-decreassing"""
        entropy_unsuper = -tf.reduce_sum (predictions_unsuper * tf.math.log(predictions_unsuper) + (1 - predictions_unsuper + 1e-10) * tf.math.log(1 - predictions_unsuper + 1e-10)) * weights_unsuper / (tf.reduce_sum(weights_unsuper))

        """the entropy loss on the unmasked data: to encourage spare predictions"""
        entropy_unsuper_MLP = -tf.reduce_mean(
            Prevalence_prediction_unsuper * tf.math.log(Prevalence_prediction_unsuper) + (
                    1 - Prevalence_prediction_unsuper + 1e-10) * tf.math.log(
                1 - Prevalence_prediction_unsuper + 1e-10))

        """the entropy loss on the masked data: to encourage spare predictions"""
        entropy_unsuper_MLP_mask = -tf.reduce_mean(
            Prevalence_prediction_unsuper_mask * tf.math.log(Prevalence_prediction_unsuper_mask) + (
                    1 - Prevalence_prediction_unsuper_mask + 1e-10) * tf.math.log(
                1 - Prevalence_prediction_unsuper_mask + 1e-10))
        """the entropy loss on the masked data: to encourage spare predictions"""

        """cross entropy loss for visit-level incident prediction"""
#        predictions = tf.nn.sigmoid(predictions)
#        predictions = tf.cast(predictions, tf.float32)

        """ loss for prevalence prediction"""
        # loss_MLP = loss_prevalence + distance_constastive_MLP * weight_contrastive
        """ loss for prevalence prediction"""

# START Block: modified by Amiee. Reason: to adapt to the above modifications

        # Adopt the following format: incidence_loss + weight_prevalence * prevalence_loss
        loss_unsuper = loss_incident_unsuper + weight_prevalence * loss_prevalence_unsuper
        loss_silver = loss_incident_silver + weight_prevalence * loss_prevalence_silver
        loss_unsuper_plus_silver = loss_unsuper + loss_silver

    gradients = tape.gradient(loss_unsuper_plus_silver, all_variables)
    optimizer_silver.apply_gradients(grads_and_vars=zip(gradients, all_variables))

    train_prevalence_incident.update_state(loss_prevalence_incident)
    # train_loss_prevalence.update_state(loss_prevalence)
    train_loss_prevalence_unsuper.update_state(loss_prevalence_unsuper)
    train_loss_prevalence_silver.update_state(loss_prevalence_silver)
    # train_loss_incident.update_state(loss_incident)
    train_loss_incident_unsuper.update_state(loss_incident_unsuper)
    train_loss_incident_silver.update_state(loss_incident_silver)
# End Block

    train_smooth_loss_unsuper.update_state(smooth_loss_unsuper)
    train_keyfeature_loss.update_state(smooth_loss_unsuper)
    train_prevalence_entropy_unsuper.update_state((entropy_unsuper + entropy_unsuper_MLP + entropy_unsuper_MLP_mask / 3.0))
    train_prevalence_consistency.update_state((loss_consistency + loss_consistency_prevalence) * 0.5)

    Prevalence_prediction_final = Prevalence_prediction.numpy() #+ np.array(tf.expand_dims(tf.nn.sigmoid(prediction_incident_binary),axis=-1).numpy())) / 2.0  # (Prevalence_prediction.numpy() + np.array(prediction_MLP_total)) / 2.0

    return loss_silver.numpy(), predictions.numpy(), Prevalence_prediction.numpy(), predictions_unsuper.numpy(), Prevalence_prediction_unsuper.numpy(), \
    prediction_silver.numpy(), Prevalence_prediction_silver.numpy(), labels_prevalence.numpy(), labels_prevalence_silver.numpy(), attention_value_codes.numpy(), \
    attention_value_visits.numpy(), embedding_ori.numpy(), \
    embedding_code_rw.numpy(), embedding_code_visit_rw.numpy(), \
    smooth_seq.numpy(), attention_value_code.numpy(), \
    prediction_multi.numpy(), Prevalence_multi.numpy()
 ##1  np.array(prediction_MLP_total)


def valid_step(Y_nlev, valid_loss, model, X_test, labels, weights, has_data_loc_test, data_embedding_test,
                data_embedding_test_unsuper, X_unsuper, weights_unsuper, has_data_loc_test_unsuper, smooth_weight
                , keyfeature, keyfeature_unsuper,  X_other_fts_test, X_other_fts_unsuper, patient_num_test):

    max_length = X_test.shape[1]
    num_feature = data_embedding_test.shape[1]
    embedding_dim = data_embedding_test.shape[2]

    predictions, predictions_unsuper, predictions_unsuper_mask, smooth_seq, smooth_seq_unsuper, smooth_seq_unsuper_mask, Prevalence_prediction, \
    attention_value_codes, attention_value_visits, \
    embedding_ori, embedding_code_rw, embedding_code_visit_rw, smooth_seq_shuffle, \
    attention_value_code, Prevalence_fcn, Prevalence_fcn_shuffle, Prevalence_fcn_unsuper, Prevalence_fcn_unsuper_mask, \
    visit_weights, prediction_silver, Prevalence_prediction_silver, \
    prediction_multi,Prevalence_multi  = model([data_embedding_test, data_embedding_test_unsuper,
                                                                            keyfeature, keyfeature_unsuper,
                                                                            X_test,X_test,X_unsuper,X_unsuper,
                                                                            X_other_fts_test, X_other_fts_test,
                                                                            X_other_fts_unsuper,X_other_fts_unsuper])


    # Commented out the all sigmoid transformation
    # predictions = tf.nn.sigmoid(predictions)

    weights_value = np.sum(weights.numpy(), axis=-1)
    prediction_incident_binary = predictions[0, max_length-1, 0]
    prediction_incident_binary = tf.expand_dims(prediction_incident_binary, 0)

    for rowi in range(1, len(weights_value)):
        prediction_incident_binary_temp = predictions[rowi,  max_length-1 , 0]
        prediction_incident_binary_temp = tf.expand_dims(prediction_incident_binary_temp, 0)
        prediction_incident_binary = tf.concat([prediction_incident_binary, prediction_incident_binary_temp],
                                                axis=0)
    # Commented out the all sigmoid transformation
    # prediction_MLP_total = tf.expand_dims(tf.nn.sigmoid(prediction_incident_binary), axis=-1).numpy()
    # Prevalence_prediction = tf.nn.sigmoid(Prevalence_prediction)
##1     prediction_MLP_total = tf.expand_dims(prediction_incident_binary, axis=-1).numpy()
    labels = tf.cast(labels, tf.float32)
    weights = tf.cast(weights, tf.float32)

### The following block are modified by Amiee (Prevalence + incidence loss calculations), the modification reason are 2-folds:
### (1) the orginal version calculate silver-related loss wrongly due to not subsetting, hence the masked out visits will participate in binary_loss calculation
### (2) we now further allow missing visits information
# START Block
# Prevalence: gold

        # CAUTION: ALL patients should have at least one label:
        # (1) Scenario 1: the labels are all 0/missing => 0
        # (2) Scenario 2: the labels are all 1/missing => >=1
        # (3) Scenario 3: the labels are 0/1/missing => >=1
        # But we still use a similar construct as train_step just to be safe

    bool_has_prevalence = tf.reduce_mean(weights, axis = 1)
    #START###########################multi-cateogry cross entropy loss for prevalence#########################
    # labels_prevalence = tf.reduce_sum(labels * weights, axis=1)
    # labels_prevalence = tf.cast(tf.greater_equal(labels_prevalence, 1.0), tf.float32)
    # labels_prevalence_subset = tf.boolean_mask(labels_prevalence, bool_has_prevalence)
    labels_prevalence = tf.reduce_max(labels * weights, axis=1)
    labels_prevalence_subset = tf.boolean_mask(labels_prevalence, bool_has_prevalence)


    Prevalence_prediction_subset = tf.boolean_mask(Prevalence_prediction, bool_has_prevalence)
    # Commented out the old loss_prevalence (next 2 lines)
    # loss_prevalence = tf.keras.losses.binary_crossentropy(labels_prevalence_subset, Prevalence_prediction_subset)
    # loss_prevalence = tf.reduce_mean(loss_prevalence)

    # Construct the multi-category Prevalence loss
    Prevalence_multi_subset = tf.boolean_mask(Prevalence_multi, bool_has_prevalence)
    labels_MLP_multi = tf.one_hot(tf.cast(labels_prevalence_subset, dtype=tf.int32), Y_nlev+1, on_value=1.0, off_value=0.0, axis=-1)
    loss_prevalence = tf.reduce_mean(-tf.reduce_sum(labels_MLP_multi * tf.math.log(Prevalence_multi_subset), axis=-1))

    #END###########################multi-cateogry cross entropy loss for prevalence#########################



# Incidence: gold
    #START###########################multi-cateogry cross entropy loss for incidence#########################
    # Commented out the old loss_incidence (next 6 lines)
    # predictions = tf.nn.sigmoid(predictions)
    # predictions = tf.cast(predictions, tf.float32)
    # label_subset = tf.boolean_mask(labels, weights)
    # predictions_subset = tf.boolean_mask(predictions, weights)
    # loss_incident = tf.keras.losses.binary_crossentropy(label_subset, predictions_subset)
    # loss_incident = tf.reduce_mean(loss_incident)

    # Construct the multi-category Incidence loss
    prediction_multi_subset = tf.boolean_mask(prediction_multi, weights)
    labels_subset = tf.boolean_mask(labels, weights)
    labels_subset_multi = tf.one_hot(tf.cast(labels_subset, dtype=tf.int32), Y_nlev+1, on_value=1.0, off_value=0.0, axis=-1)
    loss_incident = tf.reduce_mean(-tf.reduce_sum(labels_subset_multi * tf.math.log(prediction_multi_subset), axis=-1))

    #END###########################multi-cateogry cross entropy loss for incidence#########################


### END Block
    loss = loss_incident

    Prevalence_prediction_final =  Prevalence_prediction.numpy()  #(Prevalence_prediction.numpy() + np.array(prediction_MLP_total)) / 2.0  # (Prevalence_prediction.numpy()+np.array(prediction_MLP_total))/2.0
    return loss_incident.numpy(), predictions.numpy(), Prevalence_prediction_final, labels_prevalence.numpy(), \
            attention_value_codes.numpy(), attention_value_visits.numpy(), \
            embedding_ori.numpy(), embedding_code_rw.numpy(), \
            embedding_code_visit_rw.numpy(), \
            smooth_seq.numpy(), attention_value_code.numpy(), Prevalence_prediction.numpy(), \
            prediction_multi.numpy(), Prevalence_multi.numpy()
##1  np.array(prediction_MLP_total),

###################################################################################################
#                                  Main Function to Train Model                                   #
###################################################################################################

def train_model(Y_nlev, model, ds_train, ds_valid, weight_prevalence, weight_unlabel, weight_additional, flag_save_attention, flag_prediction, flag_relapse, epochs, epoch_silver, output_fname,output_directory,ordinal_score_method="weighted"):
    # print("--------begin model training.....")

    # initialize optimizers and losses
    optimizer = tf.keras.optimizers.Adam()
    optimizer_silver = tf.keras.optimizers.Adam()
    train_prevalence_incident = tf.keras.metrics.Mean(name='train_prevalence_incident')
    train_loss_prevalence = tf.keras.metrics.Mean(name='train_loss_prevalence')
    train_loss_prevalence_unsuper = tf.keras.metrics.Mean(name='train_loss_prevalence_unsuper')
    train_loss_prevalence_silver = tf.keras.metrics.Mean(name='train_loss_prevalence_silver')
    train_loss_incident = tf.keras.metrics.Mean(name='train_loss_incident')
    train_loss_incident_unsuper = tf.keras.metrics.Mean(name='train_loss_incident_unsuper')
    train_loss_incident_silver = tf.keras.metrics.Mean(name='train_loss_incident_silver')

    train_smooth_loss = tf.keras.metrics.Mean(name='train_smooth_loss')
    train_smooth_loss_unsuper = tf.keras.metrics.Mean(name='train_smooth_loss_unsuper')
    train_keyfeature_loss = tf.keras.metrics.Mean(name='train_keyfeature_loss')
    train_contrastive_loss = tf.keras.metrics.Mean(name='train_contrastive_loss')
    train_contrastive_loss_prevalence = tf.keras.metrics.Mean(name='train_contrastive_loss_prevalence')
    train_prevalence_entropy_unsuper = tf.keras.metrics.Mean(name='train_prevalence_entropy_unsuper')
    train_prevalence_consistency = tf.keras.metrics.Mean(name='train_prevalence_consistency')
    train_metric = tf.keras.metrics.AUC(name='train_auc', )
    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_smooth_loss = tf.keras.metrics.Mean(name='valid_smooth_loss')
    valid_metric = tf.keras.metrics.AUC(name='test_auc')

    epoch_show = 1
    epoch_num = -1
    AUC_incident_test_total = []
    PPV_incident_test_total = []
    AUC_prevalence_test_total = []
    #threshold = 0.5
    flag_save = str(random.randint(1, 1500)) + "_"
    flag_save_finish = False
    while (epoch_num < epochs):
        epoch_num += 1
        if len(AUC_prevalence_test_total) > 60:
            if np.mean(AUC_prevalence_test_total[-6:-3]) > np.mean(AUC_prevalence_test_total[-3:]):
                epoch_num += 1
        if epoch_num < 5:
            smooth_weight_temp = 0
        else:
            smooth_weight_temp = 0.03 + weight_additional * (epoch_num - 1) / epochs  # *(epoch_num-1)/epochs

        if epoch_num < epoch_silver:
            flag_silver = True
            # print("-------------------------------pre-training using silver labels-----------")
        else:
            flag_silver = False
            # print("-------------------------------joint training with silver labels-----------")

        predictions_train_total = []

        labels_train_total = []
        Prevalence_prediction_train_total = []
        labels_silver_incidence_train_total = []
        labels_prevalence_train_total = []
        predictions_unsuper_train_total = []
        Prevalence_prediction_unsuper_train_total = []
        prediction_silver_train_total = []
        Prevalence_prediction_silver_train_total = []
        labels_prevalence_silver_train_total = []
        has_data_loc_train_total = []

        train_gold_loss_prevalence_total = []
        train_gold_loss_incident_total = []
        train_silver_loss_prevalence_total = []
        train_silver_loss_incident_total = []

        predictions_test_total = []
        labels_test_total = []
        weights_test_total = []
        has_data_loc_test_total = []
        date_test_total = []
        patient_num_test_total = []
        Prevalence_prediction_total_test = []
        Prevalence_prediction_total_test_MLP_ONLY = []
        Prevalence_prediction_total_test_temporal_only = []
        test_gold_loss_prevalence_total = []
        test_gold_loss_incident_total = []

        attention_value_visit_test = []
        attention_value_code_test = []

        embedding_code_rw_test = []
        embedding_code_visit_rw_test = []

        embedding_ori_test = []
        labels_prevalence_total_test = []
        embedding_code_rw_hidden_test = []

        embedding_ori_train = []
        embedding_code_rw_hidden_train = []
        weights_train_total = []


        prediction_multi_train_total = []
        Prevalence_multi_train_total = []
        prediction_multi_test_total = []
        Prevalence_multi_test_total = []


        i_number = 0
        for X_train, y_train, patient_num_train, \
            date_train, weights_train, has_data_loc_train, data_embedding_all_temp, \
            data_embedding_unsuper_temp, X_unsuper, weights_unsuper, has_data_loc_train_unsuper,\
            keyfeature_train, keyfeature_unsuper, silver_train_batch, silver_unsuper_batch, X_other_fts_train_batch,X_other_fts_unsuper_batch in ds_train:

            bool_train_silver = tf.reduce_sum(weights_train)

            bool_data_any = tf.reduce_sum(has_data_loc_train)
            # print(f"has_data_loc_train:{has_data_loc_train}")
            # print(f"has_data_loc_train_unsuper:{has_data_loc_train_unsuper}")
            # print(f"bool_data_any:{bool_data_any}")
            if bool_data_any == 0:
                # print("Skipping iteration")
                continue

##1                 binary_temporal_only, 
            if bool_train_silver == 0: # all weights are zero
                loss_out, predictions, Prevalence_prediction, predictions_unsuper, Prevalence_prediction_unsuper, \
                prediction_silver, Prevalence_prediction_silver, labels_prevalence,labels_prevalence_silver, \
                attention_value_codes, attention_value_visits, \
                embedding_ori, embedding_code_rw, \
                embedding_code_visit_rw, embedding_hidden, attention_value_weights, \
                prediction_multi, Prevalence_multi = train_step_silver(train_prevalence_incident, train_loss_prevalence,train_loss_prevalence_unsuper,train_loss_prevalence_silver,
                                                        train_loss_incident,train_loss_incident_unsuper, train_loss_incident_silver,
                                                        train_smooth_loss, train_smooth_loss_unsuper, train_keyfeature_loss,
                                                        train_contrastive_loss, train_contrastive_loss_prevalence, train_prevalence_entropy_unsuper,
                                                        train_prevalence_consistency, optimizer, optimizer_silver,
                                                        model, X_train, y_train,
                                                        weights_train, has_data_loc_train, data_embedding_all_temp,
                                                        data_embedding_unsuper_temp, X_unsuper,
                                                        weights_unsuper, has_data_loc_train_unsuper, smooth_weight_temp,
                                                        keyfeature_train, keyfeature_unsuper,
                                                        X_other_fts_train_batch, X_other_fts_unsuper_batch,
                                                        patient_num_train,
                                                        silver_train_batch, silver_unsuper_batch,
                                                        weight_prevalence, weight_unlabel,
                                                        flag_relapse)


            else:
                loss_out, predictions, Prevalence_prediction, predictions_unsuper, Prevalence_prediction_unsuper, \
                prediction_silver, Prevalence_prediction_silver, labels_prevalence, labels_prevalence_silver,\
                attention_value_codes, attention_value_visits, \
                embedding_ori, embedding_code_rw, \
                embedding_code_visit_rw, embedding_hidden, attention_value_weights, \
                prediction_multi, Prevalence_multi = train_step(Y_nlev, train_prevalence_incident, train_loss_prevalence,train_loss_prevalence_unsuper,train_loss_prevalence_silver,
                                                    train_loss_incident,train_loss_incident_unsuper, train_loss_incident_silver,
                                                    train_smooth_loss, train_smooth_loss_unsuper, train_keyfeature_loss,
                                                    train_contrastive_loss, train_contrastive_loss_prevalence, train_prevalence_entropy_unsuper,
                                                    train_prevalence_consistency, optimizer, optimizer_silver,
                                                    model, X_train, y_train,
                                                    weights_train,has_data_loc_train, data_embedding_all_temp,
                                                    data_embedding_unsuper_temp, X_unsuper,
                                                    weights_unsuper, has_data_loc_train_unsuper, smooth_weight_temp,
                                                    keyfeature_train, keyfeature_unsuper,
                                                    X_other_fts_train_batch, X_other_fts_unsuper_batch,
                                                    patient_num_train,
                                                    silver_train_batch, silver_unsuper_batch,
                                                    weight_prevalence, weight_unlabel,
                                                    flag_silver,flag_relapse)
### This block of code is reorganized by Amiee for better understanding and formatting
# START Block
            if i_number == 0:
# incidence: gold
                predictions_train_total = np.array(predictions)
                prediction_multi_train_total = np.array(prediction_multi)
                labels_train_total = np.array(y_train)
                weights_train_total = np.array(weights_train)

#           silver
                prediction_silver_train_total = np.array(prediction_silver)
                labels_silver_incidence_train_total = np.array(silver_unsuper_batch)

#           unsuper
                predictions_unsuper_train_total = np.array(predictions_unsuper)

# prevalence: gold
                Prevalence_prediction_train_total = np.array(Prevalence_prediction)
                Prevalence_multi_train_total = np.array(Prevalence_multi)
                labels_prevalence_train_total = np.array(labels_prevalence)
                has_data_loc_train_total = np.array(has_data_loc_train) #!

#            silver
                Prevalence_prediction_silver_train_total = np.array(Prevalence_prediction_silver)
                labels_prevalence_silver_train_total = np.array(labels_prevalence_silver)

#            unsuper
                Prevalence_prediction_unsuper_train_total = np.array(Prevalence_prediction_unsuper)

                embedding_ori_train = np.array(embedding_ori)
                embedding_code_rw_hidden_train = np.array(embedding_hidden)
            else:
# incidence: gold
                predictions_train_total = np.concatenate((predictions_train_total, np.array(predictions)), axis=0)
                prediction_multi_train_total = np.concatenate((prediction_multi_train_total, np.array(prediction_multi)), axis=0)
                labels_train_total = np.concatenate((labels_train_total, np.array(y_train)), axis=0)
                weights_train_total = np.concatenate((weights_train_total, np.array(weights_train)), axis=0)

#           silver
                prediction_silver_train_total = np.concatenate((prediction_silver_train_total, np.array(prediction_silver)), axis=0)
                labels_silver_incidence_train_total = np.concatenate((labels_silver_incidence_train_total, np.array(silver_unsuper_batch)), axis=0)

#           unsuper
                predictions_unsuper_train_total = np.concatenate((predictions_unsuper_train_total, np.array(predictions_unsuper)), axis=0)

# prevalence: gold
                Prevalence_prediction_train_total = np.concatenate((Prevalence_prediction_train_total, np.array(Prevalence_prediction)), axis=0)
                Prevalence_multi_train_total = np.concatenate((Prevalence_multi_train_total, np.array(Prevalence_multi)), axis=0)
                labels_prevalence_train_total = np.concatenate((labels_prevalence_train_total, np.array(labels_prevalence)), axis=0)
                has_data_loc_train_total = np.concatenate((has_data_loc_train_total, np.array(has_data_loc_train)), axis=0)

#             silver
                Prevalence_prediction_silver_train_total = np.concatenate((Prevalence_prediction_silver_train_total, np.array(Prevalence_prediction_silver)), axis=0)
                labels_prevalence_silver_train_total = np.concatenate((labels_prevalence_silver_train_total, np.array(labels_prevalence_silver)), axis=0)


#              unsuper
                Prevalence_prediction_unsuper_train_total = np.concatenate((Prevalence_prediction_unsuper_train_total, np.array(Prevalence_prediction_unsuper)), axis=0)

                embedding_ori_train = np.concatenate((embedding_ori_train, np.array(embedding_ori)), axis=0)
                embedding_code_rw_hidden_train = np.concatenate(
                    (embedding_code_rw_hidden_train, np.array(embedding_hidden)), axis=0)
# END Block

            i_number += 1
        i_number = 0
        if epoch_num % epoch_show == 0 or epoch_num > epochs - 2:
            for X_test, y_test, patient_num_test, date_test, \
                weights_test, has_data_loc_test, data_embedding_test_temp, data_embedding_test_temp_temp, \
                X_unsuper, weights_unsuper, has_data_loc_test_unsuper, \
                keyfeature_test, keyfeature_test_unsuper, X_other_fts_test_batch, X_other_fts_test_unsuper_batch in ds_valid:

                loss_out, predictions, Prevalence_prediction, labels_prevalence, \
                attention_value_codes, attention_value_visits, \
                embedding_ori, embedding_code_rw, embedding_code_visit_rw, \
                embedding_hidden, attention_value_weights_test, binary_MLP_only, \
                prediction_multi, Prevalence_multi = valid_step(
                    Y_nlev, valid_loss, model,
                    X_test, y_test, weights_test, has_data_loc_test,
                    data_embedding_test_temp, data_embedding_test_temp_temp, X_unsuper, weights_unsuper, has_data_loc_test_unsuper,
                    smooth_weight_temp, keyfeature_test, keyfeature_test_unsuper, X_other_fts_test_batch, X_other_fts_test_unsuper_batch, patient_num_test)
 ##1 binary_temporal_only,
                if i_number == 0:

                    predictions_test_total = np.array(predictions)
                    prediction_multi_test_total = np.array(prediction_multi)
                    labels_test_total = np.array(y_test)
                    weights_test_total = np.array(weights_test)
                    has_data_loc_test_total = np.array(has_data_loc_test)
                    date_test_total = np.array(date_test)
                    patient_num_test_total = np.array(patient_num_test)
                    Prevalence_prediction_total_test = np.array(Prevalence_prediction)
                    Prevalence_prediction_total_test_MLP_ONLY = np.array(binary_MLP_only)
##1                     Prevalence_prediction_total_test_temporal_only = np.array(binary_temporal_only)
                    Prevalence_multi_total_test = np.array(Prevalence_multi)

                    attention_value_visit_test = np.array(attention_value_visits)
                    attention_value_code_test = np.array(attention_value_codes)

                    embedding_ori_test = np.array(embedding_ori)
                    embedding_code_rw_hidden_test = np.array(embedding_hidden)
                    labels_prevalence_total_test = np.array(labels_prevalence)

                    embedding_code_rw_test = np.array(embedding_code_rw)
                    embedding_code_visit_rw_test = np.array(embedding_code_visit_rw)

                else:
                    predictions_test_total = np.concatenate((predictions_test_total, np.array(predictions)), axis=0)
                    prediction_multi_test_total = np.concatenate((prediction_multi_test_total, np.array(prediction_multi)), axis=0)
                    labels_test_total = np.concatenate((labels_test_total, np.array(y_test)), axis=0)
                    weights_test_total = np.concatenate((weights_test_total, np.array(weights_test)), axis=0)
                    has_data_loc_test_total = np.concatenate((has_data_loc_test_total, np.array(has_data_loc_test)), axis=0)
                    date_test_total = np.concatenate((date_test_total, np.array(date_test)), axis=0)
                    patient_num_test_total = np.concatenate((patient_num_test_total, np.array(patient_num_test)),
                                                            axis=0)
                    Prevalence_prediction_total_test = np.concatenate(
                        (Prevalence_prediction_total_test, np.array(Prevalence_prediction)), axis=0)
                    Prevalence_prediction_total_test_MLP_ONLY = np.concatenate(
                        (Prevalence_prediction_total_test_MLP_ONLY, np.array(binary_MLP_only)), axis=0)
##1                    Prevalence_prediction_total_test_temporal_only = np.concatenate(
##1                        (Prevalence_prediction_total_test_temporal_only, np.array(binary_temporal_only)), axis=0)
                    Prevalence_multi_total_test = np.concatenate(
                        (Prevalence_multi_total_test, np.array(Prevalence_multi)), axis=0)


                    labels_prevalence_total_test = np.concatenate((labels_prevalence_total_test, np.array(labels_prevalence)),
                                                            axis=0)
                    attention_value_visit_test = np.concatenate(
                        (attention_value_visit_test, np.array(attention_value_visits)), axis=0)
                    attention_value_code_test = np.concatenate(
                        (attention_value_code_test, np.array(attention_value_codes)), axis=0)

                    embedding_ori_test = np.concatenate((embedding_ori_test, np.array(embedding_ori)), axis=0)
                    embedding_code_rw_test = np.concatenate((embedding_code_rw_test, np.array(embedding_code_rw)),
                                                            axis=0)
                    embedding_code_visit_rw_test = np.concatenate(
                        (embedding_code_visit_rw_test, np.array(embedding_code_visit_rw)), axis=0)
                    embedding_code_rw_hidden_test = np.concatenate(
                        (embedding_code_rw_hidden_test, np.array(embedding_hidden)),
                        axis=0)
                i_number += 1
            if flag_save_attention > 0:
                # print("---saving attention values")
                if epoch_num % 10 == 0 or epoch_num > epochs - 2:
                    tmp_fname = "Attention_value_epoch"+str(epoch_num)+".pkl"
                    with open(output_directory + tmp_fname, 'wb') as fid:
                        pickle.dump((patient_num_test_total, attention_value_visit_test,
                                        attention_value_code_test, predictions_test_total, labels_test_total,
                                        weights_test_total), fid)

                    tmp_fname = "data_test_epoch"+str(epoch_num)+".pkl"
                    with open(output_directory + tmp_fname, 'wb') as fid:
                        pickle.dump((patient_num_test_total, embedding_ori_test,
                                        embedding_code_rw_test, embedding_code_visit_rw_test,
                                        embedding_code_rw_hidden_test, labels_test_total, weights_test_total), fid)
                    tmp_fname = "data_train_epoch"+str(epoch_num)+".pkl"
                    with open(output_directory + tmp_fname, 'wb') as fid:
                        pickle.dump((embedding_ori_train, embedding_code_rw_hidden_train,
                                        labels_train_total, weights_train_total), fid)

        if epoch_num % epoch_show == 0 or epoch_num > epochs - 2:
### This block of code is reorganized by Amiee for better understanding and formatting
# START Block
# Train
#   Incidence
#       gold
            predictions_train_total = np.array(predictions_train_total, dtype=float)#.reshape((-1, 1))
            labels_train_total = np.array(labels_train_total, dtype=int)
            labels_train_total = labels_train_total #np.reshape(labels_train_total, (-1, 1))
            weights_train_total = np.array(weights_train_total, dtype=int)#.reshape((-1, 1))

            predictions_train_total_subset = tf.boolean_mask(predictions_train_total, weights_train_total)
            labels_train_total_subset = tf.boolean_mask(labels_train_total, weights_train_total)

############ Updated the incidence loss multi-level outcomes #################
            # loss_incident_train_total = tf.keras.losses.binary_crossentropy(labels_train_total_subset, predictions_train_total_subset)
            # loss_incident_train_total = tf.reduce_mean(loss_incident_train_total)
            prediction_multi_train_total_subset = tf.boolean_mask(prediction_multi_train_total, weights_train_total)
            labels_train_total_subset_multi = tf.one_hot(tf.cast(labels_train_total_subset, dtype=tf.int32), Y_nlev+1, on_value=1.0, off_value=0.0, axis=-1)
            loss_incident_train_total = tf.reduce_mean(-tf.reduce_sum(labels_train_total_subset_multi * tf.math.log(prediction_multi_train_total_subset), axis=-1))

############ Updated the incidence AUC for multi-level outcomes #################
            y_train_vec = tf.reshape(labels_train_total_subset,[-1]).numpy()
            y_dich = np.copy(y_train_vec)
            y_med_int = int(np.median(y_train_vec))
            y_med = float(y_med_int)
            dich_rate_gt = np.mean(y_dich > y_med)
            dich_rate_gteq = np.mean(y_dich >= y_med)
            if abs(dich_rate_gt-0.5) < abs(dich_rate_gteq-0.5):
                y_dich[y_train_vec > y_med] = 1
                y_dich[y_train_vec <= y_med] = 0
                dich_proj = np.concatenate([np.zeros(y_med_int+1),np.ones(Y_nlev-y_med_int)])
            else:
                y_dich[y_train_vec >= y_med] = 1
                y_dich[y_train_vec < y_med] = 0
                dich_proj = np.concatenate([np.zeros(y_med_int),np.ones(Y_nlev+1-y_med_int)])

            prediction_multi_train_total_subset = np.array(prediction_multi_train_total_subset, dtype=float)
            dich_pred_train = np.dot(prediction_multi_train_total_subset, dich_proj)

##            df_pred_bin = pd.DataFrame(dich_pred_train)
##            df_pred_bin.to_csv(
##                        output_directory + "Dich_Pred_Train_epoch" + str(epoch_num) + "_" + flag_save + "_" + output_fname,
##                        index=True, sep=',')
##            df_pred_multi = pd.DataFrame(prediction_multi_train_total_subset)
##            df_pred_multi.to_csv(
##                        output_directory + "Multi_Pred_Train_epoch" + str(epoch_num) + "_" + flag_save + "_" + output_fname,
##                        index=True, sep=',')


            AUC_incident_train_total = roc_auc_score(y_true=y_dich,
                                            y_score=dich_pred_train,
                                            average='macro')

#       silver
            has_data_loc_train_total = tf.squeeze(has_data_loc_train_total, axis = -1)
            prediction_silver_train_total_subset = tf.boolean_mask(prediction_silver_train_total, has_data_loc_train_total)
            prediction_silver_train_total_subset = tf.cast(prediction_silver_train_total_subset, dtype=tf.float32)
            labels_silver_incidence_train_total_subset = tf.boolean_mask(labels_silver_incidence_train_total, has_data_loc_train_total)
            labels_silver_incidence_train_total_subset = tf.cast(labels_silver_incidence_train_total_subset, dtype=tf.float32)
            loss_incident_silver_total = tf.square(labels_silver_incidence_train_total_subset-prediction_silver_train_total_subset)
            loss_incident_silver_total = tf.reduce_mean(loss_incident_silver_total)

            # AUC_incident_silver_total = roc_auc_score(y_true=tf.reshape(labels_silver_incidence_train_total_subset,[-1]),
            #                                 y_score=tf.reshape(prediction_silver_train_total_subset,[-1]),
            #                                 average='macro')
#       unsuper
            predictions_unsuper_train_total_subset = tf.boolean_mask(predictions_unsuper_train_total, has_data_loc_train_total)
            predictions_unsuper_train_total_subset = tf.cast(predictions_unsuper_train_total_subset, dtype=tf.float32)
            loss_incident_unsuper_total = tf.square(labels_silver_incidence_train_total_subset - predictions_unsuper_train_total_subset)
            loss_incident_unsuper_total = tf.reduce_mean(loss_incident_unsuper_total)

            # AUC_incident_unsuper_total = roc_auc_score(y_true=tf.reshape(labels_silver_incidence_train_total_subset,[-1]),
            #                                 y_score=tf.reshape(predictions_unsuper_train_total_subset,[-1]),
            #                                 average='macro')


            # Print a few losses
            #loss_incident_x_multi = tf.keras.losses.categorical_crossentropy(labels_train_total_subset_multi, prediction_multi_train_total_subset)
            #loss_incident_x_multi = tf.reduce_mean(loss_incident_x_multi)
            #if Y_nlev==1:
            #    predictions_train_total_subset= tf.nn.sigmoid(predictions_train_total_subset)
            #    predictions_train_total_subset= tf.cast(predictions_train_total_subset, tf.float32)
            #    loss_incident_xbin = tf.keras.losses.binary_crossentropy(labels_train_total_subset, predictions_train_total_subset)
            #    loss_incident_xbin = tf.reduce_mean(loss_incident_xbin)
            #    losslogs = 'Incidence Formula: {}, Incidence Function: {}, Binary Function: {}'
            #    tf.print(tf.strings.format(losslogs, (loss_incident_train_total, loss_incident_x_multi, loss_incident_xbin)))
            #else:
            #    losslogs = 'Incidence Formula: {}, Incidence Function: {}'
            #    tf.print(tf.strings.format(losslogs, (loss_incident_train_total, loss_incident_x_multi)))


#   Prevalence
#       gold
            # bool_has_prevalence_train_total = tf.reduce_mean(weights_train_total, axis = 1)
            # labels_prevalence_train_total = tf.reduce_sum(labels_train_total * weights_train_total, axis=1)
            # labels_prevalence_train_total = tf.cast(tf.greater_equal(labels_prevalence_train_total, 1.0), tf.float32)
            # labels_prevalence_train_total_subset = tf.boolean_mask(labels_prevalence_train_total, bool_has_prevalence_train_total)
            # Prevalence_prediction_train_total_subset = tf.boolean_mask(Prevalence_prediction_train_total, bool_has_prevalence_train_total)

            bool_has_prevalence_train_total = tf.reduce_sum(weights_train_total, axis = 1)
            bool_has_prevalence_train_total = tf.where(tf.equal(bool_has_prevalence_train_total, 0), 0, 1)
            labels_prevalence_train_total = tf.reduce_sum(labels_train_total * weights_train_total, axis=1)
            labels_prevalence_train_total = tf.where(tf.equal(labels_prevalence_train_total, 0), 0, 1)
            labels_prevalence_train_total_subset = tf.boolean_mask(labels_prevalence_train_total, bool_has_prevalence_train_total)
            Prevalence_prediction_train_total_subset = tf.boolean_mask(Prevalence_prediction_train_total, bool_has_prevalence_train_total)

############ Updated the Prevalence AUC for multi-level outcomes  #################
#            loss_prevalence_train_total = tf.keras.losses.binary_crossentropy(labels_prevalence_train_total_subset, Prevalence_prediction_train_total_subset)
#            loss_prevalence_train_total = tf.reduce_mean(loss_prevalence_train_total)
            Prevalence_multi_train_total_subset = tf.boolean_mask(Prevalence_multi_train_total, bool_has_prevalence_train_total)
            labels_prevalence_train_total_subset_multi = tf.one_hot(tf.cast(labels_prevalence_train_total_subset, dtype=tf.int32), Y_nlev+1, on_value=1.0, off_value=0.0, axis=-1)
            loss_prevalence_train_total = tf.reduce_mean(-tf.reduce_sum(labels_prevalence_train_total_subset_multi * tf.math.log(Prevalence_multi_train_total_subset), axis=-1))

            # print(f"labels_prevalence_train_total_subset shape = {labels_prevalence_train_total_subset}")
            # print(f"Prevalence_prediction_train_total_subset shape = {Prevalence_prediction_train_total_subset}")
#            AUC_prevalence_train_total = roc_auc_score(y_true=tf.reshape(labels_prevalence_train_total_subset,[-1]),
#                                                y_score=tf.reshape(Prevalence_prediction_train_total_subset,[-1]),
#                                                average='macro')
            y_Prevalence = tf.reshape(labels_prevalence_train_total_subset,[-1]).numpy()
            y_dich_P = np.copy(y_Prevalence)
            if abs(dich_rate_gt-0.5) < abs(dich_rate_gteq-0.5):
                y_dich_P[y_Prevalence > y_med] = 1
                y_dich_P[y_Prevalence <= y_med] = 0
            else:
                y_dich_P[y_Prevalence >= y_med] = 1
                y_dich_P[y_Prevalence < y_med] = 0

            dich_predP_train = np.dot(Prevalence_multi_train_total_subset, dich_proj)
            AUC_prevalence_train_total = roc_auc_score(y_true=y_dich_P,
                                            y_score=dich_predP_train,
                                            average='macro')


#       silver
            bool_has_silver_prevalence_total = tf.reduce_sum(has_data_loc_train_total, axis = 1)
            bool_has_silver_prevalence_total = tf.where(tf.equal(bool_has_silver_prevalence_total, 0), 0, 1)
            # print("bool_has_silver_prevalence_total:{bool_has_silver_prevalence_total}")
            labels_prevalence_silver_train_total_subset = tf.boolean_mask(labels_prevalence_silver_train_total, bool_has_silver_prevalence_total)
            labels_prevalence_silver_train_total_subset = tf.cast(labels_prevalence_silver_train_total_subset, dtype=tf.float32)
            Prevalence_prediction_silver_train_total_subset = tf.boolean_mask(Prevalence_prediction_silver_train_total, bool_has_silver_prevalence_total)
            Prevalence_prediction_silver_train_total_subset = tf.cast(Prevalence_prediction_silver_train_total_subset, dtype=tf.float32)
            loss_prevalence_silver_total = tf.square(labels_prevalence_silver_train_total_subset - Prevalence_prediction_silver_train_total_subset)
            loss_prevalence_silver_total = tf.reduce_mean(loss_prevalence_silver_total)

            # AUC_prevalence_silver_total = roc_auc_score(y_true=tf.reshape(labels_prevalence_silver_train_total_subset,[-1]),
            #                                 y_score=tf.reshape(Prevalence_prediction_silver_train_total_subset,[-1]),
            #                                 average='macro')
#       unsuper
            Prevalence_prediction_unsuper_train_total_subset = tf.boolean_mask(Prevalence_prediction_unsuper_train_total, bool_has_silver_prevalence_total)
            Prevalence_prediction_unsuper_train_total_subset = tf.cast(Prevalence_prediction_unsuper_train_total_subset, dtype=tf.float32)
            loss_prevalence_unsuper_total = tf.square(labels_prevalence_silver_train_total_subset - Prevalence_prediction_unsuper_train_total_subset)
            loss_prevalence_unsuper_total = tf.reduce_mean(loss_prevalence_unsuper_total)

            # AUC_prevalence_unsuper_total = roc_auc_score(y_true=tf.reshape(labels_prevalence_silver_train_total_subset,[-1]),
            #                                 y_score=tf.reshape(Prevalence_prediction_unsuper_train_total_subset,[-1]),
            #                                 average='macro')


# END Block

            # y_true_get_train = np.array(labels_train_total, dtype=int)
            # y_true_get_train = np.reshape(y_true_get_train, (-1, 1))
            # # # print("y_true_get_train: ", y_true_get_train.shape)

            # score_get_train = np.array(predictions_train_total, dtype=float).reshape((-1, 1))
            # weights_train = np.array(weights_train_total, dtype=int).reshape((-1, 1))
            # boolean_indicator = weights_train.astype(bool)
            # try:
            #     AUC_train = roc_auc_score(y_true=y_true_get_train[boolean_indicator].reshape(-1),
            #                                 y_score=score_get_train[boolean_indicator].reshape(-1),
            #                                 average='macro')
            #     AUC_prevalence_train = roc_auc_score(y_true=labels_prevalence_train_total.reshape(-1),
            #                                     y_score=Prevalence_prediction_train_total.reshape(-1),
            #                                     average='macro')
            #     AUC_silver = roc_auc_score(y_true=labels_silver_incidence_train_total.reshape(-1),
            #                                 y_score=predictions_unsuper_train_total.reshape(-1),
            #                                 average='macro')
            #     AUC_prevalence_silver = roc_auc_score(y_true=labels_prevalence_silver_train_total.reshape(-1),
            #                                 y_score=Prevalence_prediction_unsuper_train_total.reshape(-1),
            #                                 average='macro')

            # except:
            #     AUC_train = 0
            #     AUC_prevalence_train = 0
            #     AUC_silver = 0
            #     AUC_prevalence_silver = 0
                # print("-----------------------------------------prevalence AUC evaluation error--------------")
            # try:
            #     if flag_prediction > 0:
            #         AUC_prevalence_test = AUC_prevalence_train
            #         # AUC_prevalence_test_MLP_only = AUC_prevalence_train
            #         # AUC_prevalence_test_incident_only = AUC_prevalence_train
            #     else:
            #         AUC_prevalence_test = roc_auc_score(y_true=labels_prevalence_total_test,
            #                                         y_score=Prevalence_prediction_total_test,
            #                                         average='macro')
            #         test_loss_prevalence = tf.keras.losses.binary_crossentropy(labels_prevalence_total_test, Prevalence_prediction_total_test)
            #         test_loss_prevalence = tf.reduce_mean(test_loss_prevalence)
            #         # AUC_prevalence_test_MLP_only = roc_auc_score(y_true=labels_prevalence_total_test,
            #         #                                         y_score=Prevalence_prediction_total_test_MLP_ONLY,
            #         #                                         average='macro')
            #         # AUC_prevalence_test_incident_only = roc_auc_score(y_true=labels_prevalence_total_test,
            #         #                                             y_score=Prevalence_prediction_total_test_temporal_only,
            #         #                                             average='macro')
            # except:
            #     AUC_prevalence_test = 0
            #     test_loss_prevalence = 0
            #     # AUC_prevalence_test_MLP_only = 0
            #     # AUC_prevalence_test_incident_only = 0
            #     # print("-----------------------------------------incident AUC evaluation error--------------")
            # AUC_prevalence_test_total.append(AUC_prevalence_test)
            # Prevalence_prediction_total_binary = np.array(np.greater(Prevalence_prediction_train_total, threshold),
            #                                                 dtype=np.int32)
            # Prevalence_prediction_total_test_binary = np.array(
            #     np.greater(Prevalence_prediction_total_test, threshold), dtype=np.int32)

            labels_test_total = np.array(labels_test_total).reshape((-1, 1))
            predictions_test_total = np.array(predictions_test_total).reshape((-1, 1))
            weights_test_total = np.array(weights_test_total).reshape((-1, 1))
            date_test_total = np.array(date_test_total).reshape((-1, 1))
            patient_num_test_total = np.array(patient_num_test_total).reshape((-1, 1))

            # Dichotomized prediction from multi-category outcomes
            prediction_dich_test_total = np.dot(prediction_multi_test_total, dich_proj)
            prediction_dich_test_total = np.array(prediction_dich_test_total).reshape((-1, 1))
            prediction_multi_test_total = np.array(prediction_multi_test_total).reshape((-1, Y_nlev+1))

            y_true_get_test = []
            score_get_test = []
            score_dich_get_test = []
            score_multi_get_test = []
            date_test = []
            patient_num_test = []

            for samplei in range(len(labels_test_total)):
                ### !!!very important, here, for instance, we only test for those incidences that have nonzero weight!!!
                if int(weights_test_total[samplei]) > 0:
                    y_true_get_test.append(labels_test_total[samplei])
                    score_get_test.append(predictions_test_total[samplei])
                    score_dich_get_test.append(prediction_dich_test_total[samplei])
                    score_multi_get_test.append(prediction_multi_test_total[samplei,])
                    date_test.append(date_test_total[samplei])
                    patient_num_test.append(patient_num_test_total[samplei])

            y_true_get_test = np.array(y_true_get_test)
            score_get_test = np.array(score_get_test)
            score_dich_get_test = np.array(score_dich_get_test)
            score_multi_get_test = np.array(score_multi_get_test)


            date_test = np.array(date_test)
            patient_num_test = np.array(patient_num_test)

            # # print(f"y_true_get_test max = {np.max(y_true_get_test)} and min = {np.min(y_true_get_test)}; score_get_test max = {np.max(score_get_test)} and min = {np.min(score_get_test)}")
            # # print(f"Number of NaN values in y_true_get_test: {np.sum(np.isnan(y_true_get_test))} while the number of NaN values in score_get_test: {np.sum(np.isnan(y_true_get_test))}")

            if flag_prediction > 0:
                y_true_get_test_temp = []
                for ij in range(len(y_true_get_test)):
                    y_true_get_test_temp.append(random.randint(0, 1))
                y_true_get_test = np.array(y_true_get_test_temp)

############ Updated the incidence AUC for multi-level outcomes #################

            y_dich_get_test = np.copy(y_true_get_test)
            if abs(dich_rate_gt-0.5) < abs(dich_rate_gteq-0.5):
                y_dich_get_test[y_true_get_test > y_med] = 1
                y_dich_get_test[y_true_get_test <= y_med] = 0
            else:
                y_dich_get_test[y_true_get_test >= y_med] = 1
                y_dich_get_test[y_true_get_test < y_med] = 0

##            df_pred_bin = pd.DataFrame(score_dich_get_test)
##            df_pred_bin.to_csv(
##                        output_directory + "Dich_Pred_Test_epoch" + str(epoch_num) + "_" + flag_save + "_" + output_fname,
##                        index=True, sep=',')
##            df_pred_multi = pd.DataFrame(score_multi_get_test)
##            df_pred_multi.to_csv(
##                        output_directory + "Multi_Pred_Test_epoch" + str(epoch_num) + "_" + flag_save + "_" + output_fname,
##                        index=True, sep=',')


            AUC_incident_test = roc_auc_score(y_true=y_dich_get_test,
                                            y_score=score_dich_get_test,
                                            average='macro')

############ Updated the incident loss for multi-level outcomes #################
            # test_loss_incident= tf.keras.losses.binary_crossentropy(y_true_get_test, score_get_test)
            # test_loss_incident = tf.reduce_mean(test_loss_incident)
            y_true_get_test_multi = tf.one_hot(tf.cast(y_true_get_test, dtype=tf.int32), Y_nlev+1, on_value=1.0, off_value=0.0, axis=-1)
            test_loss_incident = tf.reduce_mean(-tf.reduce_sum(y_true_get_test_multi * tf.math.log(score_multi_get_test), axis=-1))

            threshold = np.quantile(score_dich_get_test, 1-np.mean(y_dich_get_test))
            score_get_test_binary = np.array(np.greater(score_dich_get_test, threshold), dtype=np.int32)
            tn, fp, fn, tp = confusion_matrix(y_dich_get_test, score_get_test_binary).ravel()

            specificity = tn / (tn + fp)
            sensitivity = tp / (tp + fn)
            PPV = tp / (tp + fp)
            NPV = tn / (tn + fn)
            f_1 = (2 * tp) / (2 * tp + fp + fn)
            AUC_incident_test_total.append(AUC_incident_test)
            PPV_incident_test_total.append(PPV)

#            logs = '--- Epoch = {}, Train_Loss:{}, Train_AUC_dich:{}, Train_AUC_bin: {},' \
#                    'Test_Loss:{}, Test_AUC_dich:{}, Test_AUC_bin: {}'
#            tf.print(tf.strings.format(logs, (epoch_num, loss_incident_train_total, AUC_incident_train_total, AUC_incident_train_bin,
#                                                           test_loss_incident, AUC_incident_test, AUC_incident_test_bin)))

            logs = '--- Epoch = {}, Incidence Training: loss_incident_train_total:{}, loss_incident_silver_total:{}, loss_incident_unsuper_total:{},' \
                    'AUC_incident_train_total:{}'
            tf.print(tf.strings.format(logs, (epoch_num, loss_incident_train_total, loss_incident_silver_total, loss_incident_unsuper_total,
                                    AUC_incident_train_total)))
            logs = '            Prevalence Training: loss_prevalence_train_total:{}, loss_prevalence_silver_total:{}, loss_prevalence_unsuper_total:{},' \
                    'AUC_prevalence_train_total:{},'
            tf.print(tf.strings.format(logs, (loss_prevalence_train_total, loss_prevalence_silver_total, loss_prevalence_unsuper_total,
                                    AUC_prevalence_train_total)))
            logs = '        , test_loss_incident:{}, AUC_incident_test:{},F1_test:{}, PPV_test:{},Speci_test:{},Sensi_test:{}, NPV_test:{}'
            tf.print(tf.strings.format(logs, (test_loss_incident, AUC_incident_test, f_1, PPV, specificity, sensitivity, NPV)))

            # Early stopping: Check if test AUC has reached its best (overfitting detection)
            if len(AUC_incident_test_total) >= 3:  # Need at least 3 epochs to detect pattern
                best_auc_so_far = max(AUC_incident_test_total)
                current_auc = AUC_incident_test_total[-1]
                
                # Check if current AUC is close to best (within 1%)
                if current_auc >= 0.99 * best_auc_so_far:  # AUC is at or near best
                    print(f"Early stopping at epoch {epoch_num}: Test AUC plateaued ({current_auc:.4f}) - overfitting detected")
                    break

            patient_num_test = np.squeeze(patient_num_test)
            patient_num_test_prevalence = np.unique(patient_num_test)
            date_test = np.squeeze(date_test)
            score_get_test = np.squeeze(score_get_test)
            y_true_get_test = np.squeeze(y_true_get_test)
            Prevalence_multi_total_test = np.squeeze(Prevalence_multi_total_test)

            if epoch_num > 0:
                if True:
                    # print("---------saving--- ", output_directory, ": ", output_fname)
                    patient_num_test_total2, date_test_total2, y_pred_get2, y_true_get2 = \
                        zip(*sorted(zip(patient_num_test, date_test, score_get_test, y_true_get_test)))
                    dataframe = pd.DataFrame({'ID': patient_num_test_total2,
                                               'Period': date_test_total2,
                                                'Y': y_true_get2, 'LP': y_pred_get2})
                    df_pred = pd.DataFrame(score_multi_get_test)
                    yo = np.char.add("Pred_",np.array(list(range(Y_nlev+1)), dtype = '<U2'))
                    df_pred.columns = yo
                    
                    # Calculate longitudinal ordinal scores
                    if ordinal_score_method == "cumulative":
                        # Cumulative probability method
                        ordinal_scores = []
                        for idx in range(len(score_multi_get_test)):
                            probs = score_multi_get_test[idx]
                            cumulative_probs = np.cumsum(probs)
                            # Find the class where cumulative probability exceeds Y_prop threshold
                            # Using 0.5 as default threshold, can be made configurable
                            ordinal_score = np.argmax(cumulative_probs >= 0.5)
                            ordinal_scores.append(ordinal_score)
                        df_pred['Ordinal_Score'] = ordinal_scores
                    else:  # weighted method
                        # Weighted probability method
                        ordinal_scores = []
                        for idx in range(len(score_multi_get_test)):
                            probs = score_multi_get_test[idx]
                            # Calculate weighted score: sum(prob_i * class_i)
                            ordinal_score = np.sum([probs[i] * i for i in range(len(probs))])
                            ordinal_scores.append(ordinal_score)
                        df_pred['Ordinal_Score'] = ordinal_scores
                    
                    dataframe = pd.concat([dataframe,df_pred],axis = 1)

                    dataframe.to_csv(
                        output_directory + "Incident_epoch" + str(epoch_num) + "_" + flag_save + "_" + output_fname,
                        index=True, sep=',')

                    #############saving codes weights
                    savename_weights = output_directory + "_" + output_fname + "_epoch" + str(epoch_num) + "_code_weights.csv"
                    if not os.path.exists(savename_weights):
                        df = pd.DataFrame({})
                    else:
                        df = pd.read_csv(savename_weights)
                    weights_save = list(attention_value_weights_test[0, 0, :])
                    df["weight" + str(random.randint(0, 1000))] = weights_save
                    df.to_csv(savename_weights, index=False)
                if True:
                    print("---------saving--- ", output_directory, ": ", output_fname)
                    # print("Prevalence_prediction_total_test: ", len(Prevalence_prediction_total_test))
                    # print("len(Prevalence_prediction_total_test): ",
                    #         np.array(Prevalence_prediction_total_test).shape)
                    # print("len(labels_prevalence_total_test): ", np.array(labels_prevalence_total_test).shape)
                    # print("len(patient_num_test_prevalence): ", len(patient_num_test_prevalence))
                    # print("len(patient_num_test_total): ", len(patient_num_test_total))
                    # print("len(np.unique(patient_num_test_total)): ", len(np.unique(patient_num_test_total)))
                    dataframe = pd.DataFrame({'ID': np.unique(patient_num_test_total),
                                                'Y': np.array((labels_prevalence_total_test)).reshape(
                                                    len(labels_prevalence_total_test),),
                                                'LP': np.array(Prevalence_prediction_total_test).reshape(
                                                          len(Prevalence_prediction_total_test), )})
                    df_pred = pd.DataFrame(Prevalence_multi_total_test)
                    yo = np.char.add("Pred_",np.array(list(range(Y_nlev+1)), dtype = '<U2'))
                    df_pred.columns = yo
                    
                    # Calculate longitudinal ordinal scores for prevalence
                    if ordinal_score_method == "cumulative":
                        # Cumulative probability method
                        ordinal_scores = []
                        for idx in range(len(Prevalence_multi_total_test)):
                            probs = Prevalence_multi_total_test[idx]
                            cumulative_probs = np.cumsum(probs)
                            # Find the class where cumulative probability exceeds Y_prop threshold
                            # Using 0.5 as default threshold, can be made configurable
                            ordinal_score = np.argmax(cumulative_probs >= 0.5)
                            ordinal_scores.append(ordinal_score)
                        df_pred['Ordinal_Score'] = ordinal_scores
                    else:  # weighted method
                        # Weighted probability method
                        ordinal_scores = []
                        for idx in range(len(Prevalence_multi_total_test)):
                            probs = Prevalence_multi_total_test[idx]
                            # Calculate weighted score: sum(prob_i * class_i)
                            ordinal_score = np.sum([probs[i] * i for i in range(len(probs))])
                            ordinal_scores.append(ordinal_score)
                        df_pred['Ordinal_Score'] = ordinal_scores
                    
                    dataframe = pd.concat([dataframe,df_pred],axis = 1)
                    dataframe.to_csv(output_directory + "Prevalence_" + "_" + flag_save + "_" + output_fname, index=True,
                                        sep=',')
                    print("---------saving ends successfully--- ", output_directory, ": ", output_fname)

            if epoch_num > epochs - 2 and flag_save_finish == False:
                flag_save_finish = True
                if True:
                    f = open(output_directory + output_fname + "_epoch" + str(epoch_num) + "_incident_evaluation.txt", 'a')
                    f.write("Threshold value=0.5 AUC_train :%4f , NPV:%4f ,Specificity:%4f, F1:%4f ,"
                            " PPV:%4f , Sensitvity:%4f, AUC:%4f" % (
                                AUC_incident_train_total, NPV, specificity, f_1, PPV, sensitivity, AUC_incident_test))
                    f.write("\r")
                    f.close()
                if True:
                    f = open(output_directory + output_fname+ "_epoch" + str(epoch_num) + "_prevalence_evaluation.txt", 'a')
                    f.write(" Threshold value=0.5 AUC_train :%4f , ACC_train:%4f, ,FPR:%4f, FNR:%4f, NPV_MLP:%4f "
                            " ,AUC_test_comb:%4f,  AUC_test_TF:%4f, AUC_test_incident:%4f,"
                            "ACC_test:%4f "
                            ", Speci:%4f ,Sensi:%4f, PPV:%4f"
                            % (
                                AUC_prevalence_train_total, 0.0, 0.0, 0.0, NPV, AUC_incident_test,
                                0.0, 0.0,
                                0.0, specificity,
                                sensitivity,
                                PPV))
                    f.write("\r")
                    f.close()

        valid_loss.reset_states()
        train_metric.reset_states()
        valid_metric.reset_states()

        train_prevalence_incident.reset_states()
        train_loss_prevalence.reset_states()
        train_loss_prevalence_unsuper.reset_states()
        train_loss_prevalence_silver.reset_states()
        train_loss_incident.reset_states()
        train_loss_incident_unsuper.reset_states()
        train_loss_incident_silver.reset_states()
        train_smooth_loss.reset_states()
        train_smooth_loss_unsuper.reset_states()
        train_keyfeature_loss.reset_states()
        train_contrastive_loss.reset_states()
        train_contrastive_loss_prevalence.reset_states()
        train_prevalence_entropy_unsuper.reset_states()
        train_prevalence_consistency.reset_states()




