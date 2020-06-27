'''
    Author: Chen Lin
    Email: chen.lin@emory.edu
    Date created: 2020/6/24
    Python Version: 3.6
'''
import sys
sys.path.append('.')
import numpy as np
import pandas as pd
from airpolnowcast.data.utils import read_raw_data
from airpolnowcast.features.build_features import process_data
import pickle
from airpolnowcast.data.utils import read_query_from_file, create_folder_exist
import os


# generate sequence input features for LSTM training
def generate_input_sequence(input_array, seq_length):
    """

    :param input_array: np.array
        shape: N*P
        N: number of days
        P: number of features for day i
    :param seq_length: int
        sequence length for LSTM model
    :return: np.array
        shape: N*seq_length*P
    """
    embedding_dim = input_array.shape[1]
    input_embedding = []
    for i in range(len(input_array)):
        input_series = []
        for days_index in range(i - seq_length + 1, i + 1):
            if days_index >= 0:
                day_embedding = input_array[days_index]
            else:
                na_vec = np.array([0. for i in range(embedding_dim)])
                day_embedding = na_vec

            input_series.append(day_embedding)
        input_embedding.append(np.array(input_series))
    input_embedding = np.array(input_embedding)
    return input_embedding


def get_delta(masking, delta):
    # fill the delta vectors
    for index, value in np.ndenumerate(masking):
        '''
        index[0] = row, agg
        index[1] = col, time
        '''
        if index[1] == 0:
            delta[index[0], index[1]] = 0
        elif masking[index[0], index[1] - 1] == 0:
            delta[index[0], index[1]] = 1 + delta[index[0], index[1] - 1]
        else:
            delta[index[0], index[1]] = 1

    return delta


def trend_fea_to_delta(trend_fea):
    trend_for_grud = trend_fea.fillna(0.0)
    trend_for_grud = (trend_for_grud != 0)
    masking = np.array(trend_for_grud.astype(int).T)
    delta = np.zeros((masking.shape[0], masking.shape[1]))
    delta = get_delta(masking, delta)

    return masking, delta


def trend_fea_to_x(trend_fea):
    trend_fill_na = trend_fea.fillna(0.).replace(0., np.nan).fillna(method='ffill').fillna(0.)
    trend_fill_na_norm = (trend_fill_na - trend_fill_na.mean())/ trend_fill_na.std()
    trend_fill_na_norm.fillna(0., inplace = True)
    x_mean_aft_nor = np.array(trend_fill_na_norm.mean())
    x_median_aft_nor = np.array(trend_fill_na_norm.median())
    return np.array(trend_fill_na_norm), x_mean_aft_nor, x_median_aft_nor


def generate_x_seq(x, masking, delta):
    trend_seq = generate_input_sequence(x, 7)
    masking_seq = generate_input_sequence(masking.T, 7)
    delta_seq = generate_input_sequence(delta.T, 7)
    t_dataset = np.stack((trend_seq, masking_seq, delta_seq), axis=1)
    t_dataset = np.einsum('klij->klji', t_dataset)

    return t_dataset


# apply lags to search features
def lag_search_features(input_df, lag):
    """

    :param input_df: pd.DataFrame
        the search feature input data
        shape: N*M
        N: number of days
        M: number of search terms
    :param lag: int
        lag of days applied ont search data
    :return: pd.DataFrame
        shape: N*M
        for day i, we have the info of day i+lag (later)
    """
    # record column names
    df_column_names = input_df.columns
    input_df = np.array(input_df)
    embedding_dim = input_df.shape[1]
    reveserse_embeddings = input_df[::-1]
    lag_features = np.roll(reveserse_embeddings, lag, axis=0)
    for i in range(lag):
        na_embedding = np.array([0. for k in range(embedding_dim)])
        lag_features[i] = na_embedding
    lag_features = lag_features[::-1]
    return pd.DataFrame(lag_features, columns=df_column_names)


def save_pkl_dataset(train_data_path, pars):
    # global parameters
    # seed word list
    seed_path = pars['extract_search_trend']['term_list_path']
    seed_word_list = read_query_from_file(seed_path)
    # seq_length = int(pars['train_model']['seq_length'])
    search_lag = int(pars['train_model']['search_lag'])

    # path to save pickle files
    savepkl_folder_path = os.path.join(pars['build_features']['savepkl_folder_path'],
                                       os.path.basename(train_data_path).split('.')[0])

    print(savepkl_folder_path)

    x_mean_path = os.path.join(savepkl_folder_path, pars['build_features']['x_mean_path'])
    x_median_path = os.path.join(savepkl_folder_path, pars['build_features']['x_median_path'])
    t_dataset_path = os.path.join(savepkl_folder_path, pars['build_features']['t_dataset_path'])
    y_out_path = os.path.join(savepkl_folder_path, pars['build_features']['y_out_path'])

    # create folder if not exist
    create_folder_exist(os.path.dirname(x_mean_path))
    train_data = read_raw_data(train_data_path)
    y_data, pol_val, trend_fea, phys_fea = process_data(train_data)
    trend_fea = lag_search_features(trend_fea, search_lag)
    trend_fea = trend_fea[seed_word_list]
    masking, delta = trend_fea_to_delta(trend_fea)
    x, x_mean_aft_nor, x_median_aft_nor = trend_fea_to_x(trend_fea)
    t_dataset = generate_x_seq(x, masking, delta)

    with open(x_mean_path, 'wb') as fi:
        pickle.dump(x_mean_aft_nor, fi)
    with open(x_median_path, 'wb') as fi:
        pickle.dump(x_median_aft_nor, fi)
    with open(t_dataset_path, 'wb') as fo:
        pickle.dump(t_dataset, fo)
    with open(y_out_path, 'wb') as fi:
        pickle.dump(np.array(y_data), fi)








