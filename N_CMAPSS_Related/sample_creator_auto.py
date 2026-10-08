
import gc
import argparse
import json
import os
import numpy as np
import pandas as pd
import random
from os.path import basename as opb, splitext as ops

from .N_CMAPSS_create_sample import df_all_creator, df_train_creator, df_test_creator, Input_Gen
from glob import glob

seed = 0
random.seed(0)
np.random.seed(seed)


data_filedir = './N_CMAPSS'


_GEN_CONFIG_MARKER = '.gen_config.json'


def _sampler_up_to_date(sample_dir_path, sequence_length, stride, sampling):

    marker_path = os.path.join(sample_dir_path, _GEN_CONFIG_MARKER)
    if not os.path.exists(marker_path):
        return False
    try:
        with open(marker_path, 'r') as f:
            saved = json.load(f)
    except Exception:
        return False
    return (saved.get('sequence_length') == sequence_length
            and saved.get('stride') == stride
            and saved.get('sampling') == sampling)


def _write_sampler_config(sample_dir_path, sequence_length, stride, sampling):
    marker_path = os.path.join(sample_dir_path, _GEN_CONFIG_MARKER)
    with open(marker_path, 'w') as f:
        json.dump({'sequence_length': sequence_length, 'stride': stride, 'sampling': sampling}, f)


def sampler(args):
    sequence_length = args.input_length
    stride = args.s
    sampling = args.sampling

    file_devtest_df = pd.read_csv("./N_CMAPSS_Related/File_DevUnits_TestUnits.csv")
    for data_filepath in glob('./N_CMAPSS/*'):
        if data_filepath.endswith('.h5') is not True:
            continue

        if args.Data_id_N_CMAPSS in data_filepath:
            sample_dir_path = os.path.join(data_filedir, 'Samples_whole', ops(opb(data_filepath))[0])
            if _sampler_up_to_date(sample_dir_path, sequence_length, stride, sampling):
                print(f"{args.Data_id_N_CMAPSS}: samples already generated for "
                      f"input_length={sequence_length} stride={stride} sampling={sampling} -- skipping regeneration")
                continue

            print(args.Data_id_N_CMAPSS)

            '''
            W: operative conditions (Scenario descriptors)
            X_s: measured signals
            X_v: virtual sensors
            T(theta): engine health parameters
            Y: RUL [in cycles]
            A: auxiliary data
            '''

            df_all = df_all_creator(data_filepath, sampling)
            max_cycle_life = df_all['RUL'].max()
            max_life = int(args.rate * max_cycle_life)
            df_all['RUL'] = df_all['RUL'].clip(upper=max_life)
            '''
            Split dataframe into Train and Test
            Training units: 2, 5, 10, 16, 18, 20
            Test units: 11, 14, 15

            ,File,Dev Units,Test Units
            0,dataset/N-CMAPSS_DS01-005.h5,[1 2 3 4 5 6],[ 7  8  9 10]
            1,dataset/N-CMAPSS_DS04.h5,[1 2 3 4 5 6],[ 7  8  9 10]
            2,dataset/N-CMAPSS_DS08a-009.h5,[1 2 3 4 5 6 7 8 9],[10 11 12 13 14 15]
            3,dataset/N-CMAPSS_DS05.h5,[1 2 3 4 5 6],[ 7  8  9 10]
            4,dataset/N-CMAPSS_DS02-006.h5,[ 2  5 10 16 18 20],[11 14 15]
            5,dataset/N-CMAPSS_DS08c-008.h5,[1 2 3 4 5 6],[ 7  8  9 10]
            6,dataset/N-CMAPSS_DS03-012.h5,[1 2 3 4 5 6 7 8 9],[10 11 12 13 14 15]
            7,dataset/N-CMAPSS_DS07.h5,[1 2 3 4 5 6],[ 7  8  9 10]
            8,dataset/N-CMAPSS_DS06.h5,[1 2 3 4 5 6],[ 7  8  9 10]

            '''
            units_index_train = np.fromstring(
                file_devtest_df[file_devtest_df.File == opb(data_filepath)]["Dev Units"].values[0][1:-1],
                dtype=float, sep=' ').tolist()
            units_index_test = np.fromstring(
                file_devtest_df[file_devtest_df.File == opb(data_filepath)]["Test Units"].values[0][1:-1],
                dtype=float, sep=' ').tolist()


            df_train = df_train_creator(df_all, units_index_train)

            df_test = df_test_creator(df_all, units_index_test)

            del df_all
            gc.collect()
            df_all = pd.DataFrame()
            sample_folder = os.path.isdir(sample_dir_path)
            if not sample_folder:
                os.makedirs(sample_dir_path)

            cols_normalize = df_train.columns.difference(['RUL', 'unit', 'cycle'])
            sequence_cols = df_train.columns.difference(['RUL', 'unit', 'cycle'])
            index_cols = df_train.columns.difference(cols_normalize)


            for unit_index in units_index_train:
                data_class = Input_Gen(df_train, df_test, cols_normalize, index_cols, sequence_length, sequence_cols,
                                           sample_dir_path,
                                           unit_index, sampling, max_life, stride=stride)
                data_class.seq_gen()


            for unit_index in units_index_test:
                data_class = Input_Gen(df_train, df_test, cols_normalize, index_cols, sequence_length, sequence_cols,
                                           sample_dir_path,
                                           unit_index, sampling, max_life,  stride=stride)
                data_class.seq_gen()

            _write_sampler_config(sample_dir_path, sequence_length, stride, sampling)
