########################################################################################################
###################################
######################### This script compute the models for the keras program by using the match DES DR
2 X VIPERS ########################
########################################################################################################
###################################

from scipy.sparse import hstack

# Neural Network Libs
from tensorflow import keras
from sklearn.preprocessing import KBinsDiscretizer
from tensorflow.keras import regularizers
from tensorflow.keras import layers
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Activation
from tensorflow.keras.models import Sequential
import tensorflow as tf
import numpy as np
import pandas as pd
from astropy.table import Table
import argparse
import os


def remove_outliers(df, feat):
    df.loc[df[feat[0]]>90,feat[0]] = df[df[feat[0]]<90][feat[0]].max()
    df.loc[df[feat[1]]>90,feat[1]] = df[df[feat[1]]<90][feat[1]].max()
    df.loc[df[feat[2]]>90,feat[2]] = df[df[feat[2]]<90][feat[2]].max()
    df.loc[df[feat[3]]>90,feat[3]] = df[df[feat[3]]<90][feat[3]].max()
    df.loc[df[feat[4]]>90,feat[4]] = df[df[feat[4]]<90][feat[4]].max()

    return df

def get_setting_and_features(df, feat, redshift):
    X = np.zeros(shape=(len(df), 5))
    X[:, 0] = df[feat[0]].values - df[feat[1]].values
    X[:, 1] = df[feat[1]].values - df[feat[2]].values
    X[:, 2] = df[feat[2]].values - df[feat[3]].values
    X[:, 3] = df[feat[3]].values - df[feat[4]].values
    X[:, 4] = df[feat[4]].values

    y = df[redshift].values
    print(y)
    y = y.reshape(-1, 1)
    kbins = KBinsDiscretizer(200, encode="onehot", strategy="uniform")
    kbins.fit(y.reshape(-1, 1))
    y_bins = kbins.transform(y.reshape(-1, 1))

    y_total = hstack([y_bins, y])
    y_total = y_total.toarray()

    X = np.concatenate((X, df[feat].values), axis=1)

    return X, y_total


def main(DR2_folder, vipers_folder, mag_G, mag_R, mag_I, mag_Z, mag_Y, number_sim, redshift, outpath):
    """
    """
    assert os.path.exists(DR2_folder), "DES catalog folder doesn't exist."
    assert os.path.exists(vipers_folder), "VIPERS catalog folder doesn't exist."
    if not os.path.exists(os.path.dirname(outpath)):
        os.mkdir(os.path.dirname(outpath))

    data_aux = Table.read(vipers_folder).to_pandas()
    feat = [mag_G, mag_R, mag_I, mag_Z, mag_Y]
    data = remove_outliers(data_aux, feat)

    X, y_total = get_setting_and_features(data, feat, redshift)

    number_sim = int(number_sim, base=10)


    for i in range(number_sim):
        EarlyStop = EarlyStopping(monitor='reg_mse', mode='min', patience=25)
        BATCH_SIZE = 64
        STEPS_PER_EPOCH = len(data)//BATCH_SIZE
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            0.0001,
            decay_steps=STEPS_PER_EPOCH*1000,
            decay_rate=1,
            staircase=False)
        inputs = keras.layers.Input(5)
        x = BatchNormalization()(inputs)
        x = Dense(30, kernel_initializer='normal',  kernel_constraint=max_norm(2.), activation='tanh', k
ernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=regularizers.l2(1e-4), activity
_regularizer=regularizers.l2(1e-5))(x)
        output1 = Dense(1, activation="linear", name="reg")(x)
        output2 = Dense(200, activation="softmax", name="pdf")(x)
        model_ann = keras.Model(inputs=inputs, outputs=[output1, output2], name="rafael")
        model_ann.compile(loss={'reg': 'mean_absolute_error', 'pdf': keras.losses.CategoricalCrossentrop
y()}, loss_weights=[0.1, 0.9],
                          optimizer=tf.keras.optimizers.Adam(lr_schedule),
                          metrics={'pdf': "acc", 'reg': "mse"})
        history = model_ann.fit(X[:, :5],
                                {'pdf': y_total[:, :200], 'reg': y_total[:, 200]}, batch_size=128, epoch
s=256, validation_split=0.2)

        model_ann.save(outpath + '/keras_model_' + str(i+1) + ".h5")

    return None


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=
    '''
    FIXME
    FIXME-FIXME-FIXME
    '''
    , fromfile_prefix_chars='@')

    parser.add_argument('--DR2_folder')
    parser.add_argument('--vipers_folder')
    parser.add_argument('--mag_G')
    parser.add_argument('--mag_R')
    parser.add_argument('--mag_I')
    parser.add_argument('--mag_Z')
    parser.add_argument('--mag_Y')
    parser.add_argument('--number_sim')
    parser.add_argument('--redshift')
    parser.add_argument('--outpath')

    args = parser.parse_args()
    print('DR2 folder:', args.DR2_folder)
    print('VIPERS catalog:', args.vipers_folder)
    main(args.DR2_folder, args.vipers_folder, args.mag_G, args.mag_R, args.mag_I, args.mag_Z, args.mag_Y
, args.number_sim, args.redshift, args.outpath)
