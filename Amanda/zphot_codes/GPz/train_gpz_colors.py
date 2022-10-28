########################################################################################################
###################################################################
## This script trains the GPz machine learning with the match from DES DR2 X VIPERS USING 4 COLORS AND 1
 MAGNITUDE
########################################################################################################
###################################################################

import numpy as np
import pandas as pd
from astropy.table import Table
import GPz
import math
import random
import pickle
import argparse
import os

def remove_outliers(df):
    feat = ['MAG_AUTO_G_DERED', 'MAG_AUTO_R_DERED', 'MAG_AUTO_I_DERED', 'MAG_AUTO_Z_DERED', 'MAG_AUTO_Y_
DERED']
    df.loc[df[feat[0]]>90,feat[1]] = df[df[feat[0]]<90][feat[0]].max()
    df.loc[df[feat[1]]>90,feat[1]] = df[df[feat[1]]<90][feat[1]].max()
    df.loc[df[feat[2]]>90,feat[2]] = df[df[feat[2]]<90][feat[2]].max()
    df.loc[df[feat[3]]>90,feat[3]] = df[df[feat[3]]<90][feat[3]].max()
    df.loc[df[feat[4]]>90,feat[4]] = df[df[feat[4]]<90][feat[4]].max()

    return df

def split(n, trainSample, validSample, testSample):

    if(trainSample<=1):
        validSample = math.ceil(n * validSample)
        testSample = math.ceil(n * testSample)
        trainSample = min([math.ceil(n * trainSample), n - testSample - validSample])

    r = np.random.permutation(n)

    validSample = int(validSample)
    testSample = int(testSample)
    trainSample = int(trainSample)

    validation = np.zeros(n, dtype=bool)
    testing = np.zeros(n, dtype=bool)
    training = np.zeros(n, dtype=bool)

    validation[r[0:validSample]] = True
    testing[r[validSample:validSample + testSample]] = True
    training[r[validSample + testSample:validSample + testSample + trainSample]] = True

    return training, validation, testing

def main(DR2_folder, VIPERS_catalog, train_size, test_size, number_sim, method, maxIter, maxAttempts, m,
 binWidth, outpath):
    """
    """
    assert os.path.exists(DR2_folder), "DR2 catalog folder doesn't exist."
    assert os.path.exists(VIPERS_catalog), "VIPERS catalog folder doesn't exist."
    if not os.path.exists(os.path.dirname(outpath)):
        os.mkdir(os.path.dirname(outpath))

    data_aux = Table.read(VIPERS_catalog).to_pandas()
    vipers = data_aux[data_aux["source"] == b'VIPERS'].copy()

    data = remove_outliers(vipers)

    X = pd.DataFrame()
    X["G-R"] = data["MAG_AUTO_G_DERED"] - data ["MAG_AUTO_R_DERED"]
    X["R-I"] = data["MAG_AUTO_R_DERED"]- data ["MAG_AUTO_I_DERED"]
    X["I-Z"] = data["MAG_AUTO_I_DERED"]- data ["MAG_AUTO_Z_DERED"]
    X["Z-Y"] = data["MAG_AUTO_Z_DERED"]- data ["MAG_AUTO_Y_DERED"]
    X["MAG_AUTO_I_DERED"] = data["MAG_AUTO_I_DERED"]
    X["MAGERR_AUTO_G-R"] = np.sqrt(data["MAGERR_AUTO_G"]**2 + data ["MAGERR_AUTO_R"]**2)
    X["MAGERR_AUTO_R-I"] = np.sqrt(data["MAGERR_AUTO_R"]**2 + data ["MAGERR_AUTO_I"]**2)
    X["MAGERR_AUTO_I-Z"] = np.sqrt(data["MAGERR_AUTO_Z"]**2 + data ["MAGERR_AUTO_Z"]**2)
    X["MAGERR_AUTO_Z-Y"] = np.sqrt(data["MAGERR_AUTO_Y"]**2 + data ["MAGERR_AUTO_Y"]**2)
    X["MAGERR_AUTO_I"] = data["MAGERR_AUTO_I"]
    X = X.to_numpy()
    Y = data["z"].values.reshape(-1, 1)

    n, d = X.shape
    trainSplit = float(train_size)                      # percentage of data to use for training
    testSplit = float(test_size)                        # percentage of data to use for testing
    validSplit = 1 - (trainSplit + testSplit)           # percentage of data to use for validation
    training, validation, testing = split(n, trainSplit, validSplit, testSplit)
    filters = int(d/2)

    maxIter = int(maxIter)              # maximum number of iterations [default=200]
    maxAttempts = int(maxAttempts)     # maximum iterations to attempt if there is no progress on the va
lidation set [default=infinity]
    m = int(m)                          # number of basis functions to use [required]
    heteroscedastic = True              # learn a heteroscedastic noise process, set to false interested
 only in point estimates
    csl_method = 'normal'               # cost-sensitive learning option: [default='normal']
    joint = True                        # jointly learn a prior linear mean function [default=true]
    decorrelate = True                  # preprocess the data using PCA [default=False]
    binWidth = float(binWidth)            # the width of the bin for 'balanced' cost-sensitive learning
[default=range(z_spec)/100]

    for i in range(int(number_sim)):
        model = GPz.GP(m, method=method, joint=joint,heteroscedastic=heteroscedastic, decorrelate=decorr
elate)
        random.seed(i)
        omega = GPz.getOmega(Y, method=csl_method)
        model.train(X.copy(), Y.copy(), omega=omega, training=training,validation=validation, maxIter=ma
xIter, maxAttempts=maxAttempts)
        pickle.dump(model, open(outpath + 'model_GPz_' + str(i+1) + ".pkl", "wb"))

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
    parser.add_argument('--VIPERS_catalog')
    parser.add_argument('--train_size')
    parser.add_argument('--test_size')
    parser.add_argument('--number_sim')
    parser.add_argument('--method')
    parser.add_argument('--maxIter')
    parser.add_argument('--maxAttempts')
    parser.add_argument('--m')
    parser.add_argument('--binWidth')
    parser.add_argument('--outpath')

    args = parser.parse_args()
    print('DR2 folder:', args.DR2_folder)
    print('VIPERS catalog:', args.VIPERS_catalog)
    print('Number of simulations:', args.number_sim)
    print('Size of train dataset:', args.train_size)
    print('Size of test dataset:', args.test_size)
    print('Method:', args.method)
    main(args.DR2_folder, args.VIPERS_catalog, args.train_size , args.test_size , args.number_sim , args
.method , args.maxIter , args.maxAttempts , args.m, args.binWidth, args.outpath)
