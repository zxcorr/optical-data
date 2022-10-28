########################################################################################################
########################################################################## This script compute the photo
-z for the keras program #############################################
########################################################################################################
###################################

import numpy as np
import pandas as pd
from astropy.table import Table
import os
import pickle
from tensorflow.keras.models import load_model
import argparse
import time

def remove_outliers(df, feat):
    df.loc[df[feat[0]]>90,feat[1]] = df[df[feat[0]]<90][feat[0]].max()
    df.loc[df[feat[1]]>90,feat[1]] = df[df[feat[1]]<90][feat[1]].max()
    df.loc[df[feat[2]]>90,feat[2]] = df[df[feat[2]]<90][feat[2]].max()
    df.loc[df[feat[3]]>90,feat[3]] = df[df[feat[3]]<90][feat[3]].max()
    df.loc[df[feat[4]]>90,feat[4]] = df[df[feat[4]]<90][feat[4]].max()

    # if the data has nan values
    df[feat[0]].fillna(df[df[feat[0]]<90][feat[0]].max())
    df[feat[1]].fillna(df[df[feat[1]]<90][feat[1]].max())
    df[feat[2]].fillna(df[df[feat[2]]<90][feat[2]].max())
    df[feat[3]].fillna(df[df[feat[3]]<90][feat[3]].max())
    df[feat[4]].fillna(df[df[feat[4]]<90][feat[4]].max())

    return df

def load_models(number_sim, models_folder):
    models = []
    for i in range(10):
        model =load_model(models_folder + 'keras_model_' + str(i+1) + '.h5')
        models.append(model)

    return models

def main(input_folder, models_folder, number_sim, core, modulus, mag_G, mag_R, mag_I, mag_Z, mag_Y, pref
ix, sufix, outpath):
    """
    """
    assert os.path.exists(input_folder), "input catalog folder doesn't exist."
    assert os.path.exists(models_folder), "models catalog folder doesn't exist."
    if not os.path.exists(os.path.dirname(outpath)):
        os.mkdir(os.path.dirname(outpath))

    number_sim = int(number_sim, base=10)
    core = int(core, base=10)
    modulus = int(modulus, base=10)

    models = load_models(number_sim, models_folder)

    feat = [mag_G, mag_R, mag_I, mag_Z, mag_Y]

    filename = os.listdir(input_folder)
    for i in range(len(filename)):
        if i%core==modulus:
            start_time = time.time()
            print(os.path.join(input_folder, filename[i]))
            data_aux = Table.read(os.path.join(input_folder, filename[i])).to_pandas()
            data = remove_outliers(data_aux, feat)

            X = np.zeros(shape=(len(data), 5))
            X[:, 0] = data[feat[0]].values - data[feat[1]].values
            X[:, 1] = data[feat[1]].values - data[feat[2]].values
            X[:, 2] = data[feat[2]].values - data[feat[3]].values
            X[:, 3] = data[feat[3]].values - data[feat[4]].values
            X[:, 4] = data[feat[4]].values


            dataset_output = pd.DataFrame()
            dataset_output["COADD_OBJECT_ID"] = data["COADD_OBJECT_ID"]

            mu_keras = []

            for j in range(number_sim):
                predictions = models[j].predict(X)
                zphot = predictions[0].flatten()
                dataset_output["keras:zphot_" + str(j+1)] = zphot
                mu_keras.append(zphot)


            #pdf = pd.DataFrame(predictions[1])
            #pdf.to_csv('/home/afarias/pdf_keras.csv', index=False)


            mean_keras = pd.Series(np.round(sum(np.array(mu_keras)/10),6), name="mean_keras")
            standard_dev_keras = pd.Series(np.round(np.std(np.array(mu_keras), axis=0),6), name="StanDev
_keras")

            dataset_output["Mean_keras"] = mean_keras
            dataset_output["StanDev_keras"] = standard_dev_keras

            pixel = filename[i][len(prefix):len(prefix) + 5]
            nome = prefix + pixel + "_KERAS" + sufix
            print(outpath + nome)
            dataset_output.to_csv(outpath + str(nome),index = False)
            print("--- %s seconds ---" % (time.time() - start_time))



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

    parser.add_argument('--input_folder')
    parser.add_argument('--models_folder')
    parser.add_argument('--number_sim')
    parser.add_argument('--core')
    parser.add_argument('--modulus')
    parser.add_argument('--mag_G')
    parser.add_argument('--mag_R')
    parser.add_argument('--mag_I')
    parser.add_argument('--mag_Z')
    parser.add_argument('--mag_Y')
    parser.add_argument('--prefix')
    parser.add_argument('--sufix')
    parser.add_argument('--outpath')

    args = parser.parse_args()
    print('input folder:', args.input_folder)
    print('models catalog:', args.models_folder)
    print('outpath folder:', args.outpath)
    main(args.input_folder, args.models_folder, args.number_sim, args.core, args.modulus, args.mag_G, ar
gs.mag_R, args.mag_I, args.mag_Z, args.mag_Y, args.prefix, args.sufix, args.outpath)
