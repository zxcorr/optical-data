########################################################################################################
##############################################################
# This script computes the photo-z for DES DR2 X VIPERS
########################################################################################################
#############################################################

import numpy as np
import pandas as pd
from astropy.table import Table
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
        model = pickle.load(open(models_folder + "modelo_GPz_" + str(i+1) + ".pkl", "rb"))
        models.append(model)

    return models

def main(DR2_folder, models_folder, number_sim, outpath):
    """
    """
    assert os.path.exists(DR2_folder), "DR2 catalog folder doesn't exist."
    assert os.path.exists(models_folder), "models catalog folder doesn't exist."
    if not os.path.exists(os.path.dirname(outpath)):
        os.mkdir(os.path.dirname(outpath))

    number_sim = int(number_sim)
    models = load_models(number_sim, models_folder)


    filename = os.listdir(DR2_folder)
    for i in range(len(filename)):
        data_aux = Table.read(os.path.join(DR2_folder, filename[i])).to_pandas()
        data = remove_outliers(data_aux)

        X = data[['MAG_AUTO_G_DERED', 'MAG_AUTO_R_DERED', 'MAG_AUTO_I_DERED', 'MAG_AUTO_Z_DERED', 'MAG_A
UTO_Y_DERED']].values
        err = data[["MAGERR_AUTO_G", "MAGERR_AUTO_R", "MAGERR_AUTO_I", "MAGERR_AUTO_Z", "MAGERR_AUTO_Y"]
].values

        X = np.concatenate((X, err), axis=1)

        dataset_output = pd.DataFrame()
        dataset_output["COADD_OBJECT_ID"] = data["COADD_OBJECT_ID"]
        #dataset_output["TILENAME"] = data["TILENAME"]
        #dataset_output["HPIX_32"] = data["HPIX_32"]
        #dataset_output["HPIX_64"] = data["HPIX_64"]
        #dataset_output["HPIX_1024"] = data["HPIX_1024"]
        #dataset_output["HPIX_4096"] = data["HPIX_4096"]
        #dataset_output["HPIX_16384"] = data["HPIX_16384"]
        #dataset_output["RA"] = data["RA"]
        #dataset_output["DEC"] = data["DEC"]
        #dataset_output["MAG_AUTO_G_DERED"] = data["MAG_AUTO_G_DERED"]
        #dataset_output["MAG_AUTO_R_DERED"] = data["MAG_AUTO_R_DERED"]
        #dataset_output["MAG_AUTO_I_DERED"] = data["MAG_AUTO_I_DERED"]
        #dataset_output["MAG_AUTO_Z_DERED"] = data["MAG_AUTO_Z_DERED"]
        #dataset_output["MAG_AUTO_Y_DERED"] = data["MAG_AUTO_Y_DERED"]

        mu_GPz = []
        mu_GPz_sigma = []

        for j in range(number_sim):
            mu, sigma, modelV,noiseV, _ = models[j].predict(X[:,:].copy(),model='best')
            zphot = mu.flatten()
            dataset_output['GPz:zphot_' + str(j+1)] = zphot
            dataset_output['GPz:sigma_' + str(j+1)] = sigma.flatten()
            mu_GPz.append(zphot)
            mu_GPz_sigma.append(sigma.flatten())

        mean_GPz = pd.Series(np.round(sum(np.array(mu_GPz)/10),6), name="mean_GPz")
        standard_dev_GPz = pd.Series(np.round(np.std(np.array(mu_GPz), axis=0),6), name="StanDev_GPz")
        mean_GPz_sigma = pd.Series(np.round(sum(np.array(mu_GPz_sigma)/10),6), name="mean_GPz_sigma")
        standard_dev_GPz_sigma = pd.Series(np.round(np.std(np.array(mu_GPz_sigma), axis=0),6), name="Sta
nDev_GPz_sigma")

        dataset_output["Mean_GPz"] = mean_GPz
        dataset_output["StanDev_GPz"] = standard_dev_GPz
        dataset_output["Mean_GPz_sigma"] = mean_GPz_sigma
        dataset_output["StanDev_GPz_sigma"] = standard_dev_GPz_sigma

        name = filename[i][:17] + "_GPz" + filename[i][17:]
        print(outpath + str(name))
        dataset_output.to_csv(outpath + str(name),index = False)

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
    parser.add_argument('--models_folder')
    parser.add_argument('--number_sim')
    parser.add_argument('--outpath')

    args = parser.parse_args()
    print('DR2 folder:', args.DR2_folder)
    print('models catalog:', args.models_folder)
    print('Number of simulations:', args.number_sim)
    main(args.DR2_folder, args.models_folder, args.number_sim, args.outpath)
