########################################################################################################
##############################################################
# This script computes the photo-z for DES DR2 X VIPERS USING 4 COLORS AND 1 MAGNITUDE
########################################################################################################
#############################################################

import numpy as np
import pandas as pd
from astropy.table import Table
import pickle
import argparse
import os
import time

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
        model = pickle.load(open(models_folder + "modelo_GPz_color_" + str(i+1) + ".pkl", "rb"))
        models.append(model)

    return models

def main(DR2_folder, models_folder, number_sim, outpath, core, modulus):
    """
    """
    assert os.path.exists(DR2_folder), "DR2 catalog folder doesn't exist."
    assert os.path.exists(models_folder), "models catalog folder doesn't exist."
    if not os.path.exists(os.path.dirname(outpath)):
        os.mkdir(os.path.dirname(outpath))

    number_sim = int(number_sim)
    models = load_models(number_sim, models_folder)

    core = int(core, base=10)
    modulus = int(modulus, base=10)

    filename = os.listdir(DR2_folder)
    for i in range(len(filename)):
        if i%core == modulus:
            start_time = time.time()
            data_aux = Table.read(os.path.join(DR2_folder, filename[i])).to_pandas()
            data = remove_outliers(data_aux)

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

            dataset_output = pd.DataFrame()
            dataset_output["COADD_OBJECT_ID"] = data["COADD_OBJECT_ID"]

            mu_GPz = []
            mu_GPz_sigma = []

            for j in range(number_sim):
                mu, sigma, modelV,noiseV, phi = models[j].predict(X[:,:].copy(),model='best')
                zphot = mu.flatten()
                dataset_output['GPz_color:zphot_' + str(j+1)] = zphot
                dataset_output['GPz_color:sigma_' + str(j+1)] = sigma.flatten()
                mu_GPz.append(zphot)
                mu_GPz_sigma.append(sigma.flatten())

            phi = pd.DataFrame(phi)
            phi.to_csv('/home/afarias/phi_GPzC.csv', index = False)
            mean_GPz = pd.Series(np.round(sum(np.array(mu_GPz)/10),6), name="mean_GPz")
            standard_dev_GPz = pd.Series(np.round(np.std(np.array(mu_GPz), axis=0),6), name="StanDev_GPz
")
            mean_GPz_sigma = pd.Series(np.round(sum(np.array(mu_GPz_sigma)/10),6), name="mean_GPz_sigma"
)
            standard_dev_GPz_sigma = pd.Series(np.round(np.std(np.array(mu_GPz_sigma), axis=0),6), name=
"StanDev_GPz_sigma")

            dataset_output["Mean_GPz_color"] = mean_GPz
            dataset_output["StanDev_GPz_color"] = standard_dev_GPz
            dataset_output["Mean_GPz_sigma_color"] = mean_GPz_sigma
            dataset_output["StanDev_GPz_sigma_color"] = standard_dev_GPz_sigma

            name = filename[i][:17] + "_GPz_color" + filename[i][17:]
            print(outpath + str(name))
            dataset_output.to_csv(outpath + str(name),index = False)
            print("--- %s seconds --" % (time.time() - start_time))

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
    parser.add_argument('--core')
    parser.add_argument('--modulus')

    args = parser.parse_args()
    print('DR2 folder:', args.DR2_folder)
    print('models catalog:', args.models_folder)
    print('Number of simulations:', args.number_sim)
    print('core:', args.core)
    print('mod:', args.modulus)
    main(args.DR2_folder, args.models_folder, args.number_sim, args.outpath, args.core, args.modulus)
