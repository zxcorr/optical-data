########################################################################################################
##############################################################################
# This script computes the cuts of the KdTree
########################################################################################################
##############################################################################

import numpy as np
import pandas as pd
from astropy.table import Table
import os
import pickle
import argparse


def data_processing(data):
    feat = ['MAG_AUTO_G_DERED', 'MAG_AUTO_R_DERED', 'MAG_AUTO_I_DERED', 'MAG_AUTO_Z_DERED', 'MAG_AUTO_Y_
DERED']
    values_mags = [38.06618881225586, 36.027408599853516, 23.97100067138672, 32.51519012451172, 35.71221
160888672]

    data.loc[data[feat[0]]>90,feat[0]] = data[data[feat[0]]<90][feat[0]].max()
    data.loc[data[feat[1]]>90,feat[1]] = data[data[feat[1]]<90][feat[1]].max()
    data.loc[data[feat[2]]>90,feat[2]] = data[data[feat[2]]<90][feat[2]].max()
    data.loc[data[feat[3]]>90,feat[3]] = data[data[feat[3]]<90][feat[3]].max()
    data.loc[data[feat[4]]>90,feat[4]] = data[data[feat[4]]<90][feat[4]].max()

     # if the data has nan values
    data[feat[0]] = data[feat[0]].replace(np.nan, values_mags[0])
    data[feat[1]] = data[feat[1]].replace(np.nan, values_mags[1])
    data[feat[2]] = data[feat[2]].replace(np.nan, values_mags[2])
    data[feat[3]] = data[feat[3]].replace(np.nan, values_mags[3])
    data[feat[4]] = data[feat[4]].replace(np.nan, values_mags[4])

    return data

def models_kdtree():
    kdtree_ponto_mais_perto = pickle.load(open("/media/BINGODATA0/optical-data/DES_processed_data/DES_DR
2_processed_KdTree_cuts/closest_point_function.pkl", "rb"))
    kdtree_real = pickle.load(open("/media/BINGODATA0/optical-data/DES_processed_data/DES_DR2_processed_
KdTree_cuts/KdTree.pkl", "rb"))
    grupos = pickle.load(open("/media/BINGODATA0/optical-data/DES_processed_data/DES_DR2_processed_KdTre
e_cuts/color_group.pkl", "rb"))
    kdtree_ponto_mais_perto_mag = pickle.load(open("/media/BINGODATA0/optical-data/DES_processed_data/DE
S_DR2_processed_KdTree_cuts/closest_point_function_mag.pkl", "rb"))
    kdtree_real_mag = pickle.load(open("/media/BINGODATA0/optical-data/DES_processed_data/DES_DR2_proces
sed_KdTree_cuts/KdTree_mag.pkl", "rb"))
    grupos_mag = pickle.load(open("/media/BINGODATA0/optical-data/DES_processed_data/DES_DR2_processed_K
dTree_cuts/color_group_mag.pkl", "rb"))
    rede_neural = pickle.load(open("/media/BINGODATA0/optical-data/DES_processed_data/DES_DR2_processed_
KdTree_cuts/ML_lambda.pkl", "rb"))
    rede_neural_mag = pickle.load(open("/media/BINGODATA0/optical-data/DES_processed_data/DES_DR2_proces
sed_KdTree_cuts/ML_lambda_mag.pkl", "rb"))

    return kdtree_ponto_mais_perto, kdtree_real, grupos, kdtree_ponto_mais_perto_mag, kdtree_real_mag, g
rupos_mag, rede_neural, rede_neural_mag


def main(DR2_folder, core, modulus):
    """
    """
    assert os.path.exists(DR2_folder), "DES catalog folder doesn't exist."

    kdtree_ponto_mais_perto, kdtree_real, grupos, kdtree_ponto_mais_perto_mag, kdtree_real_mag, grupos_m
ag, rede_neural, rede_neural_mag = models_kdtree()

    filename = os.listdir(DR2_folder)
    core = int(core, base=10)
    modulus = int(modulus, base=10)
    novo = 5050


    for i in range(len(filename)-novo):
        #print(type(core))
        #print(type(modulus))
        if i%core==modulus:
            print(os.path.join(DR2_folder, filename[i + novo]))
            #print(i + novo, "\n")
            df = Table.read(os.path.join(DR2_folder, filename[i + novo])).to_pandas()
            data = data_processing(df)

            dataset_output = pd.DataFrame()
            dataset_output["COADD_OBJECT_ID"] = data["COADD_OBJECT_ID"]
            dataset_output["TILENAME"] = data["TILENAME"]
            dataset_output["HPIX_32"] = data["HPIX_32"]
            dataset_output["HPIX_64"] = data["HPIX_64"]
            dataset_output["HPIX_1024"] = data["HPIX_1024"]
            dataset_output["HPIX_4096"] = data["HPIX_4096"]
            dataset_output["HPIX_16384"] = data["HPIX_16384"]
            dataset_output["RA"] = data["RA"]
            dataset_output["DEC"] = data["DEC"]
            dataset_output["MAG_AUTO_G_DERED"] = data["MAG_AUTO_G_DERED"]
            dataset_output["MAG_AUTO_R_DERED"] = data["MAG_AUTO_R_DERED"]
            dataset_output["MAG_AUTO_I_DERED"] = data["MAG_AUTO_I_DERED"]
            dataset_output["MAG_AUTO_Z_DERED"] = data["MAG_AUTO_Z_DERED"]
            dataset_output["MAG_AUTO_Y_DERED"] = data["MAG_AUTO_Y_DERED"]

            #KdTree
            cores_des = pd.DataFrame()
            cores_des["G-R"] = dataset_output["MAG_AUTO_G_DERED"] - dataset_output["MAG_AUTO_R_DERED"]
            cores_des["R-I"] = dataset_output["MAG_AUTO_R_DERED"] - dataset_output["MAG_AUTO_I_DERED"]
            cores_des["I-Z"] = dataset_output["MAG_AUTO_I_DERED"] - dataset_output["MAG_AUTO_Z_DERED"]
            cores_des["Z-Y"] = dataset_output["MAG_AUTO_Z_DERED"] - dataset_output["MAG_AUTO_Y_DERED"]
            cores_des["I"]   = dataset_output["MAG_AUTO_I_DERED"]

            #print('folha')
            folha = []
            folha_mag = []

            for m in range(len(cores_des)):
                ponto_perto = kdtree_ponto_mais_perto(kdtree_real, np.array(cores_des.iloc[m,0:4]))
                folha.append(np.where(ponto_perto == grupos)[0][0])
                ponto_perto = kdtree_ponto_mais_perto_mag(kdtree_real_mag, np.array(cores_des.iloc[m,0:5
]))
                folha_mag.append(np.where(ponto_perto == grupos_mag)[0][0])

            lambda_ = rede_neural(cores_des.iloc[:,0:4])
            lambda_mag = rede_neural_mag(cores_des.iloc[:,0:5])

            #adicionando as novas colunas
            dataset_output["folha"] = folha
            dataset_output["lambda"] = lambda_
            dataset_output["folha_mag"] = folha_mag
            dataset_output["lambda_mag"] = lambda_mag


            nome = filename[i + novo][:17] + "_KdTree.csv"
            dataset_output.to_csv("/media/BINGODATA0/optical-data/DES_processed_data/DES_DR2_processed_K
dTree_cuts/"+str(nome),index = False)


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

    parser.add_argument('--folder')
    parser.add_argument('--core')
    parser.add_argument('--modulus')

    args = parser.parse_args()
    print(args.folder, args.core, args.modulus)
    main(args.folder, args.core, args.modulus)
