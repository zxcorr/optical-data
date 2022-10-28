########################################################################################################
############################################################################################## LIBRARIES
 #######################################################################
########################################################################################################
####################################

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import healpy as hp
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#import seaborn as sns
import matplotlib
from keras import models, layers
from tensorflow import keras
import time
import copy
from statistics import variance
from os import walk
import math

#path
from astropy.table import Table
#importar o resto



########################################################################################################
############################################################################################## VIPERS ##
########################################################################
########################################################################################################
####################################

pathMatch = "/media/BINGODATA0/optical-data/SPEC_MATCH_DATA/DES_MATCHS_OZDESGRC/VIPERSxDES_DR1.fits"
match = Table.read(pathMatch)
match = match.to_pandas()
match = match.drop_duplicates()
match_auto_G = match["MAG_AUTO_G_DERED"]
match_auto_R = match["MAG_AUTO_R_DERED"]
match_auto_Y = match["MAG_AUTO_Y_DERED"]
match_auto_I = match["MAG_AUTO_I_DERED"]
match_auto_Z = match["MAG_AUTO_Z_DERED"]
match_ID = match["COADD_OBJECT_ID"]
match_z = match["z"]

#Retirando as duplicadas do match.
ind_duplicadas = np.where(match_ID.duplicated(keep='first')==True)[0]

match_auto_G = match_auto_G.drop(ind_duplicadas)
match_auto_Y = match_auto_Y.drop(ind_duplicadas)
match_auto_R = match_auto_R.drop(ind_duplicadas)
match_auto_I = match_auto_I.drop(ind_duplicadas)
match_auto_Z = match_auto_Z.drop(ind_duplicadas)
match_z = match_z.drop(ind_duplicadas)
match_ID = match_ID.drop(ind_duplicadas)

match_auto_G = match_auto_G.reset_index(drop=True)
match_auto_Y = match_auto_Y.reset_index(drop=True)
match_auto_R = match_auto_R.reset_index(drop=True)
match_auto_I = match_auto_I.reset_index(drop=True)
match_auto_Z = match_auto_Z.reset_index(drop=True)
match_z = match_z.reset_index(drop=True)
match_ID = match_ID.reset_index(drop=True)


########################################################################################################
############################################################################################## DES #####
######################################################################
########################################################################################################
###################################

match_pixel = match['HPIX_64']
match_pixel = match_pixel.drop_duplicates()
match_pixel = hp.nest2ring(64, match_pixel)
match_pixel = np.asarray(match_pixel)
filename = []
for i in range(len(match_pixel)):
    filename.append("PixelFit_64_"+str(match_pixel[i])+".fits")

pathDES = "/media/BINGODATA0/optical-data/DES_original_data/DES_DR1_64_Pixels/"+str(filename[0])
des = Table.read(pathDES)
des = des.to_pandas()
for i in range(len(filename)-1):
    pathDES = "/media/BINGODATA0/optical-data/DES_original_data/DES_DR1_64_Pixels/"+str(filename[i+1])
    des_aux = Table.read(pathDES)
    des_aux = des_aux.to_pandas()
    des = des.append(des_aux,ignore_index = True)

des_auto_G = des["MAG_AUTO_G_DERED"]
des_auto_R = des["MAG_AUTO_R_DERED"]
des_auto_Y = des["MAG_AUTO_Y_DERED"]
des_auto_I = des["MAG_AUTO_I_DERED"]
des_auto_Z = des["MAG_AUTO_Z_DERED"]
des_ID = des["COADD_OBJECT_ID"]

def tirando_os_extremos (vetor):
    VetorSemEX = vetor.copy()
    VetorSemEX.drop(VetorSemEX[VetorSemEX>=90].index,inplace = True)
    vetor =  np.where(vetor>=90,max(VetorSemEX),vetor)
    #print("Máximo valor do filtro:",max(vetor))
    return vetor

des_auto_G = pd.Series(des_auto_G, name='MAG_AUTO_G_DERED')
des_auto_R = pd.Series(des_auto_R, name='MAG_AUTO_R_DERED')
des_auto_I = pd.Series(des_auto_I, name='MAG_AUTO_I_DERED')
des_auto_Z = pd.Series(des_auto_Z, name='MAG_AUTO_Z_DERED')
des_auto_Y = pd.Series(des_auto_Y, name='MAG_AUTO_Y_DERED')

match_auto_G = pd.Series(match_auto_G, name='MAG_AUTO_G_DERED')
match_auto_R = pd.Series(match_auto_R, name='MAG_AUTO_R_DERED')
match_auto_I = pd.Series(match_auto_I, name='MAG_AUTO_I_DERED')
match_auto_Z = pd.Series(match_auto_Z, name='MAG_AUTO_Z_DERED')
match_auto_Y = pd.Series(match_auto_Y, name='MAG_AUTO_Y_DERED')


########################################################################################################
############################################################################################## DATA PROC
ESSING ###############################################################
########################################################################################################
###################################

#### DES
des_auto_G = tirando_os_extremos(des_auto_G)
des_auto_R = tirando_os_extremos(des_auto_R)
des_auto_I = tirando_os_extremos(des_auto_I)
des_auto_Z = tirando_os_extremos(des_auto_Z)
des_auto_Y = tirando_os_extremos(des_auto_Y)
#### MATCH
match_auto_G = tirando_os_extremos(match_auto_G)
match_auto_R = tirando_os_extremos(match_auto_R)
match_auto_I = tirando_os_extremos(match_auto_I)
match_auto_Z = tirando_os_extremos(match_auto_Z)
match_auto_Y = tirando_os_extremos(match_auto_Y)


########################################################################################################
############################################################################################## GALAXY'S
MATCH ###########################################################################
########################################################################################################
######################################

ID_TF = des_ID.isin(match_ID)
ID_bi =  np.where(ID_TF==True,1,0)
ID_bi = pd.Series(ID_bi, name='ID_bi')

#making the dataset with DES colors.
cores_des = pd.DataFrame()
cores_des["G-R"] = des_auto_G - des_auto_R
cores_des["R-I"] = des_auto_R - des_auto_I
cores_des["I-Z"] = des_auto_I - des_auto_Z
cores_des["Z-Y"] = des_auto_Z - des_auto_Y

cores_vipers = pd.DataFrame()
cores_vipers["G-R"] = match_auto_G - match_auto_R
cores_vipers["R-I"] = match_auto_R - match_auto_I
cores_vipers["I-Z"] = match_auto_I - match_auto_Z
cores_vipers["Z-Y"] = match_auto_Z - match_auto_Y

#Preparando o dataset
ID_bi_1 = np.where(ID_bi==1)[0]
ID_bi_0 = np.where(ID_bi==0)[0][0:47646]
ID_bi_pos = np.concatenate((ID_bi_1, ID_bi_0), axis=0)

X = pd.DataFrame()
X["G-R"] = cores_des["G-R"].iloc[ID_bi_pos]
X["R-I"] = cores_des["R-I"].iloc[ID_bi_pos]
X["I-Z"] = cores_des["I-Z"].iloc[ID_bi_pos]
X["Z-Y"] = cores_des["Z-Y"].iloc[ID_bi_pos]
X["I"] = des_auto_I[ID_bi_pos]
X["match"] = ID_bi[ID_bi_pos]

X = X.sample(frac=1).reset_index(drop=True)
#print(X)


########################################################################################################
############################################################################################## ML ######
#####################################################################
########################################################################################################
###################################
model = KNeighborsClassifier(n_neighbors=4, weights= 'distance')
treino_in, teste_in, treino_out, teste_out = train_test_split(X.iloc[:,0:4], X.iloc[:,5], test_size=0.3,
 random_state=None)
model.fit(treino_in, treino_out)
def rede_neural(total):
    predito = []
    for i in range(10):
        y_pred = model.predict_proba(total)
        predito.append(y_pred[:,1])
    predito_final = np.mean(predito,axis=0)
    return predito_final


model_mag = KNeighborsClassifier(n_neighbors=5, weights= 'distance')
treino_in, teste_in, treino_out, teste_out = train_test_split(X.iloc[:,0:5], X.iloc[:,5], test_size=0.3,
 random_state=None)
model_mag.fit(treino_in, treino_out)
def rede_neural_mag(total):
    predito = []
    for i in range(10):
        y_pred = model_mag.predict_proba(total)
        predito.append(y_pred[:,1])
    predito_final = np.mean(predito,axis=0)
    return predito_final


########################################################################################################
############################################################################################## KdTREE FU
NCTIONS  ###########################################################################
########################################################################################################
######################################

#Calcula a distancia entre dois pontos em um espaço 4D.
def distancia(point1,point2):
    w1, x1, y1, z1  = point1
    w2, x2, y2, z2  = point2

    dw = w1 - w2
    dx = x1 - x2
    dy = y1 - y2
    dz = z1 - z2


    return math.sqrt(dw*dw + dx*dx + dy*dy + dz*dz)

def menor_distancia(pivot, p1, p2):
    if p1 is None:
        return p2

    if p2 is None:
        return p1

    d1 = distancia(pivot,p1)
    d2 = distancia(pivot,p2)

    if d1<d2:
        return p1
    else:
        return p2

def kdtree_ponto_mais_perto(root, ponto, depth=0):
    if root is None:
        return None

    axis = depth % k

    proximo_no = None
    no_oposto = None

    if ponto[axis]< root['point'][axis]:
        proximo_no = root['left']
        no_oposto = root['right']
    else:
        proximo_no = root['right']
        no_oposto = root['left']

    melhor = menor_distancia(ponto,
                             kdtree_ponto_mais_perto(proximo_no, ponto, depth+1),
                             root['point'])

    if distancia(ponto,melhor)>abs(ponto[axis] - root['point'][axis]):
        melhor = menor_distancia(ponto,
                             kdtree_ponto_mais_perto(proximo_no, ponto, depth+1),
                             melhor)

    return melhor

#Número de eixos (cores) a serem cortados.
k=4
grupos = []

def construir_kdtree_test (pontos, depth=0):
    n = len(pontos)
    #print(n)
    if n<=0:
        return None

    #Para sempre cortar nos eixos certos, vamos fazer o mod de k
    axis = depth % k

    #A cada iteração eu 'corto' em uma direção (axis) diferente.
    #Então a cda iteração eu vou ordenando o dataset de acordo com a direção
    #que será cortada, para ter o mesmo
    #número de pontos nos dois lados da árvore.
    pontos_ordenados = sorted(pontos, key=lambda point: point[axis])

#----------------------------------------------------------------------------------
    #Como queremos uma árvore de 8 dimensões, vamos pegar todos os nós de quando
    #estivermos na octagésima ordem:
    #print(depth)
#1 tira o # e no segundo coloca
    if depth==12:
        #print('ue', pontos_ordenados[int(n/2)])
        grupos.append(pontos_ordenados[int(n/2)])
#-----------------------------------------------------------------------------------
    #print(pontos_ordenados)
    #construindo o nó.
    return {
        'point': pontos_ordenados[int(n/2)],
        'left': construir_kdtree_test(pontos_ordenados[:int(n/2)], depth+1),
        'right': construir_kdtree_test(pontos_ordenados[int(n/2)+1:], depth+1)
    }

def construir_kdtree (pontos, depth=0):
    n = len(pontos)

    if n<=0:
        return None

    #Para sempre cortar nos eixos certos, vamos fazer o mod de k
    axis = depth % k

    #A cada iteração eu 'corto' em uma direção (axis) diferente.
    #Então a cda iteração eu vou ordenando o dataset de acordo com a direção
    #que será cortada, para ter o mesmo
    #número de pontos nos dois lados da árvore.
    pontos_ordenados = sorted(pontos, key=lambda point: point[axis])

#----------------------------------------------------------------------------------
    #Como queremos uma árvore de 8 dimensões, vamos pegar todos os nós de quando
    #estivermos na octagésima ordem:
    #print(depth)
    #print(pontos_ordenados)
    #construindo o nó.
    return {
        'point': pontos_ordenados[int(n/2)],
        'left': construir_kdtree(pontos_ordenados[:int(n/2)], depth+1),
        'right': construir_kdtree(pontos_ordenados[int(n/2)+1:], depth+1)
    }

df = np.array(X.iloc[:,0:4])
kdtree = construir_kdtree_test(df)
print(grupos)
kdtree_real = construir_kdtree(grupos)


########################

#Calcula a distancia entre dois pontos em um espaço 5D.
def distancia_mag (point1,point2):
    w1, x1, y1, z1, a1  = point1
    w2, x2, y2, z2, a2  = point2
    #print(point1)
    #print(point2)

    dw = w1 - w2
    dx = x1 - x2
    dy = y1 - y2
    dz = z1 - z2
    da = a1 - a2

    return math.sqrt(dw*dw + dx*dx + dy*dy + dz*dz + da*da)

def menor_distancia_mag (pivot, p1, p2):
    if p1 is None:
        return p2

    if p2 is None:
        return p1

    d1 = distancia_mag(pivot,p1)
    d2 = distancia_mag(pivot,p2)

    if d1<d2:
        return p1
    else:
        return p2

def kdtree_ponto_mais_perto_mag(root, ponto, depth=0):
    if root is None:
        return None

    axis = depth % k

    proximo_no = None
    no_oposto = None

    if ponto[axis]< root['point'][axis]:
        proximo_no = root['left']
        no_oposto = root['right']
    else:
        proximo_no = root['right']
        no_oposto = root['left']

    melhor = menor_distancia_mag(ponto,
                             kdtree_ponto_mais_perto_mag(proximo_no, ponto, depth+1),
                             root['point'])

    if distancia_mag(ponto,melhor)>abs(ponto[axis] - root['point'][axis]):
        melhor = menor_distancia_mag(ponto,
                             kdtree_ponto_mais_perto_mag(proximo_no, ponto, depth+1),
                             melhor)

    return melhor

#Número de eixos (cores) a serem cortados.
k_mag=5
grupos_mag = []

def construir_kdtree_mag_test (pontos, depth=0):
    n = len(pontos)

    if n<=0:
        return None

    #Para sempre cortar nos eixos certos, vamos fazer o mod de k
    axis = depth % k

    #A cada iteração eu 'corto' em uma direção (axis) diferente.
    #Então a cda iteração eu vou ordenando o dataset de acordo com a direção
    #que será cortada, para ter o mesmo
    #número de pontos nos dois lados da árvore.
    pontos_ordenados = sorted(pontos, key=lambda point: point[axis])

#----------------------------------------------------------------------------------
    #Como queremos uma árvore de 8 dimensões, vamos pegar todos os nós de quando
    #estivermos na octagésima ordem:
    #print(depth)
#1 tira o # e no segundo coloca
    if depth==12:
        #print('coe', pontos_ordenados[int(n/2)])
        grupos_mag.append(pontos_ordenados[int(n/2)])
#-----------------------------------------------------------------------------------
    #print(pontos_ordenados)
    #construindo o nó.
    return {
        'point': pontos_ordenados[int(n/2)],
        'left': construir_kdtree_mag_test(pontos_ordenados[:int(n/2)], depth+1),
        'right': construir_kdtree_mag_test(pontos_ordenados[int(n/2)+1:], depth+1)
    }

def construir_kdtree_mag (pontos, depth=0):
    n = len(pontos)

    if n<=0:
        return None

    #Para sempre cortar nos eixos certos, vamos fazer o mod de k
    axis = depth % k

    #A cada iteração eu 'corto' em uma direção (axis) diferente.
    #Então a cda iteração eu vou ordenando o dataset de acordo com a direção
    #que será cortada, para ter o mesmo
    #número de pontos nos dois lados da árvore.
    pontos_ordenados = sorted(pontos, key=lambda point: point[axis])

#----------------------------------------------------------------------------------
    #Como queremos uma árvore de 8 dimensões, vamos pegar todos os nós de quando
    #estivermos na octagésima ordem:

    #print(pontos_ordenados)
    #construindo o nó.
    return {
        'point': pontos_ordenados[int(n/2)],
        'left': construir_kdtree_mag(pontos_ordenados[:int(n/2)], depth+1),
        'right': construir_kdtree_mag(pontos_ordenados[int(n/2)+1:], depth+1)
    }

X_mag = np.array(X.iloc[:,0:5])
kdtree_mag = construir_kdtree_mag_test(X_mag)

#grupo das 512 galáxias que separam todo o dataset.
print(grupos_mag)
#árvore utilizada para definir os 512 grupos de acordo com as 4 cores.
kdtree_real_mag = construir_kdtree(grupos_mag)

########################################################################################################
#########################################################################################################
######################################

def giving_variables():
    return kdtree_ponto_mais_perto, kdtree_real, grupos, kdtree_ponto_mais_perto_mag, kdtree_real_mag, g
rupos_mag, rede_neural, rede_neural_mag



