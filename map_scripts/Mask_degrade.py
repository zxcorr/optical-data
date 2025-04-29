import healpy as hp
import numpy as np
from numpy import inf
from astropy.io import fits
import matplotlib.pyplot as plt

print("start")


print("reading DES_MASK_DR2_G")

DES_MASK_DR2_G =hp.read_map('DES_MASK_DR2_G.fits')
DES_MASK_DR2_G.shape

hp.mollview(DES_MASK_DR2_G, title='DES MASK DR2 G \n NSIDE=4096', cmap='viridis', norm="hist", nest=True)

DES_MASK_DR2_G[DES_MASK_DR2_G == -inf] = 0
DES_MASK_DR2_G[DES_MASK_DR2_G == inf] = 0

DES_MASK_DR2_G_4096 = hp.ud_grade(DES_MASK_DR2_G, 256, order_in='NEST')
hp.write_map("DES_MASK_DR2_G_256.fits", DES_MASK_DR2_G_4096)



print("reading DES_MASK_DR2_I")

DES_MASK_DR2_I =hp.read_map('DES_MASK_DR2_I.fits')
DES_MASK_DR2_I.shape

hp.mollview(DES_MASK_DR2_I, title='DES MASK DR2 I \n NSIDE=4096', cmap='viridis', norm="hist", nest=True)

DES_MASK_DR2_I[DES_MASK_DR2_I == -inf] = 0
DES_MASK_DR2_I[DES_MASK_DR2_I == inf] = 0

DES_MASK_DR2_I_4096 = hp.ud_grade(DES_MASK_DR2_I, 256,order_in='NEST')
hp.write_map("DES_MASK_DR2_I_256.fits", DES_MASK_DR2_I_4096)



print("reading DES_MASK_DR2_R")

DES_MASK_DR2_R =hp.read_map('DES_MASK_DR2_R.fits')
DES_MASK_DR2_R.shape

hp.mollview(DES_MASK_DR2_R, title='DES MASK DR2 R \n NSIDE=4096', cmap='viridis', norm="hist", nest=True)

DES_MASK_DR2_R[DES_MASK_DR2_R == -inf] = 0
DES_MASK_DR2_R[DES_MASK_DR2_R == inf] = 0

DES_MASK_DR2_R_4096 = hp.ud_grade(DES_MASK_DR2_R, 256,order_in='NEST')
hp.write_map("DES_MASK_DR2_R_256.fits", DES_MASK_DR2_R_4096)


print("reading DES_MASK_DR2_Y")

DES_MASK_DR2_Y =hp.read_map('DES_MASK_DR2_Y.fits')
DES_MASK_DR2_Y.shape

hp.mollview(DES_MASK_DR2_Y, title='DES MASK DR2 Y \n NSIDE=4096', cmap='viridis', norm="hist", nest=True)

DES_MASK_DR2_Y[DES_MASK_DR2_Y == -inf] = 0
DES_MASK_DR2_Y[DES_MASK_DR2_Y == inf] = 0

DES_MASK_DR2_Y_4096 = hp.ud_grade(DES_MASK_DR2_Y, 256,order_in='NEST')
hp.write_map("DES_MASK_DR2_Y_256.fits", DES_MASK_DR2_Y_4096)


print("reading DES_MASK_DR2_Z")

DES_MASK_DR2_Z =hp.read_map('DES_MASK_DR2_Z.fits')
DES_MASK_DR2_Z.shape

hp.mollview(DES_MASK_DR2_Z, title='DES MASK DR2 Z \n NSIDE=4096', cmap='viridis', norm="hist", nest=True)

DES_MASK_DR2_Z[DES_MASK_DR2_Z == -inf] = 0
DES_MASK_DR2_Z[DES_MASK_DR2_Z == inf] = 0

DES_MASK_DR2_Z_4096 = hp.ud_grade(DES_MASK_DR2_Z, 256,order_in='NEST')
hp.write_map("DES_MASK_DR2_Z_256.fits", DES_MASK_DR2_Z_4096)



DES_DR2_GPz=hp.read_map('hp_map_DR2_GPZ_np10_6k_nocuts_4096_N4096_z0.00-100000000.00.fits')
DES_DR2_GPz.shape


DES_DR2_GPz[DES_DR2_GPz == -inf] = 0
DES_DR2_GPz[DES_DR2_GPz == inf] = 0

DES_MASK_DR2_Z_4096 = hp.ud_grade(DES_MASK_DR2_Z, 256,order_in='NEST')
hp.write_map("DES_MASK_GPz_256.fits", DES_MASK_DR2_Z_4096)




