#!/usr/bin/env python
# coding: utf-8

import healpy as hp
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

###########   GPZ     ####################
print("Reading GPz Map Nside 4096")

DES_DR2_GPz=hp.read_map('hp_map_DR2_GPZ_np10_6k_nocuts_4096_N4096_z0.00-100000000.00.fits')
DES_DR2_GPz.shape

hp.mollview(DES_DR2_GPz, title='DES DR2 GPz \n NSIDE=4096 z-0.00- infinity', cmap='viridis',nest=True)
plt.savefig('DES_DR2_GPz_4096.png', dpi=200, bbox_inches='tight', pad_inches=0.5)

#plt.hist(DES_DR2_GPz, bins=1000,alpha=0.5, label='DES_DR2_GPz')
#plt.legend()
#plt.xlim(0.8, 1.02)
#plt.ylim(0.9, 7000)
#plt.savefig('DES_DR2_GPz_4096Hist.png', dpi=200, bbox_inches='tight', pad_inches=0.5)

#plt.hist(DES_DR2_GPz, bins=1000)
#plt.xlim(-0.10, 1.02)
#plt.ylim(0.0, 100)
#plt.savefig('DES_DR2_GPz_4096HistZoom.png', dpi=200, bbox_inches='tight', pad_inches=0.5)

hp.mollzoom(DES_DR2_GPz,xsize=100,nest=True)
plt.savefig('DES_DR2_GPz_4096_mollzoom.png', dpi=200, bbox_inches='tight', pad_inches=0.5)


########### DES MASK G   ####################

print("Reading DES Mask G Nside 4096")

DES_MASK_DR2_G =hp.read_map('DES_MASK_DR2_G.fits')
DES_MASK_DR2_G.shape
hp.mollview(DES_MASK_DR2_G, title='DES MASK DR2 G \n NSIDE=4096', cmap='viridis', nest=True)
plt.savefig('DES_MASK_DR2_G_4096.png', dpi=200, bbox_inches='tight', pad_inches=0.5)


#plt.hist(DES_MASK_DR2_G, bins=1000,alpha=0.5, label='DES_MASK_DR2_G')
#plt.legend()
#plt.xlim(0.8, 1.02)
#plt.ylim(0.0, 7000)
#plt.savefig('DES_Mask_DR2_G_4096_Hist.png', dpi=200, bbox_inches='tight', pad_inches=0.5)

#plt.hist(DES_MASK_DR2_G, bins=1000)
#plt.xlim(-0.10, 1.02)
#plt.ylim(0.0, 100)
#plt.savefig('DES_Mask_DR2_G_4096_Hist_zoom.png', dpi=200, bbox_inches='tight', pad_inches=0.5)


hp.mollzoom(DES_MASK_DR2_G,xsize=100,nest=True)
plt.savefig('DES_Mask_DR2_G_4096_mollzoom.png', dpi=200, bbox_inches='tight', pad_inches=0.5)


########### DES MASK I  ####################

print("Reading DES Mask I Nside 4096")

DES_MASK_DR2_I =hp.read_map('DES_MASK_DR2_I.fits')
DES_MASK_DR2_I.shape
hp.mollview(DES_MASK_DR2_I, title='DES MASK DR2 I \n NSIDE=4096', cmap='viridis', nest=True)
plt.savefig('DES_MASK_DR2_I_4096.png', dpi=200, bbox_inches='tight', pad_inches=0.5)

#plt.hist(DES_MASK_DR2_I, bins=1000,alpha=0.5, label='DES_MASK_DR2_I')
#plt.legend()
#plt.xlim(0.8, 1.02)
#plt.ylim(0.0, 7000)
#plt.savefig('DES_Mask_DR2_I_4096_Hist.png', dpi=200, bbox_inches='tight', pad_inches=0.5)

#plt.hist(DES_MASK_DR2_I, bins=1000)
#plt.xlim(-0.10, 1.02)
#plt.ylim(0.0, 100)
#plt.savefig('DES_Mask_DR2_I_4096_Hist_zoom.png', dpi=200, bbox_inches='tight', pad_inches=0.5)

hp.mollzoom(DES_MASK_DR2_I,xsize=100,nest=True)
plt.savefig('DES_Mask_DR2_I_4096_mollzoom.png', dpi=200, bbox_inches='tight', pad_inches=0.5)

########### DES MASK R  ####################

print("Reading DES Mask R Nside 4096")

DES_MASK_DR2_R =hp.read_map('DES_MASK_DR2_R.fits')
DES_MASK_DR2_R.shape
hp.mollview(DES_MASK_DR2_R, title='DES MASK DR2 R \n NSIDE=4096', cmap='viridis', nest=True)
plt.savefig('DES_MASK_DR2_R_4096.png', dpi=200, bbox_inches='tight', pad_inches=0.5)


#plt.hist(DES_MASK_DR2_R, bins=1000,alpha=0.5, label='DES_MASK_DR2_R')
#plt.legend()
#plt.xlim(0.8, 1.02)
#plt.ylim(0.0, 7000)
#plt.savefig('DES_Mask_DR2_R_4096_Hist.png', dpi=200, bbox_inches='tight', pad_inches=0.5)


#plt.hist(DES_MASK_DR2_R, bins=1000)
#plt.xlim(-0.10, 1.02)
#plt.ylim(0.0, 100)
#plt.savefig('DES_Mask_DR2_R_4096_Hist_zoom.png', dpi=200, bbox_inches='tight', pad_inches=0.5)

hp.mollzoom(DES_MASK_DR2_R,xsize=100,nest=True)
plt.savefig('DES_Mask_DR2_R_4096_mollzoom.png', dpi=200, bbox_inches='tight', pad_inches=0.5)

########### DES MASK Y  ####################

print("Reading DES Mask Y Nside 4096")

DES_MASK_DR2_Y =hp.read_map('DES_MASK_DR2_Y.fits')
DES_MASK_DR2_Y.shape
hp.mollview(DES_MASK_DR2_Y, title='DES MASK DR2 Y \n NSIDE=4096', cmap='viridis', nest=True)
plt.savefig('DES_MASK_DR2_Y_4096.png', dpi=200, bbox_inches='tight', pad_inches=0.5)

#plt.hist(DES_MASK_DR2_Y, bins=1000,alpha=0.5, label='DES_MASK_DR2_Y')
#plt.legend()
#plt.xlim(0.8, 1.02)
#plt.ylim(0.0, 7000)
#plt.savefig('DES_MASK_DR2_Y_4096_Hist.png', dpi=200, bbox_inches='tight', pad_inches=0.5)

#plt.hist(DES_MASK_DR2_Y, bins=1000)
#plt.xlim(-0.10, 1.02)
#plt.ylim(0.0, 100)
#plt.savefig('DES_MASK_DR2_Y_4096_Hist_zoom.png', dpi=200, bbox_inches='tight', pad_inches=0.5)

hp.mollzoom(DES_MASK_DR2_Y,xsize=100,nest=True)
plt.savefig('DES_MASK_DR2_Y_4096_mollzoom.png', dpi=200, bbox_inches='tight', pad_inches=0.5)


########### DES MASK Z  ####################

print("Reading DES Mask Z Nside 4096")

DES_MASK_DR2_Z =hp.read_map('DES_MASK_DR2_Z.fits')
DES_MASK_DR2_Z.shape
hp.mollview(DES_MASK_DR2_Z, title='DES MASK DR2 Z \n NSIDE=4096', cmap='viridis', nest=True)
plt.savefig('DES_MASK_DR2_Z.png', dpi=200, bbox_inches='tight', pad_inches=0.5)

#plt.hist(DES_MASK_DR2_Z, bins=1000,alpha=0.5, label='DES_MASK_DR2_Z')
#plt.legend()
#plt.xlim(0.8, 1.02)
#plt.ylim(0.0, 7000)
#plt.savefig('DES_MASK_DR2_Z_Hist.png', dpi=200, bbox_inches='tight', pad_inches=0.5)

#plt.hist(DES_MASK_DR2_Z, bins=1000)
#plt.xlim(-0.10, 1.02)
#plt.ylim(0.0, 100)
#plt.savefig('DES_MASK_DR2_Z_Hist_zoom.png', dpi=200, bbox_inches='tight', pad_inches=0.5)

hp.mollzoom(DES_MASK_DR2_Z,xsize=100,nest=True)
plt.savefig('DES_MASK_DR2_Z_mollzoom.png', dpi=200, bbox_inches='tight', pad_inches=0.5)


########## Mask Final ######################

print("Creating mask with cut > 0.9")

DES_DR2_GPz=hp.read_map('hp_map_DR2_GPZ_np10_6k_nocuts_4096_N4096_z0.00-100000000.00.fits')

mask_final=((DES_DR2_GPz > 0.9)& (DES_MASK_DR2_G > 0.9) & (DES_MASK_DR2_I > 0.9) & (DES_MASK_DR2_R > 0.9) & (DES_MASK_DR2_Y > 0.9) & (DES_MASK_DR2_Z > 0.9 ))

hp.mollview(mask_final, title='DES MASK Final > 0.9 \n NSIDE=4096', cmap='viridis', nest=True)
hp.write_map('mask_final_cut09_nside4096.fits', mask_final, nest=True)
plt.savefig('DES_MASK_Final.png', dpi=200, bbox_inches='tight', pad_inches=0.5)


print("Creating Histograms")
#plt.hist(mask_final, bins=1000,alpha=0.5, label='Full mask')
#plt.legend()
#plt.ylim(0,10000)
#plt.xlim(0.8,1.02)
#plt.savefig('DES_MASK_Final_Hist.png', dpi=200, bbox_inches='tight', pad_inches=0.5)

#plt.hist(mask_final, bins=1000,alpha=0.5, label='Full mask')
#plt.legend()
#plt.ylim(0,250)
#plt.xlim(0.8,1.02)
#plt.savefig('DES_MASK_Final_Histzoom.png', dpi=200, bbox_inches='tight', pad_inches=0.5)

hp.mollzoom(mask_final,xsize=100,nest=True)
plt.savefig('mask_final_mollzoom.png', dpi=200, bbox_inches='tight', pad_inches=0.5)


############################ MASK AVERAGE ############################

print("Creating mask DES average")

maskDES_avg = 1/5*(DES_DR2_GPz  + DES_MASK_DR2_G + DES_MASK_DR2_I + DES_MASK_DR2_R +  DES_MASK_DR2_Y + DES_MASK_DR2_Z)
hp.mollview(maskDES_avg, title=" DES Mask Average \n NSIDE =4096", nest=True)
#hp.graticule()
hp.write_map('mask_final_average_nside4096.fits', mask_final, nest=True)
plt.savefig('mask_final_average_nside4096.png', dpi=200, bbox_inches='tight', pad_inches=0.5)

hp.mollzoom(maskDES_avg, xsize=1000, nest=True)
plt.savefig('mask_final_average_nside4096_mollzoom.png', dpi=200, bbox_inches='tight', pad_inches=0.5)

#plt.hist(maskDES_avg, bins=1000,alpha=0.5, label='Full mask')
#plt.legend()
#plt.ylim(0,10000)
#plt.xlim(0.8,1.02)
#plt.savefig('DES_MASK_Average_Hist.png', dpi=200, bbox_inches='tight', pad_inches=0.5)

#plt.hist(maskDES_avg, bins=1000,alpha=0.5, label='Full mask')
#plt.legend()
#plt.ylim(0,250)
#plt.xlim(0.8,1.02)
#plt.savefig('DES_MASK_Average_Histzoom.png', dpi=200, bbox_inches='tight', pad_inches=0.5)

#figure(figsize = (10, 5), dpi = 80)
#plt.hist(DES_DR2_GPz,    bins=1000,alpha=0.25, label='DES_DR2_GPz')
#plt.hist(DES_MASK_DR2_G, bins=1000,alpha=0.25, label='DES_MASK_DR2_G')
#plt.hist(DES_MASK_DR2_R, bins=1000,alpha=0.25, label='DES_MASK_DR2_R')
#plt.hist(DES_MASK_DR2_Y, bins=1000,alpha=0.25, label='DES_MASK_DR2_Y')
#plt.hist(DES_MASK_DR2_Z, bins=1000,alpha=0.25, label='DES_MASK_DR2_Z')
#plt.hist(DES_MASK_DR2_I, bins=1000,alpha=0.25, label='DES_MASK_DR2_I')
#plt.hist(mask_final,     bins=1000,alpha=0.5, label='DES DR2 MASK Final', hatch="/")
#plt.hist(maskDES_avg,    bins=1000,alpha=0.5, label='DES DR2 MASK Average', hatch="*")

#plt.ylim(0,7000)
#plt.xlim(0.8,1.02)

#plt.legend()
#plt.savefig('DES_MASKS_Hist.png', dpi=200, bbox_inches='tight', pad_inches=0.5)
