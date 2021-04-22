import numpy as np
import pandas as pd
from astroquery.ned import Ned
from astroquery.simbad import Simbad
from astropy.table import Table,QTable
import astropy.units as u
from astropy import coordinates
import healpy as hp

vvds = pd.read_csv("data/vvds-not.csv")

RA = vvds["RA"].values
DEC = vvds["DEC"].values

pixel = hp.ang2pix(64,RA,DEC,lonlat=True)
pixel = np.unique(pixel)
pixel = pixel.reshape(-1,1)
ang = hp.pix2ang(64,pixel,lonlat=True)
RA = ang[0]
DEC = ang[1]

# NED Survey
l1 = []
for i in range(len(RA)): 
    co = coordinates.SkyCoord(ra = RA[i],dec= DEC[i],unit=(u.deg,u.deg))
    ned_table = Ned.query_region(co, radius=0.1 * u.deg)
    df = ned_table.to_pandas()
    l1.append(df)
ned = pd.concat(l1, axis=0, ignore_index=True)
ned.to_csv("data/ned.csv")
# Simbad Survey
l2 = []
for i in range((len(RA))):
    co = coordinates.SkyCoord(ra = RA[i],dec= DEC[i],unit=(u.deg,u.deg))
    simbad_table = Simbad.query_region(co, radius=0.1* u.deg) 
    df = simbad_table.to_pandas()
    l2.append(df)
simbad = pd.concat(l1, axis=0, ignore_index=True)
simbad.to_csv("data/simbad.csv")