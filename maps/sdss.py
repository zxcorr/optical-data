from __future__ import division, print_function
import argparse
import os
import glob
import time
import numpy as np
import healpy as hp
import pandas as pd
from astropy.table import Table, join, vstack


def create_merged_gama_catalog(gamalist):
    ## ATTENTION!! THIS IS HARD-CODED AND WILL ONLY WORK FOR ANNZ REDSHIFTS
    colnames = ["objID", "ra", "dec", "ANNZ_best", "ANNZ_best_wgt", "ANNZ_best_err", "ANNZ_MLM_avg_0", "ANNZ_MLM_avg_0_err",
            "ANNZ_MLM_avg_0_wgt", "ANNZ_PDF_avg_0", "ANNZ_PDF_avg_0_err", "ANNZ_PDF_avg_0_wgt",
            "ANNZ_PDF_0_0", "ANNZ_PDF_0_1", "ANNZ_PDF_0_2", "ANNZ_PDF_0_3", "ANNZ_PDF_0_4", "ANNZ_PDF_0_5", "ANNZ_PDF_0_6",
            "ANNZ_PDF_0_7", "ANNZ_PDF_0_8", "ANNZ_PDF_0_9", "ANNZ_PDF_0_10", "ANNZ_PDF_0_11", "ANNZ_PDF_0_12", "ANNZ_PDF_0_13",
            "ANNZ_PDF_0_14", "ANNZ_PDF_0_15", "ANNZ_PDF_0_16", "ANNZ_PDF_0_17", "ANNZ_PDF_0_18", "ANNZ_PDF_0_19", "ANNZ_PDF_0_20",
            "ANNZ_PDF_0_21", "ANNZ_PDF_0_22", "ANNZ_PDF_0_23", "ANNZ_PDF_0_24", "ANNZ_PDF_0_25", "ANNZ_PDF_0_26", "ANNZ_PDF_0_27",
            "ANNZ_PDF_0_28", "ANNZ_PDF_0_29", "ANNZ_PDF_0_30", "ANNZ_PDF_0_31", "ANNZ_PDF_0_32", "ANNZ_PDF_0_33", "ANNZ_PDF_0_34",
            "ANNZ_PDF_0_35", "ANNZ_PDF_0_36", "ANNZ_PDF_0_37", "ANNZ_PDF_0_38", "ANNZ_PDF_0_39", "ANNZ_PDF_0_40", "ANNZ_PDF_0_41",
            "ANNZ_PDF_0_42", "ANNZ_PDF_0_43", "ANNZ_PDF_0_44", "ANNZ_PDF_0_45", "ANNZ_PDF_0_46", "ANNZ_PDF_0_47", "ANNZ_PDF_0_48",
            "ANNZ_PDF_0_49", "ANNZ_PDF_0_50", "ANNZ_PDF_0_51", "ANNZ_PDF_0_52", "ANNZ_PDF_0_53", "ANNZ_PDF_0_54", "ANNZ_PDF_0_55",
            "ANNZ_PDF_0_56", "ANNZ_PDF_0_57", "ANNZ_PDF_0_58", "ANNZ_PDF_0_59", "ANNZ_PDF_0_60", "ANNZ_PDF_0_61", "ANNZ_PDF_0_62",
            "ANNZ_PDF_0_63", "ANNZ_PDF_0_64", "ANNZ_PDF_0_65", "ANNZ_PDF_0_66", "ANNZ_PDF_0_67", "ANNZ_PDF_0_68", "ANNZ_PDF_0_69",
            "ANNZ_PDF_0_70", "ANNZ_PDF_0_71", "ANNZ_PDF_0_72", "ANNZ_PDF_0_73", "ANNZ_PDF_0_74", "ANNZ_PDF_0_75", "ANNZ_PDF_0_76",
            "ANNZ_PDF_0_77", "ANNZ_PDF_0_78", "ANNZ_PDF_0_79", "ANNZ_PDF_0_80", "ANNZ_PDF_0_81", "ANNZ_PDF_0_82", "ANNZ_PDF_0_83",
            "ANNZ_PDF_0_84", "ANNZ_PDF_0_85", "ANNZ_PDF_0_86", "ANNZ_PDF_0_87", "ANNZ_PDF_0_88", "ANNZ_PDF_0_89", "ANNZ_PDF_0_90",
            "ANNZ_PDF_0_91", "ANNZ_PDF_0_92", "ANNZ_PDF_0_93", "ANNZ_PDF_0_94", "ANNZ_PDF_0_95", "ANNZ_PDF_0_96", "ANNZ_PDF_0_97",
            "ANNZ_PDF_0_98", "ANNZ_PDF_0_99", "ANNZ_PDF_0_100", "ANNZ_PDF_0_101", "ANNZ_PDF_0_102", "ANNZ_PDF_0_103",
            "ANNZ_PDF_0_104", "ANNZ_PDF_0_105", "ANNZ_PDF_0_106", "ANNZ_PDF_0_107", "ANNZ_PDF_0_108", "ANNZ_PDF_0_109",
            "ANNZ_PDF_0_110", "ANNZ_PDF_0_111", "ANNZ_PDF_0_112", "ANNZ_PDF_0_113", "ANNZ_PDF_0_114", "ANNZ_PDF_0_115",
            "ANNZ_PDF_0_116", "ANNZ_PDF_0_117", "ANNZ_PDF_0_118", "ANNZ_PDF_0_119", "ANNZ_PDF_0_120", "ANNZ_PDF_0_121",
            "ANNZ_PDF_0_122", "ANNZ_PDF_0_123", "ANNZ_PDF_0_124", "ANNZ_PDF_0_125", "ANNZ_PDF_0_126", "ANNZ_PDF_0_127",
            "ANNZ_PDF_0_128", "ANNZ_PDF_0_129", "ANNZ_PDF_0_130", "ANNZ_PDF_0_131", "ANNZ_PDF_0_132", "ANNZ_PDF_0_133",
            "ANNZ_PDF_0_134", "ANNZ_PDF_0_135", "ANNZ_PDF_0_136", "ANNZ_PDF_0_137", "ANNZ_PDF_0_138", "ANNZ_PDF_0_139",
            "ANNZ_PDF_0_140", "ANNZ_PDF_0_141", "ANNZ_PDF_0_142", "ANNZ_PDF_0_143", "ANNZ_PDF_0_144", "ANNZ_PDF_0_145",
            "ANNZ_PDF_0_146", "ANNZ_PDF_0_147", "ANNZ_PDF_0_148", "ANNZ_PDF_0_149", "ANNZ_PDF_0_150", "ANNZ_PDF_0_151",
            "ANNZ_PDF_0_152", "ANNZ_PDF_0_153", "ANNZ_PDF_0_154", "ANNZ_PDF_0_155", "ANNZ_PDF_0_156", "ANNZ_PDF_0_157",
            "ANNZ_PDF_0_158", "ANNZ_PDF_0_159", "ANNZ_PDF_0_160", "ANNZ_PDF_0_161", "ANNZ_PDF_0_162", "ANNZ_PDF_0_163",
            "ANNZ_PDF_0_164", "ANNZ_PDF_0_165", "ANNZ_PDF_0_166", "ANNZ_PDF_0_167", "ANNZ_PDF_0_168", "ANNZ_PDF_0_169",
            "ANNZ_PDF_0_170", "ANNZ_PDF_0_171", "ANNZ_PDF_0_172", "ANNZ_PDF_0_173", "ANNZ_PDF_0_174", "ANNZ_PDF_0_175",
            "ANNZ_PDF_0_176", "ANNZ_PDF_0_177", "ANNZ_PDF_0_178", "ANNZ_PDF_0_179", "ANNZ_PDF_0_180", "ANNZ_PDF_0_181",
            "ANNZ_PDF_0_182", "ANNZ_PDF_0_183", "ANNZ_PDF_0_184", "ANNZ_PDF_0_185", "ANNZ_PDF_0_186", "ANNZ_PDF_0_187",
            "ANNZ_PDF_0_188", "ANNZ_PDF_0_189", "ANNZ_PDF_0_190", "ANNZ_PDF_0_191", "ANNZ_PDF_0_192", "ANNZ_PDF_0_193",
            "ANNZ_PDF_0_194", "ANNZ_PDF_0_195", "ANNZ_PDF_0_196", "ANNZ_PDF_0_197", "ANNZ_PDF_0_198", "ANNZ_PDF_0_199"]


    main_cats = []

    for fname in gamalist:
        incat = pd.read_csv(fname, names=colnames, header=None, skiprows=1)
        annz_main = Table.from_pandas(incat)
        del incat
        print(len(annz_main))
        main_cats.append(annz_main)
        del annz_main
        print("%s done!" % fname)
    print("Finished!")
    fullgama = vstack(main_cats)

    return fullgama


def match_fullgama_to_sdss_dr12(fullgama, sdsslist):
    outgama = []

    # Create all necessary column names
    newcols = []
    magcols = []
    extcols = []
    for magnitude in ["model", "cModel", "petro", "deV", "exp", "psf", "fiber2"]:
        for band in "ugriz":
            magcol = magnitude + "Mag_" + band
            extcol = "extinction_" + band
            newcol = "dered_" + magcol
            magcols.append(magcol)
            extcols.append(extcol)
            newcols.append(newcol)
            del magcol, extcol, newcol
    incols = ["objID"] + ["deVRad_r"] + magcols + extcols[:5]

    coltypes = {key:"float64" for key in incols if key != "objID"}
    coltypes["objID"] = "int64"

    for sdsspath in sdsslist:
        # Open catalog
        sdss = pd.read_csv(sdsspath, usecols=incols, dtype=coltypes, na_filter=False)
        sdsscat = Table.from_pandas(sdss)
        del sdss
        # Create new columns
        for magcol, extcol, newcol in zip(magcols, extcols, newcols):
            sdsscat[newcol] = sdsscat[magcol] - sdsscat[extcol]
        # Inner join to GAMA MAIN
        outcat = join(fullgama, sdsscat, keys="objID", join_type='inner')
        print(len(outcat))
        outgama.append(outcat)
        del outcat, sdsscat
        print("%s done!" % sdsspath)

    fullgama = vstack(outgama)

    return fullgama


def apply_LOWZ_cuts(sdsscat, exclude=True):
    """
    Apply BOSS LOWZ cuts to SDSS DR12 full catalog.
    """
    cpar = ( 0.7*(sdsscat["dered_modelMag_g"] - sdsscat["dered_modelMag_r"]) +
             1.2*(sdsscat["dered_modelMag_r"] - sdsscat["dered_modelMag_i"] - 0.18) )
    cperp = ( sdsscat["dered_modelMag_r"] - sdsscat["dered_modelMag_i"] -
             (sdsscat["dered_modelMag_g"] - sdsscat["dered_modelMag_r"])/4 - 0.18 )

    cperpcut = np.absolute(cperp) < 0.2
    rmodcut1 = ((sdsscat["dered_cModelMag_r"] > 16) & (sdsscat["dered_cModelMag_r"] < 19.6))
    rmodcut2 = (sdsscat["dered_cModelMag_r"] < (13.5 + cpar/0.3))

    rpsfcut = sdsscat["dered_psfMag_r"] - sdsscat["dered_cModelMag_r"] > 0.3

    mask = cperpcut & rmodcut1 & rmodcut2 & rpsfcut
    
    print("LOW-Z mask: %d" % np.sum(mask))
    
    if exclude:
        return sdsscat[~mask]
    else:
        return sdsscat[mask]


def apply_CMASS_cuts(sdsscat, exclude=True):
    """
    Apply BOSS CMASS cuts to RedMagic galaxy catalogs.
    """
    
    dperp = ( sdsscat["dered_modelMag_r"] - sdsscat["dered_modelMag_i"] -
             (sdsscat["dered_modelMag_g"] - sdsscat["dered_modelMag_r"])/8)

    dperpcut = dperp > 0.55
    imag_cut = sdsscat["dered_cModelMag_i"] < 19.86 + 1.6*(dperp - 0.8) # IMAG is dered_cModelMag_i
    imag_cut2 = (sdsscat["dered_cModelMag_i"] > 17.5) & (sdsscat["dered_cModelMag_i"] < 19.9) # IMAG is dered_modelMag_i
    
    ipsfcut = ( (sdsscat["dered_psfMag_i"] - sdsscat["dered_modelMag_i"]) >
                (0.2 + 0.2*(20 - sdsscat["dered_modelMag_i"])) )
    zpsfcut = ( (sdsscat["dered_psfMag_z"] - sdsscat["dered_modelMag_z"]) >
                (9.125 - 0.46*sdsscat["dered_modelMag_z"]) )
    
    rminusi_cut = (sdsscat["dered_modelMag_r"] - sdsscat["dered_modelMag_i"]) < 2
    ifib2_cut = sdsscat["dered_fiber2Mag_i"] < 21.5
    devRad_cut = sdsscat["deVRad_r"]/0.396 < 20 # SDSS pixel size: 0.396 arcseconds pixel-1
    
    mask = (dperpcut & imag_cut & imag_cut2 & ipsfcut & zpsfcut & rminusi_cut &
            ifib2_cut & devRad_cut)
    
    print("CMASS mask: %d" % np.sum(mask))
    
    if exclude:
        return sdsscat[~mask]
    else:
        return sdsscat[mask]


def exclude_redmagic_galaxies(sdsscat, redmagic_path):
    finalmagic_masked = Table.read(redmagic_path)
    gama_pd = sdsscat.to_pandas()
    magic_pd = finalmagic_masked.to_pandas()
    grouped = gama_pd.groupby(gama_pd["objID"].isin(magic_pd["OBJID"]))
    gama_nomagic = grouped.get_group(False)
    sdsscat = Table.from_pandas(gama_nomagic)
    return sdsscat


def main(gama_zphot_folder, gama_zphot_rootname, zname, sdssfolder,
         redmagic_path, outpath):
    """
    """
    assert os.path.exists(gama_zphot_folder), "GAMA catalog folder doesn't exist."
    assert os.path.exists(sdssfolder), "SDSS folder given doesn't exist."
    assert os.path.exists(redmagic_path), "RedMaGiC catalog doesn't exist."
    if not os.path.exists(os.path.dirname(outpath)):
        os.mkdir(os.path.dirname(outpath))

    gamapath = os.path.join(gama_zphot_folder, gama_zphot_rootname + "*")

    gamalist = glob.glob(gamapath)
    assert len(gamalist) > 0, "No GAMA files were found!"
    gamalist.sort()

    sdsslist = glob.glob(os.path.join(sdssfolder, "UCLimaging*csv"))
    assert len(sdsslist) == 100, "Not all SDSS files were found!"
    sdsslist.sort()

    fullgama = create_merged_gama_catalog(gamalist)
    fullgama = match_fullgama_to_sdss_dr12(fullgama, sdsslist)
    fullgama = apply_LOWZ_cuts(fullgama)
    fullgama = apply_CMASS_cuts(fullgama)
    fullgama = exclude_redmagic_galaxies(fullgama, redmagic_path)
    fullgama = fullgama["objID", "ra", "dec", zname]
    fullgama.write(outpath, format="fits")
        
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

    parser.add_argument('--gama_zphot_folder',
                        help="Path to GAMA photo-z catalogs")
    parser.add_argument('--gama_zphot_rootname',
                        help="Root name of GAMA photo-z catalogs")
    parser.add_argument('--zname',
                        default=None,
                        help="Redshift field name, where applicable")
    parser.add_argument('--sdssfolder',
                        help="Path to SDSS DR12 data folder")
    parser.add_argument('--redmagic_path',
                        help="Path to SDSS redmagic catalog")
    parser.add_argument('--outpath',
                        help="Output path")

    args = parser.parse_args()
    main(args.gama_zphot_folder, args.gama_zphot_rootname, args.zname,
         args.sdssfolder, args.redmagic_path, args.outpath)