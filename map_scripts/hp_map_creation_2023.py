  
"""Ha ha ha, I am learning so many things.
And I am passing all the pep8 and pep257 checks, how fucking awesome is that??
"""
import os
import errno
import glob
import sys
import argparse
import ast
import multiprocessing as mp
from functools import partial
import numpy as np
import healpy as hp
from astropy.table import Table

if sys.version_info[0] < 3:
    sys.exit("This script will only work with Python 3. Exiting code...")


# ---- Type checkers for efficient argument parsing in argparse
def nside_type(nside):
    """Check that nside is a valid Healpix int (i.e. a multiple of 2)."""
    nside = int(nside)
    if not hp.isnsideok(nside, nest=True):  # Weird nest=True needed for isnsideok to work.
        raise argparse.ArgumentTypeError("%d is not a valid Healpix Nside value." % nside)
    return nside


def zbins_type(zbins):
    """Check that zbins is a list of 2d tuples of ints or floats."""
    zbins = ast.literal_eval(zbins)
    if zbins is None:
        return None

    cond1 = isinstance(zbins, list)
    cond2 = all(isinstance(zbin, tuple) and len(zbin) == 2 for zbin in zbins)
    cond3 = all(isinstance(elem, (int, float)) for zbin in zbins for elem in zbin)
    if not (cond1 and cond2 and cond3):
        raise argparse.ArgumentTypeError("zbins is in the wrong format. Check documentation.")
    else:
        return zbins


def completeness_type(comp_cut):
    """Check that comp_cut is a float between 0 and 1."""
    comp_cut = float(comp_cut)
    if comp_cut < 0.0 or comp_cut > 1.0:
        raise argparse.ArgumentTypeError("%f not in range [0.0, 1.0]" % comp_cut)
    return comp_cut


def catformat_type(catformat):
    """Check that catformat is either 'csv' or 'fits'."""
    if catformat not in ("csv", "fits"):
        raise argparse.ArgumentTypeError("%s must be either 'csv' or 'fits'" % catformat)
    return catformat


# -------------------------------------------------------------------


def create_hp_count_map(nside, ra, dec, weights=None):
    """Create a Healpix galaxy count map from  1d arrays of RA and DEC positions.
    Takes a choice of Healpix Nside value for the output map and two 1d arrays
    with galaxy Right Ascension (RA) and Declinations (DEC). The arrays are of
    arbitrary but same length. RA and DEC must be in degrees, with RA in the
    interval [-180, 180] or, alternatively, [0, 360]. DEC must be in the range
    [-90, 90]. Optional weights per galaxy are used when summing in each pixel.
    If no weights are used, the counts in the Healpix map are a simple sum,
    otherwise the sum is weighted.
    Parameters
    ----------
    nside : int
        Choice of Healpix Nside for the map to be generated.
    ra : np.ndarray (dtype: float, dims: 1)
        Right ascension of galaxies, in degrees.
    dec : np.ndarray (dtype: float, dims: 1)
        Declination of galaxies, in degrees. Must have same length as ra.
    weights : None or np.ndarray (dtype: float or int, dims: 1)
        Galaxy weights. Count map will be a weighted sum, if not None.
        Must have same length as ra.
    Returns:
    --------
    countmap : np.ndarray (dtype: float, dims: 1, len: 12*nside*nside)
        A Healpix map of galaxy counts.
    """
    assert hp.isnsideok(nside), "nside must be a power of 2."
    assert (isinstance(ra, np.ndarray) and len(ra.shape) == 1
        and np.issubdtype(ra.dtype, float)), "ra must be a 1D numpy float array."
    assert (isinstance(dec, np.ndarray) and len(dec.shape) == 1
        and np.issubdtype(dec.dtype, float)), "dec must be a 1D numpy float array."
    assert len(ra) == len(dec), "ra/dec must have the same length."
    
    if (ra.shape[0] == 0 or dec.shape[0] == 0):
 #       print("len(ra.shape)", len(ra.shape))
        countmap = np.zeros(12*nside*nside)
        return countmap #we are returning an empty map if the ra or dec vector is empty
    
    else:
        
        assert ((ra.min() >= -180) and (ra.max() <= 180)) or ((ra.min() >= 0) and (ra.max() <= 360)), (
            "ra must be in degrees, between (-180, 180) or (0, 360)")
        assert (dec.min() >= -90) and (dec.max() <= 90), "dec must be in degrees and between (-90, 90)"

        if weights is not None:
            assert (isinstance(weights, np.ndarray) and len(weights.shape) == 1
                    and np.issubdtype(weights.dtype, float)), "weights must be a 1D numpy float array."
            assert len(weights) == len(ra), "weights must have the same length as ra/dec."

        # Assign each galaxy to a Healpix pixel
        theta = np.deg2rad(90.0 - dec)
        phi = np.deg2rad(ra)
        gal_hppix = hp.ang2pix(nside, theta=theta, phi=phi)

        # Create Healpix count map
        npix = hp.nside2npix(nside)
        countmap = np.bincount(gal_hppix, weights=weights, minlength=npix)
        assert len(countmap) == npix, "Something went wrong with pixel IDs"  # Can this catch anything?

        return countmap


def create_all_count_maps(zbins, infolder, nside, raname, decname, wname, zname, catformat):
    """Create count maps binned in redshift from input galaxy catalogs from infolder.
    Takes an input galaxy catalog with the corresponding coordinate, weight
    and redshift names, and creates a list of corresponding Healpix count maps.
    For a detailed description of how the Healpix count map creation works,
    check the 'create_hp_count_map' function documentation. For more details
    on the input variables, check the full hp_map_creation module documentation.
    Parameters:
    -----------
    infolder : str
        Input galaxy catalog folder.
    nside : int
        Healpix output count map resolution.
    raname : str
        RA column name.
    decname : str
        DEC column name.
    wname : None or str
        Galaxy weight column name. If None, no weights are used.
    zname : None or str
        Galaxy redshift column name. If None, only one redshift bin is created.
    zbins : list of 2d tuples of int of float values
        List of redshift bin limits. If None, only one redshift bin is created.
    catformat : str
        Input catalog format. Valid options: 'fits' or 'csv'.
    Returns:
    --------
    count_maps : list of 1d np.ndarrays
        List of Healpix count maps, one for each redshift bin.
    """
    #print("I'm process", os.getpid())
    #print("I am here opening the data \t",infile)
    #print("\n")

    firstrun = True
    
    inpaths = glob.glob(os.path.join(infolder, "*." + catformat))
    inpaths.sort()
    print (inpaths)

    zbinshere = []
    zbinshere.append(zbins)

    for inifilehere in inpaths:

        #print("I'm process", os.getpid())   
        #print("I'm opening",inifilehere)
        #print("Data opening done\n")

        data = Table.read(inifilehere, format=catformat)  # Don't try/except, astropy already does the job.

        assert hp.isnsideok(nside), "nside must be a power of 2."
        for name in (raname, decname):
            assert name in data.colnames, ("%s is not a column in the input data." % name)
        if zname is not None:
            assert zname in data.colnames, ("%s is not a column in the input data." % zname)
        if wname is not None:
            assert wname in data.colnames, ("%s is not a column in the input data." % wname)
        if zbinshere is not None:
            assert (isinstance(zbinshere, list)
                    and all(isinstance(zbin, tuple) and len(zbin) == 2 for zbin in zbinshere)
                    and all(isinstance(elem, (int, float)) for zbin in zbinshere for elem in zbin)), (
                        "zbins must be a list of 2d tuples of ints or floats for redshift bin limits.")
        assert catformat in ("csv", "fits"), "Catalog format must be fits or csv."

        if (zname is None):
            zname = "fake_column"
            data[zname] = np.zeros(len(data))
            zbins = [(-np.inf, np.inf)]

        zcol = data[zname]
        for (inf, sup) in zbinshere:
            subdata = data[(zcol >= inf) & (zcol < sup)]
            ra = subdata[raname]
            dec = subdata[decname]
            if wname is not None:
                weights = subdata[wname]
            else:
                weights = None
        
        if (firstrun == True): 
            count_maps = create_hp_count_map(nside, ra, dec, weights=weights)
            #print(count_maps.shape)
            firstrun = False
        else:
            map_here = create_hp_count_map(nside, ra, dec, weights=weights)
            #print (map_here.shape)
            #print (count_maps.shape)
            count_maps = np.add(map_here,count_maps)
        #count_maps.append(create_hp_count_map(nside, ra, dec, weights=weights))

    print("I have done the counting and this is the shape:",len(count_maps))
    return count_maps


def sum_maps(hpmaps):
    """Merge list of healpix galaxy maps along one axis.
    Takes a list of Healpix maps and sums the maps along the first axis. The
    input can be at most a list of lists of Healpix maps. The last dimension
    is required to be a 1d numpy array of healpix format. The main use case
    of this function is to take the result of a multiprocessing pool that
    creates healpix maps and merge the maps together. Optionally, the input
    can be a list of 1d lists of maps - usually resulting from binning in
    a galaxy property - in which case a list of maps will be returned.
    Parameters
    ----------
    hpmaps : list of arrays or list of lists of arrays
        A one- or two-dimensional list of Healpix maps.
    Returns:
    --------
    merged_hpmaps : array or list of arrays
        A Healpix map which is a sum of a 1d list of input maps or a list of
        maps which are sums along the 1st axis of a 2d list of input maps.
    Raises:
    -------
    NotImplementedError
        If any of the input maps is masked with hp.UNSEEN values, the merge
        will be aborted.
    """
    assert isinstance(hpmaps, list), "Input must be a list."

    try:
        hpmaps = np.array(hpmaps).astype("float")
    except ValueError as err:
        print(err)
        print("Input to sum_maps must be a nested list of np.ndarrays of type 'int' or 'float'")

    assert len(hpmaps.shape) == 2 or len(hpmaps.shape) == 3, ("Input must be a list or a list of " +
                                                              "lists of np.ndarrays")

    if np.any(np.isclose(hp.UNSEEN, np.array(hpmaps))):
        raise NotImplementedError("One or more of the maps is masked with Healpix UNSEEN values. " +
                                  "Summing masked maps is currently not supported.")

    merged_hpmaps = np.sum(hpmaps, axis=0)
    if len(merged_hpmaps.shape) > 1:
        merged_hpmaps = list(merged_hpmaps)

    return merged_hpmaps


def make_overdensity_map(count_map, completeness_mask, comp_cut):
    """Create a Healpix galaxy overdensity map from a count map.
    Takes a Healpix galaxy count map, completeness mask of the same resolution
    and a completeness threshold cut, and produces an upweighted galaxy
    overdensity map. The process of overdensity creation is the following:
    a completeness mask, with values between 0 and 1, indicates the fraction
    of each pixel that has been observed. The completeness threshold cut is
    applied to the mask, generating a footprint of accepted pixels above
    this threshold. The footprint and completeness masks are applied to the
    count map; where the pixel completeness is higher than the threshold,
    counts are upweighted to compensate for the unseen area. This is
    mathematically equivalent to assume that the missing area contains the
    same average of counts as the seen area in the pixel. Afterwards, the mean
    weighted galaxy count per pixel is calculated, and the overdensity map
    is defined as the mean fractional over/underdensity compared to the mean.
    To prevent information loss when passing from a count to an overdensity
    map, the raw and weighted average and total counts, and the observed sky
    fractional area are returned as a 'metadata' dictionary.
    Parameters:
    -----------
    count_map : np.ndarray (dtype: float, dims: 1)
        A Healpix map of galaxy counts.
    completeness_mask : np.ndarray (dtype: float, dims: 1)
        A Healpix map in the range [0, 1], indicating the fractional observed
        area in each Healpix pixel. Must be the same length as count_map.
    comp_cut : float ([0, 1])
        A completeness cut on pixel fractional observed area. Pixels with
        lower fraction will be set to zero. Counts in pixels with higher
        fraction will be upweighted (i.e. divided by the pixel fractional
        area).
    Returns:
    --------
    overdensity : np.ndarray (dtype: float, dims: 1)
        A Healpix map of galaxy overdensities.
    metadata : dict
        Map metadata: Includes total number of raw and weighted galaxies,
        mean number of raw and weighted galaxies per steradian and
        the fractional observed sky area.
    """
    len_maps = list(map(len, [count_map, completeness_mask]))

    if hp.UNSEEN in np.array([count_map, completeness_mask]):
        raise NotImplementedError("This function cannot currently deal with hp.UNSEEN values.")

    # These are inner pipeline checks, so use assertions
    assert np.all(list(map(hp.isnpixok, len_maps))), "All maps must have a valid Healpix Nside."
    assert len(set(len_maps)) == 1, "All inputs must have the same Healpix Nside."
    # assert np.issubdtype(count_map.dtype, np.integer), "Count map must be an integer map."
    assert count_map[count_map != hp.UNSEEN].min() == 0, "Counts in count_map must be positive."
    assert (completeness_mask.min() >= 0) and (completeness_mask.max() <= 1), (
        "Completeness must be between 0. and 1.")
    assert 0 < comp_cut <= 1, "Completeness cut must be between 0 and 1"

    nside = hp.get_nside(count_map)

    mask = completeness_mask > comp_cut

    raw_counts = np.where(mask, count_map, 0)
    weighted_counts = np.where(mask, count_map/completeness_mask, 0)

    nbar_weighted_pix = np.sum(weighted_counts[mask])/np.sum(mask)  # per pixel
    
    if nbar_weighted_pix==0:
        overdensity=np.zeros(12*nside*nside)
    else:
        overdensity = np.where(mask, weighted_counts/nbar_weighted_pix - 1, 0)

    area_steradians = hp.nside2pixarea(nside)*np.sum(mask)

    metadata = {}
    metadata["ntot_raw"] = np.sum(raw_counts[mask])
    metadata["ntot_weighted"] = np.sum(weighted_counts[mask])
    metadata["nbar_raw_steradian"] = np.sum(raw_counts[mask])/area_steradians
    metadata["nbar_weighted_steradian"] = np.sum(weighted_counts[mask])/area_steradians
    metadata["fsky"] = np.sum(mask)/len(mask)

    return (overdensity, metadata)


def create_output_path(outdir, outfileroot, nside, suffix):
    """Create output path to overdensity maps.
    This is a small utility function to format the output path of the overdensity maps. The main
    use is to automatically add the Nside and redshift bin values to the map name.
    Parameters
    ----------
    outdir : str
        Output directory path.
    outfileroot : str
        Output file root name. It will be common to all output maps.
    nside : int
        Nside of the output Healpix maps.
    suffix : str
        Suffix corresponding to the redshift bin limits or footprint mask
    Returns:
    --------
    outpath : str
        The full output path string.
    """
    suffix = "_N" + str(nside) + suffix
    outpath = os.path.join(outdir, outfileroot + suffix + ".fits")

    return outpath


def main(infolder, nside, maskpath, outdir, outfileroot, raname, decname, wname, zname, zbins,
         comp_cut, nprocs, catformat, overwrite):
    """Create Healpix overdensity maps and footprint mask from a set of galaxy catalogs.
    Run 'python $ZXCORR_HOME/scripts/map_creation/hp_map_creation.py --help'
    for detailed documentation. Replace $ZXCORR_HOME with the base directory
    of your zxcorr repository.
    """
    # ------ IO checks
    # Those are external checks, so they should not use asserts. Use if/else or try/except.
    if not os.path.exists(infolder):
        raise OSError("Input folder with catalogs doesn't exist.")

    if nprocs > mp.cpu_count():
        print("Number of processors requested is larger than available. Will use all available.")
        nprocs = mp.cpu_count()

    if nprocs > len(zbins):
        raise OSError("This program is parallelized in number of zbins, so nprocs must be =< than the number of zbins")
    

    # Mask checks
    if not os.path.exists(maskpath):
        raise OSError("Healpix mask path doesn't exist.")
    completeness_mask = hp.read_map(maskpath)
    nside_mask = hp.get_nside(completeness_mask)
    if nside_mask != nside:
        raise IOError("Nside of mask must be equal to target nside.")

    # Create output folder
    try:
        os.makedirs(outdir)
    except OSError as err:
        if err.errno != errno.EEXIST:
            raise

    # ------ Read input catalogs and run map creation in parallel and Merge maps

    inpaths = glob.glob(os.path.join(infolder, "*." + catformat))
    inpaths.sort()
    print (inpaths)
    # Create maps with multiprocessing pool

    print ("now printing the zbins:")
    print (zbins)
    print("Printing infolder", infolder)
    processors = mp.Pool(nprocs)

    lambda_create_count_maps = partial(create_all_count_maps, infolder=infolder, nside=nside, raname=raname,
                                       decname=decname, wname=wname, zname=zname, catformat=catformat)
    
    print("Done creating multifunction")
    
    count_maps = []

    for i in range(len(zbins)//nprocs):
        #print("The range is:",len(zbins)//nprocs)
        #print("The number i is:",i)
        #print(zbins[i*nprocs:i*nprocs+nprocs])
        if (i != len(zbins)//nprocs): 
            parallel_count_maps = processors.map(lambda_create_count_maps, zbins[i*nprocs:i*nprocs+nprocs])
            for each_count_map in parallel_count_maps:
                count_maps.append(each_count_map)
        else:
            parallel_count_maps = processors.map(lambda_create_count_maps, zbins[i*nprocs:])
            for each_count_map in parallel_count_maps:
                count_maps.append(each_count_map)

        #if (i == 0): 
        #    count_maps = sum_maps(parallel_count_maps)
        #else:
        #    parallel_count_maps.append(count_maps)
        #    count_maps = sum_maps(parallel_count_maps)
        #sec = input('Let us wait for user input.\n')

    # ------ Postprocess maps - Transform to overdensities and save output

    # Didn't do it earlier because create_all_count_maps needs zbins=None or a non-empty binning.
    if zbins is None:
        zbins = [("", "")]

    # Redshift binning
    for count_map, zbin in zip(count_maps, zbins):
        overdens_map, metadata = make_overdensity_map(count_map, completeness_mask, comp_cut)
        overdens_header = [item for item in metadata.items()]

        if zbin[0] == "" and zbin[1] == "":
            zsuffix = ""
        else:
            zsuffix = "_z%.2f-%.2f" % (zbin[0], zbin[1])

        outpath = create_output_path(outdir, outfileroot, nside, zsuffix)
        hp.write_map(outpath, overdens_map, extra_header=overdens_header, overwrite=overwrite)

    # Create and save footprint mask
    footprint = (completeness_mask > comp_cut).astype("int")
    outpath_foot = create_output_path(outdir, outfileroot, nside, "_footprint")
    hp.write_map(outpath_foot, footprint, overwrite=overwrite)


# -------------------------------------------------------------------


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=__doc__, fromfile_prefix_chars='@')

    PARSER.add_argument('--infolder',
                        required=True,
                        help="Path to input data folder [REQUIRED].")
    PARSER.add_argument('--nside',
                        type=nside_type,
                        required=True,
                        help="Healpix output map resolution [REQUIRED].")
    PARSER.add_argument('--maskpath',
                        required=True,
                        help="Path to input healpix footprint mask [REQUIRED].")
    PARSER.add_argument("--outdir",
                        required=True,
                        help="Output folder path [REQUIRED].")
    PARSER.add_argument("--outfileroot",
                        required=True,
                        help=("Output file root name [REQUIRED]."))
    PARSER.add_argument('--raname',
                        default='ra',
                        help="RA field name.")
    PARSER.add_argument('--decname',
                        default='dec',
                        help="DEC field name.")
    PARSER.add_argument('--wname',
                        default=None,
                        help="Weight field name, where applicable.")
    PARSER.add_argument('--zname',
                        default=None,
                        help="Redshift field name, where applicable.")
    PARSER.add_argument('--zbins',
                        type=zbins_type,
                        default=None,
                        help="List of bins. Format is [(b1inf, b1sup), ..., (bNinf, bNsup)].")
    PARSER.add_argument('--comp_cut',
                        type=completeness_type,
                        default=0,
                        help="Pixel completeness cut (between 0 and 1). Default: 0 [no exclusion].")
    PARSER.add_argument('--nprocs',
                        type=int,
                        default=1,
                        help="Number of cores used for parallelisation. Default: 1.")
    PARSER.add_argument("--catformat",
                        default="fits",
                        help=("Input format. Currently support 'csv' and 'fits'. Default: 'fits'."))
    PARSER.add_argument("--overwrite",
                        action='store_true',
                        help=("If flag is present, overwrite existing files."))

    ARGS = PARSER.parse_args()

    main(ARGS.infolder, ARGS.nside, ARGS.maskpath, ARGS.outdir, ARGS.outfileroot,
         ARGS.raname, ARGS.decname, ARGS.wname, ARGS.zname, ARGS.zbins,
         ARGS.comp_cut, ARGS.nprocs, ARGS.catformat, ARGS.overwrite)
