from helperFuncs import *

# command line arguments and basic settings
# --------------------------------------------------------------------------------------------------
init()

# just in case... (may comment this out)
if not glob.annz["doRandomReg"]:
    log.info(red(" - "+time.strftime("%d/%m/%y %H:%M:%S") +
             " - This scripts is only designed for randomRegression..."))
    sys.exit(0)

# ==================================================================================================
# The main code - randomized regression -
# --------------------------------------------------------------------------------------------------
#   - run the following:
#     python annz_rndReg_quick.py --randomRegression --genInputTrees
#     python annz_rndReg_quick.py --randomRegression --train
#     python annz_rndReg_quick.py --randomRegression --optimize
#     python annz_rndReg_quick.py --randomRegression --evaluate
# --------------------------------------------------------------------------------------------------
log.info(whtOnBlck(" - "+time.strftime("%d/%m/%y %H:%M:%S")+" - starting ANNZ"))

# --------------------------------------------------------------------------------------------------
# general options which are the same for all stages
#   - PLEASE ALSO INSPECT generalSettings(), WHICH HAS BEEN RUN AS PART OF init(), FOR MORE OPTIONS
# --------------------------------------------------------------------------------------------------
# outDirName - set output directory name
glob.annz["outDirName"] = "Amanda_june_mod9"

# nMLMs - the number of random MLMs to generate - for running the example,
# we set nMLMs at a small value, but this should be >~ 50 for production
glob.annz["nMLMs"] = 100  # 100

# zTrg            - the name of the target variable of the regression
# minValZ,maxValZ - the minimal and maximal values of the target variable (zTrg)
glob.annz["zTrg"] = "z"
glob.annz["minValZ"] = 0.0
glob.annz["maxValZ"] = 2.7467

# set the number of near-neighbours used to compute the KNN error estimator
glob.annz["nErrKNN"] = 100  # should be around ~100

# --------------------------------------------------------------------------------------------------
# pre-processing of the input dataset
# --------------------------------------------------------------------------------------------------
if glob.annz["doGenInputTrees"]:
    # inDirName    - directory in which input files are stored
    glob.annz["inDirName"] = "examples/data/"

    # inAsciiVars  - list of parameter types and parameter names, corresponding to columns in the input
    #                file, e.g., [TYPE:NAME] may be [F:MAG_U], with 'F' standing for float. (see advance
d example for detailed explanation)
    glob.annz["inAsciiVars"] = "F:MAG_AUTO_G_DERED;F:MAG_AUTO_R_DERED;F:MAG_AUTO_I_DERED;F:MAG_AUTO_Z_DE
RED;F:MAG_AUTO_Y_DERED;F:MAGERR_AUTO_G;F:MAGERR_AUTO_R;F:MAGERR_AUTO_I;F:MAGERR_AUTO_Z;F:MAGERR_AUTO_Y;D
:z"

    # splitTypeTrain - list of files for training. splitTypeTest - list of files for testing.
    glob.annz["splitTypeTrain"] = "ANNzVIPERSTrain_copy.csv"
    glob.annz["splitTypeTest"] = "ANNzVIPERSEval_copy.csv"
    # run ANNZ with the current settings
    runANNZ()

# --------------------------------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------------------------------
if glob.annz["doTrain"]:
    # for each MLM, run ANNZ
    for nMLMnow in range(glob.annz["nMLMs"]):
        glob.annz["nMLMnow"] = nMLMnow
        if glob.annz["trainIndex"] >= 0 and glob.annz["trainIndex"] != nMLMnow:
            continue

        # rndOptTypes - generate these randomized MLM types (currently "ANN", "BDT" or "ANN_BDT" are sup
ported).
        # for this example, since BDTs are much faster to train, exclude ANNs...
        glob.annz["rndOptTypes"] = "ANN_BDT"

        # inputVariables - semicolon-separated list of input variables for the MLMs. Can include math ex
pressions of the variables
        # given in inAsciiVars (see https://root.cern.ch/root/html520/TFormula.html for examples of vali
d math expressions)
        glob.annz["inputVariables"] = "MAG_AUTO_G_DERED;MAG_AUTO_R_DERED;MAG_AUTO_I_DERED;MAG_AUTO_Z_DER
ED;MAG_AUTO_Y_DERED"

        # can place here specific randomization settings, cuts and weights (see advanced example for det
ails)
        # if this is left as is, then random job options are generated internally in ANNZ, using MLM typ
es
        # given by rndOptTypes. see ANNZ::generateOptsMLM().
        # ....
        # ----------------------------------------------------------------------------------------------
----

        # run ANNZ with the current settings
        runANNZ()

# --------------------------------------------------------------------------------------------------
# optimization and evaluation
# --------------------------------------------------------------------------------------------------
if glob.annz["doOptim"] or glob.annz["doEval"]:

    # nPDFs - number of PDFs (see advanced example for a general description of PDFs)
    glob.annz["nPDFs"] = 1
    # nPDFbins - number of PDF bins, with equal width bins between minValZ and maxValZ. (see advanced ex
ample for setting other bin configurations)
    glob.annz["nPDFbins"] = 200

    # --------------------------------------------------------------------------------------------------
    # optimization
    # --------------------------------------------------------------------------------------------------
    if glob.annz["doOptim"]:
        # run ANNZ with the current settings
        runANNZ()

    # --------------------------------------------------------------------------------------------------
    # evaluation
    # --------------------------------------------------------------------------------------------------
    import os

    path = "examples/data/inputs_ANNz/"
    filenames_in = os.listdir(path)

    path_out = "/media/BINGODATA0/optical-data/DES_processed_data/DES_DR2_processed_ANNz"
    filenames_out = os.listdir(path_out)

    pixels_in = []
    for i in range(len(filenames_in)):
        pixels_in.append(filenames_in[i][12:17])

    pixels_out = []
    for i in range(len(filenames_out)):
        pixels_out.append(filenames_out[i][17:22])

    pixels_rem = list(set(pixels_in) - set(pixels_out))
    core = 10
    modulus = 9
    num = 0

    for i in filenames_in:
        name = i.split(".")
        pixel = name[0][12:17]
        if pixel in pixels_rem and num%core==modulus:
            if glob.annz["doEval"]:
                # inDirName,inAsciiFiles - directory with files to make the calculations from, and list
of input files
                glob.annz["inDirName"] = path
                glob.annz["inAsciiFiles"] = i
                # inAsciiVars - list of parameters in the input files (doesnt need to be exactly the sam
e as in doGenInputTrees, but must contain all
                #               of the parameers which were used for training)
                glob.annz["inAsciiVars"] = "F:MAG_AUTO_G_DERED;F:MAG_AUTO_R_DERED;F:MAG_AUTO_I_DERED;F:M
AG_AUTO_Z_DERED;F:MAG_AUTO_Y_DERED;F:MAGERR_AUTO_G;F:MAGERR_AUTO_R;F:MAGERR_AUTO_I;F:MAGERR_AUTO_Z;F:MAG
ERR_AUTO_Y"
                # evalDirPostfix - if not empty, this string will be added to the name of the evaluation
 directory
                #                  (can be used to prevent multiple evaluation of different input files
from overwriting each other)
                glob.annz["evalDirPostfix"] = name[0]
            else:
                print(i)

            # run ANNZ with the current settings
            runANNZ()
        num = num + 1

log.info(whtOnBlck(" - "+time.strftime("%d/%m/%y %H:%M:%S") +
         " - finished running ANNZ !"))
