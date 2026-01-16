## Imports

import warnings
warnings.filterwarnings("ignore")
from scipy import stats
from loguniform import LogUniform, ModifiedLogUniform
import sys
import os
sys.path.append(os.path.abspath("/media/paulch/One Touch3/GP_temp_config/")) #path to the code
from GtemP import *

## load data

target = 'GJ1289' # target name
template = target          # template name (often the same as the target)

variable = 'DTEMP3000'      # select your metric
variable_err = 'sDTEMP3000' # select its errs

yourGP = GP_Object(target, template, variable, variable_err)

yourGP.check_all_dtemps(Prot=73.66, dProt=0.92)

yourGP.outdir = '/media/paulch/One Touch3/GP_temp_config/out/' # your output directory

## Eventually removes outliers

yourGP.rm_outlier()
#it uses MAD rejection with a default threshold of mad_jejection = 10

## Set priors

# you can symply invoque the default prior distribution using :
yourGP.default_hyperparam()
# where default priors are :
        # self.n_eta1 = ModifiedLogUniform(y_sig, 2*y_ptp) #(Amplitude)
        # self.n_eta2 = stats.loguniform(t_av, 10*t_span)  #(Evolution timescale)
        # self.n_eta3 = stats.uniform(2, 200)              #(Rotation period)
        # self.n_eta4 = stats.loguniform(0.1, 5)           #(Smoothong Factor)
        # self.jitter = ModifiedLogUniform(y_sig, 2*y_ptp)

# or can set manualy the distribution of some or all the priors
#yourGP.n_eta2 = stats.norm(300, 14)
#yourGP.n_eta3 = stats.norm(38.8, 4)
## Run the MCMC

yourGP.run_emcee(niter=20000, discard=2000)
#you can specify the number of iterations and the number of iteration discarded
#for the burning phase
#Default are niter=10000 and discard=1000

## to check to convergence get a look at

#yourGP.chain_plot()

## Get your corner plots

#yourGP.corner_plot()

## look how it fits

#yourGP.display_fit(yourGP.mean_fit())

yourGP.save_results()
