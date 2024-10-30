## Imports

from scipy import stats
from loguniform import LogUniform, ModifiedLogUniform
import sys
import os
sys.path.append(os.path.abspath("/home/paul/Bureau/UdeM/GP/")) #path to the code
from GtemP import *

## load data

target = 'EV_LAC_SLINKY05' # target name
template = target          # template name (often the same as the target)

variable = 'DTEMP3500'      # select your metric
variable_err = 'sDTEMP3500' # select its errs

yourGP = GP_Object(target, template, variable, variable_err)

yourGP.outdir = '/home/paul/Bureau/UdeM/GP/out' # your output directory

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
yourGP.n_eta3 = stats.norm(4.36, 0.02)

## Run the MCMC

yourGP.run_emcee()
#you can specify the number of iterations and the number of iteration discarded
#for the burning phase
#Default are niter=10000 and discard=1000

## to check to convergence get a look at

yourGP.chain_plot()

## Get your corner plots

yourGP.corner_plot()

## look how it fits

yourGP.display_fit(yourGP.mean_fit())

