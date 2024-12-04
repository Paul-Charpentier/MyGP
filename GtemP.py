## Imports

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
from multiprocessing import Pool
import emcee
import corner
from loguniform import LogUniform, ModifiedLogUniform
import sys
sys.path.append(os.path.abspath("/home/paul/Bureau/IRAP/TediGP"))
import process, kernels, means
from tqdm import tqdm
#from IPython.display import display, Math
import csv
from collections import OrderedDict
sys.path.append(os.path.abspath("/media/paul/One Touch11/wapiti_workflow"))
from wapiti import wapiti_tools, wapiti
from astropy.timeseries import LombScargle
from sklearn.metrics import mean_squared_error

def patch_asscalar(a):
    return a.item()
setattr(np, "asscalar", patch_asscalar)

## Set-Up
#
# #Target
# Target = 'EV_LAC_SLINKY05'
# Template = Target
#
# #time serie to study
# var  = 'DTEMP3500'
# svar = 'sDTEMP3500'


# base work directory
cwd = "/media/paul/One Touch11/UdeM/"

## Load data

class GP_Object:
    def __init__(self, target, template, var, svar):
        self._target = target
        print('Traget = ', self._target)
        self._filename = 'lbl2_' + target + '_' + template + '.rdb'
        self._varname = var
        self._svarname = svar
        print('Variable = ', self._varname)

        data = read_rdb_asdict(cwd + 'lbl_spirou/lblrdb/' + self._filename)

        self.times = data['rjd']
        self.y = data[self._varname]
        self.yerr = data[self._svarname]
        self.outdir = cwd + 'out/'

    def rm_outlier(self, mad_rejection = 10):
        # MAD rejection threshold for the outlier removal
        outliers = ind_outliers(self.y, mad_rejection)
        if len(outliers)>=1:
            print('rejecting ', len(outliers), ' outlier')
            self.times = np.delete(np.copy(self.times), outliers)
            self.y = np.delete(np.copy(self.y), outliers)
            self.yerr = np.delete(np.copy(self.yerr), outliers)

    def default_hyperparam(self):
        y_sig = np.std(self.y)
        y_ptp = np.max(self.y) - np.min(self.y)
        t_span = np.max(self.times) - np.min(self.times)
        t_av = np.mean(self.times[1:] - self.times[:-1])

        #Priors
        self.n_eta1 = ModifiedLogUniform(y_sig, 2*y_ptp)
        self.n_eta2 = stats.loguniform(t_av, 10*t_span)
        self.n_eta3 = stats.uniform(2, 200)
        self.n_eta4 = stats.loguniform(0.1, 5)
        self.jitter = ModifiedLogUniform(y_sig, 2*y_ptp)

    def priors(self):
        return(np.array([self.n_eta1.rvs(), self.n_eta2.rvs(),
                         self.n_eta3.rvs(), self.n_eta4.rvs(),
                         self.jitter.rvs()]))

    def logPosterior(self, HyperParam):
        n1,n2,n3,n4,j = HyperParam

        logprior = self.n_eta1.logpdf(n1)
        logprior += self.n_eta2.logpdf(n2)
        logprior += self.n_eta3.logpdf(n3)
        logprior += self.n_eta4.logpdf(n4)
        logprior += self.jitter.logpdf(j)

        kernel = kernels.QuasiPeriodic(n1,n2,n3,n4) + kernels.WhiteNoise(j)
        mean = means.Constant(np.mean(self.y))
        gpOBJ = process.GP(kernel, mean, self.times, self.y, yerr = self.yerr)

        logposterior = gpOBJ.log_likelihood() + logprior
        return logposterior

    def run_emcee(self, niter = 10000, discard=1000, thin=15):
        self.ndim = self.priors().size
        nwalkers = 2*self.ndim
        self.sampler = emcee.EnsembleSampler(nwalkers, self.ndim,
                                             self.logPosterior)
        p0=[self.priors() for i in range(nwalkers)]
        self.sampler.run_mcmc(p0, niter, progress=True)
        self.flat_samples = self.sampler.get_chain(discard=discard, thin=thin,
                                                   flat=True)

    def chain_plot(self):
        #chains plot
        fig, axes = plt.subplots(self.ndim, figsize=(7, 12), sharex=True)
        samples = self.sampler.get_chain()
        labels = ["$\eta_1$", "$\eta_2$", "$\eta_3$", "$\eta_4$", "Jitter"]
        for i in range(self.ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel("step number");
        plt.show()

    def corner_plot(self):
        #corner plot
        labels = ["$\eta_1$", "$\eta_2$", "$\eta_3$", "$\eta_4$", "Jitter"]
        fig = corner.corner(self.flat_samples, labels=labels, color="k",
                            bins = 50, quantiles=[0.16, 0.5, 0.84], smooth=True,
                            smooth1d=True, show_titles=True, plot_density=True,
                            plot_contours=True, fill_contours=True,
                            plot_datapoints=False, title_fmt='g')
        plt.savefig(self.outdir + self._target + '_' + self._varname + '_corner.pdf',
                    format="pdf", bbox_inches="tight")
        #plt.show()

    def best_fit(self):
        ind_mlh = np.argmax([self.logPosterior(self.flat_samples[i])
                             for i in tqdm(range(len(self.flat_samples)))])
        sample_mlh = self.flat_samples[ind_mlh]
        return (sample_mlh, self.logPosterior(sample_mlh))

    def save_results(self):
        labels = ["$\eta_1$", "$\eta_2$", "$\eta_3$", "$\eta_4$", "Jitter"]
        post = []
        for i in range(self.ndim):
            mcmc = np.percentile(self.flat_samples[:,i], [16, 50, 84])
            q = np.diff(mcmc)
            txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
            txt = txt.format(mcmc[1], q[0], q[1], labels[i])
            print(txt)
            post.append(txt)

        mf_params = self.mean_fit()
        mf_residual, mf_wrms = self.display_fit(mf_params)

        print('wRMS = ', mf_wrms)
        post.append('wRMS = ' + str(mf_wrms))

        logpost = self.logPosterior(mf_params)
        print('LogPosterior = ', logpost)
        post.append('LogPosterior = ' + str(logpost))

        chi = chisq(mf_residual, self.yerr, np.zeros_like(mf_residual))
        print('$\chi^2$ = ', chi)
        post.append('$\chi^2$ = '+ str(chi))

        chir = chi/(len(mf_residual)-self.ndim-1)
        print('$\chi^2_{red}$ = ', chir)
        post.append('$\chi^2_{red}$ = '+ str(chir))

        chi = normalized_chi2(mf_residual, self.yerr, mf_params[-1])
        print('$\chi^2$ = ', chi)
        post.append('$\chi^2$ = '+ str(chi))

        chir = chi/(len(mf_residual)-self.ndim-1)
        print('$\chi^2_{red}$ = ', chir)
        post.append('$\chi^2_{red}$ = '+ str(chir))

        np.savetxt(self.outdir + self._target + '_' + self._varname + '_results.txt', post, fmt='%s')

        self.save_residuals()

        self.corner_plot()

    def mean_fit(self):
        return(np.median(self.flat_samples, axis=0))

    def plot_timeseries(self, fmt='ko'):
        plt.errorbar(self.times, self.y, self.yerr, fmt=fmt)
        plt.xlabel('rjd (d)', size=12, weight='bold')
        plt.ylabel(self._varname, size=12, weight='bold')
        plt.show()

    def plot_periodogram(self, c='k'):
        wf = np.ones(len(self.times))
        frequency, power = LombScargle(
            self.times,
            wf,
            fit_mean=False,
            center_data=False).autopower(minimum_frequency=0.0005,
                                         maximum_frequency=1/1.5)
        plt.plot(1/frequency, power, 'k', alpha=0.1)
        frequency, power = LombScargle(
            self.times,
            self.y).autopower(minimum_frequency=1/2000, maximum_frequency=1/1.5)
        plt.plot(1/frequency, power, c=c)
        plt.ylabel("power", size=12, weight='bold')
        plt.xlabel("period (d)", size=12, weight='bold')
        plt.xscale('log')
        ls = LombScargle(self.times, self.y)
        fap = ls.false_alarm_level(0.1)
        plt.axhline(fap, linestyle='-', color='k')
        fap = ls.false_alarm_level(0.01)
        plt.axhline(fap, linestyle='--', color='k')
        fap = ls.false_alarm_level(0.001)
        plt.axhline(fap, linestyle=':', color='k')
        plt.show()

    def display_fit(self, HyperParam, c='orange'):
        n1,n2,n3,n4,j = HyperParam

        kernel = kernels.QuasiPeriodic(n1,n2,n3,n4) + kernels.WhiteNoise(j)
        mean = means.Constant(np.mean(self.y))
        gpOBJ = process.GP(kernel, mean, self.times, self.y, yerr = self.yerr)

        tplot = np.linspace(np.min(self.times)-50, np.max(self.times)+50, 5000)

        y_mean, y_std, time = gpOBJ.prediction(kernel,mean,tplot)

        fig, ax = plt.subplots(2, 1, sharex=True, height_ratios=(2,1), figsize=(16, 8))
        ax[0].errorbar(self.times, self.y, self.yerr, fmt='ko')
        ax[0].plot(tplot, y_mean, c=c)
        ax[0].fill_between(tplot, y_mean+y_std, y_mean-y_std, color=c, alpha=0.3)
        ax[0].set_ylabel(self._varname, size=12, weight='bold')

        y_sample, _, _ = gpOBJ.prediction(kernel,mean,self.times)
        y_sample -= self.y

        self.residuals = y_sample

        ax[1].errorbar(self.times, y_sample, self.yerr, fmt='ko')
        ax[1].set_xlabel('rjd (d)', size=12, weight='bold')
        ax[1].set_ylabel('residuals', size=12, weight='bold')
        wrms = mean_squared_error(y_sample, np.zeros_like(y_sample),
                                  sample_weight=1/self.yerr, squared=False)
        ax[1].annotate( "wRMS :  " + str(wrms), (self.times[2], np.max(y_sample)))
        plt.savefig(self.outdir + self._target + '_' + self._varname + '_fit.pdf',
                    format="pdf", bbox_inches="tight")
        return(y_sample, wrms)

    def save_residuals(self):
        with open(self.outdir + self._target + '_' + self._varname + '_residuals.rdb', 'w') as file:
            # First line
            file.write('rjd \t' + self._varname + '\t' + self._svarname + '\n')
            file.write('\n')
            for i in range(len(self.times)):
                file.write(str(self.times[i]) + '\t' + str(self.residuals[i]) + '\t' + str(self.yerr[i]) + '\n')
        file.close()


## tools

def read_rdb_asdict(filename):
    """ Reads a .rdb file with header
    col1 col2 col3
    ---- ---- ----
    Returns a (ordered) dictionary
    """
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        # first line has column names
        names = next(reader)
        data = OrderedDict((name,[]) for name in names)
        # first line has data type (string or float)
        data_type = next(reader)

        for line in reader:

            values = []
            for l in range(len(line)):
                v = line[l]
                if data_type[l] == 'N':
                    v = float(v)
                values.append(v)

            [list(data.values())[i].append(v) for i, v in enumerate(values)]
        for k,v in data.items():
            data[k] = np.array(v)


    return data

def ind_outliers(v, k):
    madv= wapiti_tools.mad(v)
    rejection = wapiti_tools.absolute_deviation(v) > k*madv
    rejection_index = np.where(rejection)[0]
    return(rejection_index)

def chisq(obs, obs_err, exp):
    return(np.sum((obs - exp)**2 / obs_err**2))


def normalized_chi2(res, yerr, jitt):
    norm_res = res/np.sqrt(yerr**2 + (np.ones_like(yerr)*jitt)**2)
    chi2 = np.dot(norm_res,norm_res)
    return chi2
