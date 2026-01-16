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
sys.path.append(os.path.abspath("/media/paulch/One Touch5/GP_temp_config/TediGP"))
import process, kernels, means
from tqdm import tqdm
#from IPython.display import display, Math
import csv
from collections import OrderedDict
sys.path.append(os.path.abspath("/media/paulch/One Touch5/wapiti_workflow"))
from wapiti import wapiti_tools, wapiti
from astropy.timeseries import LombScargle
from sklearn.metrics import mean_squared_error
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

def patch_asscalar(a):
    return a.item()
setattr(np, "asscalar", patch_asscalar)

# base work directory
cwd = "/media/paulch/One Touch5/wapiti_workflow/"


## GP Object

## Load data

class GP_Object:
    def __init__(self, target, template, var, svar):
        self.post = []
        self._target = target
        print('Target = ', self._target)
        self.post.append('Target = '+ self._target)
        self._filename = 'lbl2_' + target + '_' + template + '.rdb'
        self._varname = var
        self._svarname = svar
        print('Variable = ', self._varname)
        self.post.append('Variable = ' + self._varname)

        self.data = read_rdb_asdict(cwd + 'lblrdb/' + self._filename)

        self.times = self.data['rjd']
        self.y = self.data[self._varname]
        self.yerr = self.data[self._svarname]
        self.outdir = cwd + 'out/'

    def rm_outlier(self, mad_rejection = 10):
        # MAD rejection threshold for the outlier removal
        outliers = ind_outliers(self.y, mad_rejection)
        if len(outliers)>=1:
            print('rejecting ', len(outliers), ' outlier')
            self.post.append('rejecting ' + str(len(outliers)) + ' outlier(s)')
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
        
        self.fixed_params = {"n_eta1": None,  # not fixed
                             "n_eta2": None,   
                             "n_eta3": None,
                             "n_eta4": None,
                             "jitter": None}
        
    def priors(self):
        samples = []
        if self.fixed_params["n_eta1"] is None:
            samples.append(self.n_eta1.rvs())
        if self.fixed_params["n_eta2"] is None:
            samples.append(self.n_eta2.rvs())
        if self.fixed_params["n_eta3"] is None:
            samples.append(self.n_eta3.rvs())
        if self.fixed_params["n_eta4"] is None:
            samples.append(self.n_eta4.rvs())
        if self.fixed_params["jitter"] is None:
            samples.append(self.jitter.rvs())
        return np.array(samples)
    
    def logPosterior(self, sampled_params):
        full_params = {}
        keys = ["n_eta1", "n_eta2", "n_eta3", "n_eta4", "jitter"]
        idx = 0

        for key in keys:
            if self.fixed_params[key] is not None:
                full_params[key] = self.fixed_params[key]
            else:
                full_params[key] = sampled_params[idx]
                idx += 1

        # Compute log prior
        logprior = 0
        if self.fixed_params["n_eta1"] is None:
            logprior += self.n_eta1.logpdf(full_params["n_eta1"])
        if self.fixed_params["n_eta2"] is None:
            logprior += self.n_eta2.logpdf(full_params["n_eta2"])
        if self.fixed_params["n_eta3"] is None:
            logprior += self.n_eta3.logpdf(full_params["n_eta3"])
        if self.fixed_params["n_eta4"] is None:
            logprior += self.n_eta4.logpdf(full_params["n_eta4"])
        if self.fixed_params["jitter"] is None:
            logprior += self.jitter.logpdf(full_params["jitter"])

        kernel = kernels.QuasiPeriodic(full_params["n_eta1"],
                                       full_params["n_eta2"],
                                       full_params["n_eta3"],
                                       full_params["n_eta4"]) + kernels.WhiteNoise(full_params["jitter"])

        mean = means.Constant(np.mean(self.y))
        gpOBJ = process.GP(kernel, mean, self.times, self.y, yerr=self.yerr)
        logposterior = gpOBJ.log_likelihood() + logprior
        return logposterior

    def run_emcee(self, niter = 10000, discard=1000, thin=15):
        self.ndim = self.priors().size
        nwalkers = 2*self.ndim
        self.sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self.logPosterior)
        p0=[self.priors() for i in range(nwalkers)]
        self.sampler.run_mcmc(p0, niter, progress=True)
        self.flat_samples = self.sampler.get_chain(discard=discard, thin=thin, flat=True)

    def chain_plot(self):
        fig, axes = plt.subplots(self.ndim, figsize=(7, 12), sharex=True)
        samples = self.sampler.get_chain()
        labels_all = ["a (k)", "$\lambda_e$", "$P_{rot}$", "$\lambda_p$", "Jitter"]
        sample_labels = [l for i, l in enumerate(labels_all) if list(self.fixed_params.values())[i] is None]

        for i in range(self.ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(sample_labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel("step number")
        plt.show()

    def corner_plot(self):
        labels_all = ["a (k)", "$\lambda_e$", "$P_{rot}$", "$\lambda_p$", "Jitter"]
        sample_labels = [l for i, l in enumerate(labels_all) if list(self.fixed_params.values())[i] is None]
        fig = corner.corner(self.flat_samples, labels=sample_labels, color="k",
                            bins=50, quantiles=[0.16, 0.5, 0.84], smooth=True,
                            smooth1d=True, show_titles=True, plot_density=True,
                            plot_contours=True, fill_contours=True,
                            plot_datapoints=False, title_fmt='g')
        plt.savefig(self.outdir + self._target + '_' + self._varname + '_corner.pdf', format="pdf", 
                    bbox_inches="tight")

    def best_fit(self):
        ind_mlh = np.argmax([self.logPosterior(self.flat_samples[i])
                             for i in tqdm(range(len(self.flat_samples)))])
        sample_mlh = self.flat_samples[ind_mlh]
        return (sample_mlh, self.logPosterior(sample_mlh))

    def save_results(self):
        labels_all = ["a (k)", "$\lambda_e$", "$P_{rot}$", "$\lambda_p$", "Jitter"]
        sample_labels = [l for i, l in enumerate(labels_all) if list(self.fixed_params.values())[i] is None] 
        for i in range(self.ndim):
            mcmc = np.percentile(self.flat_samples[:,i], [16, 50, 84])
            q = np.diff(mcmc)
            txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
            txt = txt.format(mcmc[1], q[0], q[1], sample_labels[i])
            print(txt)
            self.post.append(txt)

        mf_params = self.mean_fit()
        mf_residual, mf_wrms = self.display_fit(mf_params)

        print('wRMS = ', mf_wrms)
        self.post.append('wRMS = ' + str(mf_wrms))

        logpost = self.logPosterior(mf_params)
        print('LogPosterior = ', logpost)
        self.post.append('LogPosterior = ' + str(logpost))
        
        BIC = bic(mf_residual, len(sample_labels))
        print('BIC = ', BIC)
        self.post.append('BIC = '+ str(BIC))

        chi = chisq(mf_residual, self.yerr, np.zeros_like(mf_residual))
        print('$\chi^2$ = ', chi)
        self.post.append('$\chi^2$ = '+ str(chi))

        chir = chi/(len(mf_residual)-self.ndim-1)
        print('$\chi^2_{red}$ = ', chir)
        self.post.append('$\chi^2_{red}$ = '+ str(chir))

        chi = normalized_chi2(mf_residual, self.yerr, mf_params[-1])
        print('$\chi^2$ = ', chi)
        self.post.append('$\chi^2$ = '+ str(chi))

        chir = chi/(len(mf_residual)-self.ndim-1)
        print('$\chi^2_{red}$ = ', chir)
        self.post.append('$\chi^2_{red}$ = '+ str(chir))
        
        bic_quadra = self.quadra_fit()
        BF = (bic_quadra-BIC)/2
        print('Log Bayes Factor = ', BF)
        self.post.append('Bayes Factor = '+ str(BF))
        
        np.savetxt(self.outdir + self._target + '_' + self._varname + '_results.txt', self.post, fmt='%s')

        self.save_residuals()
        self.corner_plot()

    def mean_fit(self):
        return(np.median(self.flat_samples, axis=0))


    def display_fit(self, sampled_params, c='orange'):
        plt.clf()

        # Reconstruct full parameters from sampled + fixed
        full_params = {}
        keys = ["n_eta1", "n_eta2", "n_eta3", "n_eta4", "jitter"]
        idx = 0
        for key in keys:
            if self.fixed_params[key] is not None:
                full_params[key] = self.fixed_params[key]
            else:
                full_params[key] = sampled_params[idx]
                idx += 1

        # Extract full parameter values
        n1 = full_params["n_eta1"]
        n2 = full_params["n_eta2"]
        n3 = full_params["n_eta3"]
        n4 = full_params["n_eta4"]
        j = full_params["jitter"]

        # GP kernel and mean
        kernel = kernels.QuasiPeriodic(n1, n2, n3, n4) + kernels.WhiteNoise(j)
        mean = means.Constant(np.mean(self.y))
        gpOBJ = process.GP(kernel, mean, self.times, self.y, yerr=self.yerr)

        # Prediction
        tplot = np.linspace(np.min(self.times) - 50, np.max(self.times) + 50, 5000)
        y_mean, y_std, time = gpOBJ.prediction(kernel, mean, tplot)

        # Plot GP fit
        fig, ax = plt.subplots(2, 2, height_ratios=(2, 1), width_ratios=(3,2), figsize=(16, 8))
        ax[0, 0].errorbar(self.times, self.y, self.yerr, fmt='ko')
        ax[0, 0].plot(tplot, y_mean, c=c)
        ax[0, 0].fill_between(tplot, y_mean + y_std, y_mean - y_std, color=c, alpha=0.3)
        ax[0, 0].set_ylabel(self._varname, size=12, weight='bold')
        ax[0, 0].set_xlim(self.times.min()-20, self.times.max()+20)

        frequency, power = LombScargle(self.times, self.y, self.yerr).autopower(minimum_frequency=1/2000, maximum_frequency=1/1.25)
        ax[0, 1].plot(1/frequency, power, c=c)
        ax[0, 1].set_ylabel("power", size=12, weight='bold')
        ax[0, 1].set_xlabel("Period (d)", size=12, weight='bold')
        ax[0, 1].set_xscale('log')
        try :
            ax[0, 1].axvline(self.Prot, linestyle=':', color='b', alpha=0.5)
        except :
            pass    
        ls = LombScargle(self.times, self.y, self.yerr)
        fap = ls.false_alarm_level(0.1)
        ax[0, 1].axhline(fap, linestyle='-', color='k')
        fap = ls.false_alarm_level(0.01)
        ax[0, 1].axhline(fap, linestyle='--', color='k')
        fap = ls.false_alarm_level(0.001)
        ax[0, 1].axhline(fap, linestyle=':', color='k')
        
        # Residu
        y_sample, _, _ = gpOBJ.prediction(kernel, mean, self.times)
        y_sample -= self.y
        self.residuals = y_sample

        ax[1, 0].errorbar(self.times, y_sample, self.yerr, fmt='ko')
        ax[1, 0].set_xlabel('rjd (d)', size=12, weight='bold')
        ax[1, 0].set_ylabel('residuals', size=12, weight='bold')

        wrms = np.sqrt(mean_squared_error(y_sample, np.zeros_like(y_sample), sample_weight=1/self.yerr))
        ax[1, 0].annotate("wRMS :  " + str(np.round(wrms, decimals=3)), (self.times[2], np.max(y_sample)))
        fig.delaxes(ax[1, 1])

        plt.savefig(self.outdir + self._target + '_' + self._varname + '_fit.png', format="png", 
                    bbox_inches="tight", transparent=True)

        return y_sample, wrms

    def display_data(self, c='orange'):
        plt.clf()

        # Plot GP fit
        fig, ax = plt.subplots(1, 2, width_ratios=(3,2), figsize=(16, 5))
        ax[0].errorbar(self.times, self.y, self.yerr, fmt='ko')
        ax[0].set_ylabel(self._varname, size=12, weight='bold')
        ax[0].set_xlim(self.times.min()-20, self.times.max()+20)

        frequency, power = LombScargle(self.times, self.y, self.yerr).autopower(minimum_frequency=1/2000, maximum_frequency=1/1.25)
        ax[1].plot(1/frequency, power, c=c)
        ax[1].set_ylabel("power", size=12, weight='bold')
        ax[1].set_xlabel("Period (d)", size=12, weight='bold')
        ax[1].set_xscale('log')
        try :
            ax[1].axvline(self.Prot, linestyle=':', color='b', alpha=0.5)
        except :
            pass    
        ls = LombScargle(self.times, self.y, self.yerr)
        fap = ls.false_alarm_level(0.1)
        ax[1].axhline(fap, linestyle='-', color='k')
        fap = ls.false_alarm_level(0.01)
        ax[1].axhline(fap, linestyle='--', color='k')
        fap = ls.false_alarm_level(0.001)
        ax[1].axhline(fap, linestyle=':', color='k')
       
        plt.savefig(self.outdir + self._target + '_' + self._varname + '_data.png', format="png", 
                    bbox_inches="tight", transparent=True)

    
    def save_residuals(self):
        with open(self.outdir + self._target + '_' + self._varname + '_residuals.rdb', 'w') as file:
            # First line
            file.write('rjd \t' + self._varname + '\t' + self._svarname + '\n')
            file.write('\n')
            for i in range(len(self.times)):
                file.write(str(self.times[i]) + '\t' + str(self.residuals[i]) + '\t' + str(self.yerr[i]) + '\n')
        file.close()
        
    def quadra_fit(self, degree=3):
        #fit quadra
        quadra_fit = np.poly1d(np.polyfit(self.times, self.y, degree, w=1/self.yerr))
        residuals_quadra = self.y - quadra_fit(self.times)
        bic_quadra = bic(residuals_quadra, degree+1)
        return bic_quadra
        
    def check_all_dtemps(self, Prot=None, dProt=0):
        names = ['DTEMP3000', 'DTEMP3500', 'DTEMP4000', 'DTEMP4500', 'DTEMP5000', 'DTEMP5500', 'DTEMP6000']
        self.Prot = Prot
        power_temps_max_all = np.zeros_like(names)
        for n in range(len(names)):
            if Prot == None:
                freq_temp, power_temp = LombScargle(self.data['rjd'], self.data[names[n]]).autopower(minimum_frequency=0.0005, maximum_frequency=1/1.5) #nyquist_factor=15)
            elif dProt <= 0.05*Prot:
                freq_temp, power_temp = LombScargle(self.data['rjd'], self.data[names[n]]).autopower(minimum_frequency=1/(1.05*Prot), maximum_frequency=1/(0.95*Prot)) #nyquist_factor=15)
            else :
                freq_temp, power_temp = LombScargle(self.data['rjd'], self.data[names[n]]).autopower(minimum_frequency=1/(Prot+dProt), maximum_frequency=1/(Prot-dProt)) #nyquist_factor=15)
            
            try :
                peaks, _ = find_peaks(power_temp)
                sorted_peaks = np.argsort(power_temp[peaks])[::-1]  # Sort peaks by descending height
                top_peaks = sorted_peaks[0]  # Top 3 peaks
                power_temps_max_all[n] =  power_temp[peaks[top_peaks]]
                print(names[n] + f" Peak Period = {1/freq_temp[peaks[top_peaks]]:.4f} days, Value = {power_temp[peaks[top_peaks]]:.4f}")
                self.post.append(names[n] + f"Peak Period = {1/freq_temp[peaks[top_peaks]]:.4f} days, Value = {power_temp[peaks[top_peaks]]:.4f}")
            except :
                continue
        print('Best Possible DTEMP : ', names[np.argmax(power_temps_max_all)])
        self.post.append('Best Possible DTEMP : ' + names[np.argmax(power_temps_max_all)])

    def fit_sinusoid(self, P_init=None, return_fit_curve=False, c='orange'):
        """
        Fit y(t) = C + A * sin(2π t / P + phi) to the time series.

        Parameters
        ----------
        P_init : float, optional
            Initial guess for the period. If None, uses a simple automatic guess.
        return_fit_curve : bool
            If True, returns also (t_fit, y_fit) for plotting the model.

        Returns
        -------
        params : dict
            Best-fit parameters with uncertainties:
            { 'P', 'P_err', 'A', 'A_err', 'phi', 'phi_err', 'C', 'C_err' }
        """

        t = self.times
        y = self.y
        yerr = self.yerr

        # --- Sinusoidal model ---
        def model(t, P, A, phi, C):
            return C + A * np.sin(2 * np.pi * t / P + phi)

        # --- Initial guesses ---
        if P_init is None:
            # crude guess using peak-to-peak timescale
            P_init = (t.max() - t.min()) / 3
        A_init = 0.5 * (np.nanmax(y) - np.nanmin(y))
        phi_init = 0.0
        C_init = np.nanmedian(y)

        p0 = [P_init, A_init, phi_init, C_init]

        # --- Fit ---
        popt, pcov = curve_fit(
            model, t, y, p0=p0, sigma=yerr, absolute_sigma=True,
            maxfev=10000 ) #, bounds = (0, [200, 10*np.nanmax(y), 2*np.pi, np.nanmax(y)])
        #)

        P, A, phi, C = popt
        perr = np.sqrt(np.diag(pcov))
        P_err, A_err, phi_err, C_err = perr

        result = {
            "P": P, "P_err": P_err,
            "A": A, "A_err": A_err,
            "phi": phi, "phi_err": phi_err,
            "C": C, "C_err": C_err
        }

        if return_fit_curve:
            # Smooth model curve for plotting
            t_fit = np.linspace(t.min()-50, t.max()+50, 10000)
            y_fit = model(t_fit, *popt)
            #return result, (t_fit, y_fit)
            # Plot GP fit
            fig, ax = plt.subplots(2, 2, height_ratios=(2, 1), width_ratios=(3,2), figsize=(16, 8))
            ax[0, 0].errorbar(self.times, self.y, self.yerr, fmt='ko')
            ax[0, 0].plot(t_fit, y_fit, c=c)
            ax[0, 0].fill_between(t_fit, model(t_fit, *(popt + perr)), model(t_fit, *(popt - perr)), color=c, alpha=0.3)
            ax[0, 0].set_ylabel(self._varname, size=12, weight='bold')
            ax[0, 0].set_xlim(t.min()-20, t.max()+20)
    
            frequency, power = LombScargle(self.times, self.y, self.yerr).autopower(minimum_frequency=1/2000, maximum_frequency=1/1.25)
            ax[0, 1].plot(1/frequency, power, c=c)
            ax[0, 1].set_ylabel("power", size=12, weight='bold')
            ax[0, 1].set_xlabel("Period (d)", size=12, weight='bold')
            ax[0, 1].set_xscale('log')
            try :
                ax[0, 1].axvline(self.Prot, linestyle=':', color='b', alpha=0.5)
            except :
                pass    
            ls = LombScargle(self.times, self.y, self.yerr)
            fap = ls.false_alarm_level(0.1)
            ax[0, 1].axhline(fap, linestyle='-', color='k')
            fap = ls.false_alarm_level(0.01)
            ax[0, 1].axhline(fap, linestyle='--', color='k')
            fap = ls.false_alarm_level(0.001)
            ax[0, 1].axhline(fap, linestyle=':', color='k')
            
            # Residu
            y_sample = model(self.times, *popt)
            y_sample -= self.y
            #self.residuals = y_sample
    
            ax[1, 0].errorbar(self.times, y_sample, self.yerr, fmt='ko')
            ax[1, 0].set_xlabel('rjd (d)', size=12, weight='bold')
            ax[1, 0].set_ylabel('residuals', size=12, weight='bold')
    
            wrms = np.sqrt(mean_squared_error(y_sample, np.zeros_like(y_sample), sample_weight=1/self.yerr))
            ax[1, 0].annotate("wRMS :  " + str(np.round(wrms, decimals=3)), (self.times[2], np.max(y_sample)))
            fig.delaxes(ax[1, 1])

            plt.savefig(self.outdir + self._target + '_' + self._varname + '_sin_fit.png', format="png", bbox_inches="tight", transparent=True)

        return result

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
    
def bic(residuals, n_params):
    n = len(residuals)
    BIC = n*np.log(np.sum(residuals**2)/n) + n_params*np.log(n)
    return BIC
