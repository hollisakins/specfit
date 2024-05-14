from .model import model_spectrum, _get_wavelength_sampling
import bayesianflexifit as bff
import numpy as np
from copy import copy
import tqdm
import spectres
from astropy.convolution import convolve



def lnlike(data, model_components):
    x, y, yerr = data
    mod = model_spectrum(x, model_components).flux
    chisq = np.sum(np.power((y-mod)/yerr,2))
    if 'noise_scaling' in model_components:
        return -len(y)*np.log(model_components['noise_scaling']) - 0.5*chisq
    else:
        return -0.5*chisq

class fitter:
    def __init__(self, ID, data, fit_instructions, run, n_posterior):
        self.fit_instructions = fit_instructions
        ID = str(ID)
        x, y, yerr = data
        cond = (yerr!=0)&~np.isnan(y)&~np.isnan(yerr)
        x, y, yerr = x[cond], y[cond], yerr[cond]
        data = x, y, yerr
        self.wav_obs = x
        
        self.wav_obs_fine, self.oversample = _get_wavelength_sampling(self.wav_obs, fit_instructions)
        fit_instructions['wav_obs_fine'] = self.wav_obs_fine

        self.bff_fitter = bff.fit(ID, data, lnlike, fit_instructions, run=run,
                                  time_calls=True, n_posterior=n_posterior)


    def fit(self, n_live, verbose=True, use_MPI=False):
        self.bff_fitter.fit(verbose=verbose, n_live=n_live, use_MPI=use_MPI)

    def get_advanced_quantities(self):

        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if rank==0:
            nposterior = len(self.bff_fitter.posterior[self.bff_fitter.params[0]])
            mc = copy(self.bff_fitter.model_parameters)
            self.nposterior = nposterior
            line_names = [k.split('f_')[-1] for k in list(mc.keys()) if k.endswith('narrow') or k.endswith('broad')]
            self.line_grid = {l:np.zeros((nposterior,len(self.wav_obs))) for l in line_names}
            self.cont_grid = np.zeros((nposterior,len(self.wav_obs)))
            self.model_grid = np.zeros((nposterior,len(self.wav_obs)))
            self.line_grid_fine = {l:np.zeros((nposterior,len(self.wav_obs_fine))) for l in line_names}
            self.cont_grid_fine = np.zeros((nposterior,len(self.wav_obs_fine)))
            self.model_grid_fine = np.zeros((nposterior,len(self.wav_obs_fine)))
            for i in tqdm.tqdm(range(nposterior)):
                for k in self.bff_fitter.posterior:
                    mc[k] = self.bff_fitter.posterior[k][i]
                mod = model_spectrum(self.wav_obs,mc)
                self.model_grid[i] = mod.flux
                self.cont_grid[i] = mod.cont
                self.model_grid_fine[i] = mod.flux_fine
                self.cont_grid_fine[i] = mod.cont_fine
            
                for line in mod.lines:            
                    lij = convolve(mod.lines[line], mod.kernel)
                    self.line_grid[line][i,:] = spectres.spectres(self.wav_obs, self.wav_obs_fine, lij, fill=0, verbose=False)
                    self.line_grid_fine[line][i,:] = lij

            self.flam_med = np.median(self.model_grid,axis=0)
            self.flam_16 = np.percentile(self.model_grid,16,axis=0)
            self.flam_84 = np.percentile(self.model_grid,84,axis=0)

            if 'redshift' in self.bff_fitter.posterior:        
                z = np.median(self.bff_fitter.posterior['redshift'])
            else:
                z = mc['redshift']
            
            self.wav_rest = self.wav_obs/(1+z)*1e4
            self.wav_rest_fine = self.wav_obs_fine/(1+z)*1e4

            self.posterior = self.bff_fitter.posterior

        
        # wav_bins = (self.wav_obs[1:]+self.wav_obs[:-1])/2
        # wav_bins = np.append(self.wav_obs[0]-np.diff(self.wav_obs)[0]/2,wav_bins)
        # wav_bins = np.append(wav_bins, self.wav_obs[-1]+np.diff(self.wav_obs)[-1]/2)
        # self.flux, _, _ = binned_statistic(self.wav_obs_fine, y, bins=wav_bins, statistic='mean')


        #     t2 = fits.BinTableHDU.from_columns(fits.ColDefs(columns), header=fits.Header({'EXTNAME':'PROPERTIES'}))

        #     tm1 = fits.PrimaryHDU(header=fits.Header({'EXTEND':'T'}))
        #     t = fits.HDUList([tm1,t0,t1,t2])
        #     t.writeto(fitter.fname+'results.fits', overwrite=True)





        #     # print(likelihood(param_dict, prism_wav, prism_flam*1e22, prism_flam_err*1e22))

        #     # initial = np.array([m_true, b_true, np.log(f_true)]) + 0.1 * np.random.randn(3)
        #     # soln = minimize(nll, p0, args=(x, y, yerr))

        #     # # parameter setup
        #     # from scipy.stats import norm
        #     # from nautilus import Prior, Sampler
        #     # prior = Prior()
        #     # prior.add_parameter('redshift', dist=7.1358)
        #     # prior.add_parameter('f_cont', dist=(0,100))
        #     # prior.add_parameter('broad_fwhm', dist=(1000, 6000))
        #     # prior.add_parameter('narrow_fwhm', dist=norm(loc=100,scale=10))
        #     # prior.add_parameter('f_Hb_broad', dist=(0,100)) # in 1e-22 erg/s/cm2
        #     # prior.add_parameter('f_Hb_narrow', dist=(0,100)) # in 1e-22 erg/s/cm2
        #     # prior.add_parameter('f_OIII4959', dist=(0,100)) # in 1e-22 erg/s/cm2
        #     # prior.add_parameter('f_OIII5007', dist=(0,100)) # in 1e-22 erg/s/cm2
        #     # prior.add_parameter('f_LSF', dist=norm(loc=1.5,scale=0.2)) 
        #     # prior.add_parameter('noise_scaling', dist=1.0) 

        #     # sampler = Sampler(prior, likelihood, likelihood_args=data, 
        #     #                 n_live=1000, 
        #     #                 filepath='runs/ceersz7m1_run1.hdf5',
        #     #                 pool=18)
        #     # sampler.run(verbose=True, discard_exploration=True)

        #     # import spectres
        #     # spectres.spectres()

        #     # oversample = 4  # Number of samples per FWHM at resolution R
        #     # new_wavs = x_fine

        #     #             # spectrum = np.interp(new_wavs, redshifted_wavs, spectrum)
        #     #             spectrum = spectres.spectres(new_wavs, redshifted_wavs,
        #     #                                          spectrum, fill=0)
        #     #             redshifted_wavs = new_wavs

        #     #             sigma_pix = oversample/2.35  # sigma width of kernel in pixels
        #     #             k_size = 4*int(sigma_pix+1)
        #     #             x_kernel_pix = np.arange(-k_size, k_size+1)

        #     #             kernel = np.exp(-(x_kernel_pix**2)/(2*sigma_pix**2))
        #     #             kernel /= np.trapz(kernel)  # Explicitly normalise kernel

        #     #             # Disperse non-uniformly sampled spectrum
        #     #             spectrum = np.convolve(spectrum, kernel, mode="valid")
        #     #             redshifted_wavs = redshifted_wavs[k_size:-k_size]

        #     #         # Converted to using spectres in response to issue with interp,
        #     #         # see https://github.com/ACCarnall/bagpipes/issues/15
        #     #         # fluxes = np.interp(self.spec_wavs, redshifted_wavs,
        #     #         #                    spectrum, left=0, right=0)

        #     #         fluxes = spectres.spectres(self.spec_wavs, redshifted_wavs,
        #     #                                    spectrum, fill=0)

