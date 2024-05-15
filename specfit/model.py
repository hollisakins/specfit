from . import config
import numpy as np
from astropy.convolution import convolve
import spectres
from .igm_model import igm
from scipy.stats import binned_statistic 
# import datetime
from copy import copy

def gauss(x, central_wav, fwhm_vel, norm):
    sigma_vel = fwhm_vel/2.355
    sigma = sigma_vel*central_wav/2.998e5
    sigma = np.max([sigma,np.diff(x)[np.argmin(np.abs(x-central_wav))]])
    gauss = np.exp(-0.5*np.power((x-central_wav)/sigma,2))
    xrange = np.abs((x-central_wav)/sigma)<5
    gauss /= np.trapz(gauss[xrange], x=x[xrange])
    gauss *= norm
    return gauss

def _get_wavelength_sampling(wav_obs, mc):
    if 'R_curve' in mc:
        oversample = mc['oversample']
        r = config.res[mc['R_curve']]
        # x is an array of wavelengths with equal separations in R space 
        wav_obs_fine = [0.95*wav_obs[0]]
        while wav_obs_fine[-1] < 1.05*wav_obs[-1]:
            R_val = np.interp(wav_obs_fine[-1], r['WAVELENGTH'], r['R'])
            dwav = wav_obs_fine[-1]/R_val/oversample
            wav_obs_fine.append(wav_obs_fine[-1] + dwav)
        wav_obs_fine = np.array(wav_obs_fine)
    else:
        oversample = mc['oversample']
        try:
            R_val = mc['R']
        except:
            R_val = 1000
        wav_obs_fine = [0.95*wav_obs[0]]
        while wav_obs_fine[-1] < 1.05*wav_obs[-1]:
            dwav = wav_obs_fine[-1]/R_val/oversample
            wav_obs_fine.append(wav_obs_fine[-1] + dwav)
        wav_obs_fine = np.array(wav_obs_fine)
    return wav_obs_fine, oversample
    
def _calzetti(wavs):
    """ Calculate the ratio A(lambda)/A(V) for the Calzetti et al.
    (2000) attenuation curve. """
    A_lambda = np.zeros_like(wavs)
    wavs_mic = wavs*10**-4
    mask1 = (wavs < 1200.)
    mask2 = (wavs < 6300.) & (wavs >= 1200.)
    mask3 = (wavs < 31000.) & (wavs >= 6300.)
    A_lambda[mask1] = ((wavs_mic[mask1]/0.12)**-0.77
                        * (4.05 + 2.695*(- 2.156 + 1.509/0.12
                                        - 0.198/0.12**2 + 0.011/0.12**3)))
    A_lambda[mask2] = (4.05 + 2.695*(- 2.156
                                        + 1.509/wavs_mic[mask2]
                                        - 0.198/wavs_mic[mask2]**2
                                        + 0.011/wavs_mic[mask2]**3))
    A_lambda[mask3] = 2.659*(-1.857 + 1.040/wavs_mic[mask3]) + 4.05
    A_lambda /= 4.05
    return A_lambda

def Salim(wavs, delta):
    A_cont_calz = _calzetti(wavs)
    Rv_m = 4.05/((4.05+1)*(4400./5500.)**delta - 4.05)
    A_cont = A_cont_calz*Rv_m*(wavs/5500.)**delta
    A_cont /= Rv_m
    return A_cont


class model_spectrum:
    def __init__(self, wav_obs, mc):
        self.mc = mc
        self.wav_obs = wav_obs
        self.compute_model()

    def compute_model(self):
        z = self.mc['redshift']
        self.wav_rest = self.wav_obs / (1+z) * 1e4
        
        ############## generate oversampled wavelength grid ##############
        if 'wav_obs_fine' in self.mc: # already computed wavelength grid, use that
            self.wav_obs_fine = self.mc['wav_obs_fine']
            oversample = self.mc['oversample']
        else:
            self.wav_obs_fine, oversample = _get_wavelength_sampling(self.wav_obs, self.mc)

        self.wav_rest_fine = self.wav_obs_fine / (1+z) * 1e4


        # different types of continuum models
        y = np.zeros_like(self.wav_obs_fine)
        if self.mc['cont_type'] == 'flat':
            y += 1
        elif self.mc['cont_type'] == 'linear':
            mask0 = np.array(np.zeros(len(y)),dtype=bool)
            breaks = np.append(np.min(self.wav_rest_fine),self.mc['breaks'])
            breaks = np.append(breaks, np.max(self.wav_rest_fine))
            slopes = [self.mc[f'slope{i}'] for i in range(len(self.mc['breaks'])+1)]
            for i in range(len(slopes)-1):
                mask1 = (self.wav_rest_fine > breaks[i])&(self.wav_rest_fine < breaks[i+1])
                mask2 = (self.wav_rest_fine > breaks[i+1])&(self.wav_rest_fine < breaks[i+2])
                if i==0:
                    y[mask1] = self.wav_rest_fine[mask1]**slopes[i]
                y[mask2] = self.wav_rest_fine[mask2]**slopes[i+1]
                y[mask1|mask0] /= y[mask1][-1]
                y[mask2] /= y[mask2][0]
                mask0 = mask0|mask1

        if 'Av' in self.mc:
            if 'logfscat' in self.mc:
                fscat = np.power(10.,self.mc['logfscat'])
            else:
                fscat = 0
            if 'delta' in self.mc:
                delta = self.mc['delta']
            else:
                delta = 0

            trans = np.power(10.,-self.mc['Av']*Salim(self.wav_rest_fine, delta)/2.5)
            y = y*trans + y*fscat


        y /= y[np.argmin(np.abs(self.wav_rest_fine - 5100.))]
        y *= self.mc['f5100']
            # y[y<0] = 0

        self.cont = copy(y)

        line_names, line_spec = [], []
        for line in config.linelist:
            if f"dv_{line['name']}" in self.mc:
                dv = self.mc[f"dv_{line['name']}"]
            else: dv=0
            if f"f_{line['name']}_narrow" in list(self.mc.keys()):
                g = gauss(self.wav_rest_fine, line['wav']*(1+dv/2.998e5), self.mc['narrow_fwhm'], self.mc[f"f_{line['name']}_narrow"])
                line_names.append(f"{line['name']}_narrow")
                line_spec.append(g)
                y += g
            if f"f_{line['name']}_broad" in list(self.mc.keys()):
                g = gauss(self.wav_rest_fine, line['wav']*(1+dv/2.998e5), self.mc['broad_fwhm'], self.mc[f"f_{line['name']}_broad"])
                line_names.append(f"{line['name']}_broad")
                line_spec.append(g)
                y += g
        self.lines = {n:s for n,s in zip(line_names,line_spec)}
        # for line in list(broad_lines.keys()):
        #     y += gauss(x, broad_lines[line] / 1e4 * (1+mc['redshift']), mc['broad_fwhm'], mc[f'f_{line}'])
        
        # print('before igm', datetime.datetime.now()-start)
        # y *= igm(self.wav_rest_fine).trans(z)
        y[self.wav_rest_fine<1215.67] = 0
    
        ############## convolve model with the LSF ##############
        if 'R_curve' in self.mc:
            if 'f_LSF' in self.mc: f_LSF = self.mc['f_LSF']; 
            else: f_LSF=1
            sigma_pix = oversample/2.35/f_LSF  # sigma width of kernel in pixels
            k_size = 5*int(sigma_pix+1)
            x_kernel_pix = np.arange(-k_size, k_size+1)
            kernel = np.exp(-(x_kernel_pix**2)/(2*sigma_pix**2))
            kernel /= np.trapz(kernel)  # Explicitly normalise kernel
            self.kernel = kernel
            y = convolve(y, kernel)
        self.flux_fine = copy(y)
        self.cont_fine = copy(self.cont)
        # wav_bins = (self.wav_obs[1:]+self.wav_obs[:-1])/2
        # wav_bins = np.append(self.wav_obs[0]-np.diff(self.wav_obs)[0]/2,wav_bins)
        # wav_bins = np.append(wav_bins, self.wav_obs[-1]+np.diff(self.wav_obs)[-1]/2)
        # self.flux, _, _ = binned_statistic(self.wav_obs_fine, y, bins=wav_bins, statistic='mean')
        self.flux = spectres.spectres(self.wav_obs, self.wav_obs_fine, y, fill=0, verbose=False)
        self.cont = spectres.spectres(self.wav_obs, self.wav_obs_fine, convolve(self.cont, self.kernel), fill=0, verbose=False)

    # def lnlike(self):
    #     chisq = np.sum(np.power((self.flux-self.model_spectrum)/self.error,2))
    #     if 'noise_scaling' in self.mc:
    #         lnlike = -len(y)*np.log(self.mc['noise_scaling']) - 0.5*chisq
    #     else:
    #         lnlike = -0.5*chisq
    #     return lnlike
