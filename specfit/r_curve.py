from astropy.io import fits
import numpy as np
ID = 61234
f = fits.getdata(f'/data/DD6585/final_cal2/jw06585004001_s{ID}_x1d.fits')
wav_obs = f['WAVELENGTH']

oversample = 5
r = fits.getdata('specfit/res/jwst_nirspec_prism_disp.fits')
# x is an array of wavelengths with equal separations in R space 
import datetime
start = datetime.datetime.now()
import matplotlib.pyplot as plt
fig, ax = plt.subplots()

wav_obs_fine = [0.95*wav_obs[0]]
while wav_obs_fine[-1] < 1.05*wav_obs[-1]:
    R_val = np.interp(wav_obs_fine[-1], r['WAVELENGTH'], r['R']) #* f_LSF
    dwav = wav_obs_fine[-1]/R_val/oversample
    wav_obs_fine.append(wav_obs_fine[-1] + dwav)
wav_obs_fine = np.array(wav_obs_fine)
print(len(wav_obs_fine))

ax.plot(wav_obs_fine[:-1], np.diff(wav_obs_fine))
end = datetime.datetime.now()

dL = r['WAVELENGTH']/r['R']
# dL = (dL[1:]+dL[:-1])/2
wav_obs_fine = np.cumsum(dL/oversample)+r['WAVELENGTH'][0]
print(len(wav_obs_fine))
plt.plot(wav_obs_fine[:-1], np.diff(wav_obs_fine))
plt.show()


print(end-start)