from astropy.table import Table
from astropy.io import fits
import numpy as np

# global config stuff

# vacuum wavelengths compiled from NIST or SDSS data

linelist = Table(names=['name','label','wav'],dtype=[str,str,float])
linelist.add_row(['Lya',       r'${\rm Ly}\alpha$',                                      1215.670])
linelist.add_row(['CII',       r'${\rm C}\,\textsc{ii}$',                                1335.708])
linelist.add_row(['CIV',       r'${\rm C}\,\textsc{iv}]\,\lambda\lambda 1548,1551$',     1549.480])
linelist.add_row(['HeII',      r'He\,II\,$\lambda 1640$',                                1640.400])
linelist.add_row(['CIII',      r'${\rm C}\,\textsc{iii}]\,\lambda 1909$',                1908.734])
linelist.add_row(['NeIV',      r'$[{\rm Ne}\,\textsc{iv}]\,\lambda 2423$',               2439.500])
linelist.add_row(['FeXI',      r'$[{\rm Fe}\,\textsc{xi}]$',                             2649.500])
linelist.add_row(['MgII',      r'$[{\rm Mg}\,\textsc{ii}]\,\lambda\lambda 2796,2803$',   2799.117])
linelist.add_row(['HeI3188',   r'He\,I\,$\lambda 3188$',                                 3188.667])
linelist.add_row(['HeIHeII',   r'He\,I\,$\lambda 3188+$He\,II\,$\lambda 3203$',          3196.352])
linelist.add_row(['HeII3203',  r'He\,II\,$\lambda 3203$',                                3204.037])
linelist.add_row(['NeV3427',   r'$[{\rm Ne}\,\textsc{v}]\,\lambda 3427$',                3427.000])
linelist.add_row(['OII3727',   r'$[{\rm O}\,\textsc{ii}]\,\lambda\lambda 3726,3729$',    3728.484])
linelist.add_row(['NeIII3869', r'$[{\rm Ne}\,\textsc{iii}]\,\lambda 3869$',              3869.860])
linelist.add_row(['NeIII3967', r'$[{\rm Ne}\,\textsc{iii}]\,\lambda 3967$',              3968.590])
linelist.add_row(['Hd',        r'${\rm H}\delta$',                                       4102.890])
linelist.add_row(['Hg',        r'${\rm H}\gamma$',                                       4341.680])
linelist.add_row(['HgOIII4363',r'${\rm H}\gamma+[{\rm O}\,\textsc{iii}]\,\lambda 4363$', 4353.058])
linelist.add_row(['OIII4363',  r'$[{\rm O}\,\textsc{iii}]\,\lambda 4363$',               4364.436])
linelist.add_row(['FeII',      r'$[{\rm Fe}\,\textsc{ii}]\,\lambda 4287$',               4288.599])
linelist.add_row(['HeII4685',  r'He\,II\,$\lambda 4685$',                                4687.016])
linelist.add_row(['Hb',        r'${\rm H}\beta$',                                        4862.710])
linelist.add_row(['OIII4959',  r'$[{\rm O}\,\textsc{iii}]\,\lambda 4959$',               4960.295])
linelist.add_row(['OIII5007',  r'$[{\rm O}\,\textsc{iii}]\,\lambda 5007$',               5008.240])
linelist.add_row(['HeI5876',   r'He\,I\,$\lambda 5876$',                                 5877.249])
linelist.add_row(['Ha',        r'${\rm H}\alpha$',                                       6564.600])
linelist.add_row(['FeX',       r'$[{\rm Fe}\,\textsc{x}]$',                              6376.270])

import os
dirname = os.path.dirname(__file__)
res = {}
res['prism'] = fits.getdata(os.path.join(dirname,'res/jwst_nirspec_prism_disp.fits'))
res['g395m'] = fits.getdata(os.path.join(dirname,'res/jwst_nirspec_g395m_disp.fits'))


igm_redshifts = np.arange(0.0, 20 + 0.01, 0.01)
igm_wavelengths = np.arange(1.0, 1225.01, 1.0)
raw_igm_grid = fits.open(os.path.join(dirname, 'grids/d_igm_grid_inoue14.fits'))[1].data
