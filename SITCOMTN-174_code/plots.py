import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
import pandas as pd

from lsst.daf.butler import Butler
import os, sys
sys.path.append(os.path.abspath(os.path.join('.', '..', 'src')))

import data_utils as du
import psf_utils as pu
import DCR

from matplotlib import rcParams
rcParams['font.family'] = "STIXGeneral"
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.size'] = 14
rcParams['font.weight'] = 'medium'


date = '_20251023'
week = '39'
# single_visit_catalog_fp = f'../w_2025_{week}/color/single_visit_catalog{date}.h5'
# dcr_catalog_fp = f'../w_2025_{week}/color/dcr_catalog{date}.h5'
# match_catalog_fp = f'../w_2025_{week}/color/matched_catalog{date}.csv'
# coadd_catalog_fp = f'../w_2025_{week}/coadd_catalog_w{week}.h5'

date = ''
single_visit_catalog_fp = f'../w_2025_{week}/no_color/single_visit_catalog{date}.h5'
dcr_catalog_fp = f'../w_2025_{week}/no_color/dcr_catalog{date}.h5'
match_catalog_fp = f'../w_2025_{week}/no_color/matched_catalog{date}.csv'
coadd_catalog_fp = f'../w_2025_{week}/coadd_catalog_w{week}.h5'
#savepath = f'../w_2025_{week}/no_color/plots/shape_residuals_vs_dcr_{bandname}.png'

print('Loading files', flush=True)

with h5py.File(single_visit_catalog_fp, 'r') as f:
    print(f.keys())
    band = f['band'][:].astype('str')
    visit = f['visit'][:]
    collection = f['comments'][:].astype('str')[0]
with h5py.File(dcr_catalog_fp, 'r') as f:
    #print(f.keys())

    dcr1 = f['dcr_1'][:]
    dcr2 = f['dcr_2'][:]
    #zenith_angle = f['zenith_angle'][:]
    #parallactic_angle = f['parallactic_angle'][:]
    # ixx = f['sky_ixx'][:]
    # iyy = f['sky_iyy'][:]
    # ixy = f['sky_ixy'][:]

    # model_ixx = f['model_sky_ixx'][:]
    # model_iyy = f['model_sky_iyy'][:]
    # model_ixy = f['model_sky_ixy'][:]

    ixx = f['azel_ixx'][:]
    iyy = f['azel_iyy'][:]
    ixy = f['azel_ixy'][:]

    model_ixx = f['model_azel_ixx'][:]
    model_iyy = f['model_azel_iyy'][:]
    model_ixy = f['model_azel_ixy'][:]

    # ixx = f['direct_azel_ixx'][:]
    # iyy = f['direct_azel_iyy'][:]
    # ixy = f['direct_azel_ixy'][:]

    # model_ixx = f['direct_model_azel_ixx'][:]
    # model_iyy = f['direct_model_azel_iyy'][:]
    # model_ixy = f['direct_model_azel_ixy'][:]
    # ixx = f['ixx'][:]
    # iyy = f['iyy'][:]
    # ixy = f['ixy'][:]

    # model_ixx = f['model_ixx'][:]
    # model_iyy = f['model_iyy'][:]
    # model_ixy = f['model_ixy'][:]
    
with h5py.File(coadd_catalog_fp, 'r') as f:
    #print(f['data'].keys())
    #coadd_g_flux = f['g_calibFlux'][:]
    #coadd_i_flux = f['i_calibFlux'][:]
    coadd_r_flux = f['r_calibFlux'][:]
    coadd_z_flux = f['z_calibFlux'][:]
matched_cat = pd.read_csv(match_catalog_fp)
idx = matched_cat['idx']
del matched_cat


e1, e2, T = pu.moment2ellipticity(ixx, iyy, ixy)
model_e1, model_e2, model_T = pu.moment2ellipticity(model_ixx, model_iyy, model_ixy)

de1 = e1 - model_e1
de2 = e2 - model_e2
del ixx, iyy, ixy, e2, T, model_ixx, model_iyy, model_ixy

#single_visit_g_flux = coadd_g_flux[idx]
#single_visit_i_flux = coadd_i_flux[idx]
single_visit_r_flux = coadd_r_flux[idx]
single_visit_z_flux = coadd_z_flux[idx]

#single_visit_g_mag = -2.5*np.log10(single_visit_g_flux)
#single_visit_i_mag = -2.5*np.log10(single_visit_i_flux)
single_visit_r_mag = -2.5*np.log10(single_visit_r_flux)
single_visit_z_mag = -2.5*np.log10(single_visit_z_flux)

color = single_visit_r_mag - single_visit_z_mag #single_visit_g_mag - single_visit_i_mag
quantiles = np.quantile(color[np.isfinite(color)], [0.25, 0.5, 0.75])

nan_mask = np.isfinite(color)
all_mask = nan_mask & ((color > quantiles[0]) & (color < quantiles[-1]))
mask1 = nan_mask & ((color > quantiles[0]) & (color < quantiles[1]))
mask2 = nan_mask & ((color > quantiles[1]) & (color < quantiles[2]))
mask3 = nan_mask & ((color > quantiles[2]) & (color < 3.5))

#del color, single_visit_g_mag, single_visit_i_mag, single_visit_g_flux, single_visit_i_flux
del color, single_visit_r_mag, single_visit_z_mag, single_visit_r_flux, single_visit_z_flux

print('Plotting DCR', flush=True)

for bandname in 'ugrizy':
    colorname = 'r-z'
    
    fig, ax = plt.subplots(nrows=2, figsize=(8,8))
    fig.suptitle(f'band: {bandname}\nno. of stars: {np.sum(band[all_mask]==bandname)}, no. of visits: {len(np.unique(visit[band==bandname]))}', fontsize=20, y=1.0) #\n{collection}

    for mask, c, m, l in zip([mask1, mask2, mask3, all_mask], 
                             ["c", "orange", "m", "k"], 
                             ['o', 'o', 'o', 'P'], 
                             [f'{quantiles[0]:.2f} < {colorname} < {quantiles[1]:.2f}', 
                              f'{quantiles[1]:.2f} < {colorname} < {quantiles[2]:.2f}', 
                              f'{quantiles[2]:.2f} < {colorname} < 3.5', 
                              f'{quantiles[0]:.2f} < {colorname} < 3.5']):
    
        de_masked, dcr_masked = de1[mask], dcr1[mask]
        finite_mask = np.isfinite(de_masked) & np.isfinite(dcr_masked) & (band[mask]==bandname)
        print(np.sum(finite_mask))
        binned_de, dcr_bin_edges, _ = binned_statistic(dcr_masked[finite_mask], 
                                                       de_masked[finite_mask], 
                                                       statistic='mean', 
                                       bins=7, range=[-1.5, 1.5])
        de_error, _, _ = binned_statistic(dcr_masked[finite_mask], 
                                          de_masked[finite_mask], 
                                          statistic='std', 
                                          bins=7, range=[-1.5, 1.5])
        de_bin_count, _, _ = binned_statistic(dcr_masked[finite_mask], 
                                              de_masked[finite_mask], 
                                              statistic='count', 
                                              bins=7, range=[-1.5, 1.5])
        
        de_std_error = de_error/np.sqrt(de_bin_count)
        dcr_bins = 0.5*(dcr_bin_edges[1:] + dcr_bin_edges[:-1])
    
        ax[0].errorbar(dcr_bins, binned_de, yerr=de_std_error, 
                              ls='None', ms=7, alpha=0.5, capsize=1,
                              color=c, marker=m, label=l)
        ax[0].axhline(y=0, color='black', linewidth=2, linestyle='-', zorder=-1)
    
    ax[0].legend()
    ax[0].set_ylabel(r'$\delta e_1$')
    ax[0].set_xlabel(r'$\text{DCR}_1$')
    #ax[0].set_ylim(-3e-3, 3e-3)
    ax[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    
    for mask, c, m, l in zip([mask1, mask2, mask3, all_mask], 
                             ["c", "m", "orange","k"], 
                             ['o', 'o', 'o', 'P'], 
                             [f'{quantiles[0]:.2f} < {colorname} < {quantiles[1]:.2f}', 
                              f'{quantiles[1]:.2f} < {colorname} < {quantiles[2]:.2f}', 
                              f'{quantiles[2]:.2f} < {colorname} < 3.5', 
                              f'{quantiles[0]:.2f} < {colorname} < 3.5']):
    
        de_masked, dcr_masked = de2[mask], dcr2[mask]
        finite_mask = np.isfinite(de_masked) & np.isfinite(dcr_masked) & (band[mask]==bandname)
        
        binned_de, dcr_bin_edges, _ = binned_statistic(dcr_masked[finite_mask], 
                                                       de_masked[finite_mask], 
                                                       statistic='mean', 
                                       bins=7, range=[-1.5, 1.5])
        de_error, _, _ = binned_statistic(dcr_masked[finite_mask], 
                                          de_masked[finite_mask], 
                                          statistic='std', 
                                          bins=7, range=[-1.5, 1.5])
        de_bin_count, _, _ = binned_statistic(dcr_masked[finite_mask], 
                                              de_masked[finite_mask], 
                                              statistic='count', 
                                              bins=7, range=[-1.5, 1.5])
        
        de_std_error = de_error/np.sqrt(de_bin_count)
        dcr_bins = 0.5*(dcr_bin_edges[1:] + dcr_bin_edges[:-1])
        
    
            
    
        ax[1].errorbar(dcr_bins, binned_de, yerr=de_std_error, 
                              ls='None', ms=7, alpha=0.5, capsize=1,
                              color=c, marker=m, label=l)
        ax[1].axhline(y=0, color='black', linewidth=2, linestyle='-', zorder=-1)
    
    #ax[1].legend()
    ax[1].set_ylabel(r'$\delta e_2$')
    ax[1].set_xlabel(r'$\text{DCR}_2$')
    #ax[1].set_ylim(-3e-3, 3e-3)
    ax[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    for i in range(2):
        ax[i].grid(lw=0.5, alpha=0.5)
        
    #savepath = f'../w_2025_{week}/color/plots/shape_residuals_vs_dcr_{bandname}{date}.png'
    savepath = f'../notebooks/technote_plots/shape_residuals_vs_dcr_no_correction_{bandname}.png'
    plt.savefig(savepath, dpi=300, bbox_inches='tight')