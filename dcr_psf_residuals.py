"""
Reproduce the SITCOMTN-174 PSF second-moment residual batch (Figs 2-6) from the
``refit_psf_star`` PSF-star table produced by finalizeCharacterization.py.

Data access follows the rest of this repo: a visit->parquet mapping built with
``butler.getURI("refit_psf_star", ...)`` (see getData.py) is read directly with
polars, one parquet per visit, with no butler call inside the loop. The butler is
used once, at startup, only to load ``visit_table`` for the per-visit DCR inputs.

The ``refit_psf_star`` table (re-run with the user's finalizeCharacterization
changes) carries:
  - second moments in alt/az (horizon) coordinates: shape_Ialtalt/Iazaz/Ialtaz
    and psfShape_Ialtalt/Iazaz/Ialtaz  (requires do_add_sky_moments=True),
  - second moments in sky RA/Dec: shape_Iuu/Ivv/Iuv, psfShape_Iuu/Ivv/Iuv,
  - per-band FGCM magnitudes: fgcm_mag_<band>  (requires do_add_fgcm_photometry=True).

DCR inputs (zenith angle z, parallactic angle q) are taken per visit from
``visit_table`` and combined as (technote Eq. 1):
    DCR1 = tan^2(z) * cos(2q)
    DCR2 = tan^2(z) * sin(2q)

Residual definitions (technote Eq. 2, in the chosen frame):
    T  = Ixx + Iyy ,  e1 = (Ixx - Iyy)/T ,  e2 = 2 Ixy / T
    de1 = e1_star - e1_model , de2 = e2_star - e2_model , dT_T = (T_star - T_model)/T_star

The 1-D binning reuses the streaming ``meanify1D_wrms`` from SNR_vs_dT.py.
"""

import numpy as np
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os
os.environ["POLARS_MAX_THREADS"] = "1"
import polars

import pickle
import argparse

from astropy.time import Time
from astropy.coordinates import EarthLocation
import astropy.units as u


# ----------------------------------------------------------------------------
# Site (Cerro Pachon) - latitude value matches SITCOMTN-174_code/generate_dcr_catalog.py
# ----------------------------------------------------------------------------
SITE_LAT_DEG = -30.244633
CERRO_PACHON = EarthLocation(lat=-30.244633 * u.deg, lon=-70.7366 * u.deg, height=2722 * u.m)

ALL_BANDS = 'ugrizy'

# Frame -> (source moment cols, psf-model moment cols)
FRAME_COLUMNS = {
    'altaz': (('shape_Ialtalt', 'shape_Iazaz', 'shape_Ialtaz'),
              ('psfShape_Ialtalt', 'psfShape_Iazaz', 'psfShape_Ialtaz')),
    'radec': (('shape_Iuu', 'shape_Ivv', 'shape_Iuv'),
              ('psfShape_Iuu', 'psfShape_Ivv', 'psfShape_Iuv')),
}

# 4-series color-bin style (3 tertiles + "all"), following plots.py
SERIES_COLORS = ['c', 'orange', 'm', 'k']
SERIES_MARKERS = ['o', 'o', 'o', 'P']


# ----------------------------------------------------------------------------
# Streaming 1D binned mean/std/count  (copied from SNR_vs_dT.py)
# ----------------------------------------------------------------------------
class meanify1D_wrms():
    """
    Take data, build a 1D average with weighted RMS.
    O(1) memory implementation - keeps running sum/count per bin.
    """
    def __init__(self, bin_spacing=0.3, x_min=0, x_max=1000):
        self.bin_spacing = bin_spacing
        self.x_min = x_min
        self.x_max = x_max

        # Pre-allocate bins
        self.nbin = int((x_max - x_min) / bin_spacing) + 1
        self.binning = np.linspace(x_min, x_max, self.nbin)

        # Running statistics per bin (O(1) memory)
        self._sum = np.zeros(self.nbin - 1)
        self._sum_sq = np.zeros(self.nbin - 1)
        self._count = np.zeros(self.nbin - 1)

        # Bin centers
        self.x0 = self.binning[:-1] + (self.binning[1] - self.binning[0]) / 2.0

    def add_data(self, coord, param):
        """Add new data - accumulates directly into bins (O(1) memory)."""
        # Filter valid data
        valid = np.isfinite(coord) & np.isfinite(param)
        coord = coord[valid]
        param = param[valid]

        # Find bin indices for each data point
        bin_indices = np.digitize(coord, self.binning) - 1

        # Clip to valid bin range
        valid_bins = (bin_indices >= 0) & (bin_indices < self.nbin - 1)
        bin_indices = bin_indices[valid_bins]
        param = param[valid_bins]

        # Accumulate into bins
        np.add.at(self._sum, bin_indices, param)
        np.add.at(self._sum_sq, bin_indices, param ** 2)
        np.add.at(self._count, bin_indices, 1)

    def meanify(self):
        """Compute final statistics from accumulated sums."""
        with np.errstate(divide='ignore', invalid='ignore'):
            self.average = self._sum / self._count
            variance = (self._sum_sq / self._count) - (self.average ** 2)
            self.std = np.sqrt(variance)
            self.count = self._count


def bin_1d(x, y, x_min, x_max, nbins):
    """Convenience wrapper: bin y against x and return (centers, mean, std_err, count)."""
    bin_spacing = (x_max - x_min) / nbins
    m = meanify1D_wrms(bin_spacing=bin_spacing, x_min=x_min, x_max=x_max)
    m.add_data(np.asarray(x), np.asarray(y))
    m.meanify()
    with np.errstate(divide='ignore', invalid='ignore'):
        std_err = m.std / np.sqrt(m.count)
    return m.x0, m.average, std_err, m.count


# ----------------------------------------------------------------------------
# Second-moment helpers
# ----------------------------------------------------------------------------
def moment2ellipticity(ixx, iyy, ixy):
    """e1 = (Ixx-Iyy)/T, e2 = 2 Ixy / T, T = Ixx + Iyy  (technote Eq. 2)."""
    T = ixx + iyy
    with np.errstate(divide='ignore', invalid='ignore'):
        e1 = (ixx - iyy) / T
        e2 = 2.0 * ixy / T
    return e1, e2, T


# ----------------------------------------------------------------------------
# DCR per visit
# ----------------------------------------------------------------------------
def compute_visit_dcr(visit_table, site_lat_deg=SITE_LAT_DEG):
    """
    Compute per-visit DCR1/DCR2 from the butler ``visit_table``.

    Parameters
    ----------
    visit_table : astropy.table.Table or pandas.DataFrame
        Must contain columns: 'visit', 'band', 'ra', 'dec', 'zenithDistance',
        'airmass', 'expMidptMJD'.
    site_lat_deg : float
        Observatory latitude in degrees.

    Returns
    -------
    dict
        {visit_id: {'dcr1': .., 'dcr2': .., 'band': .., 'tan_z2': .., 'q': ..}}
    """
    visit = np.asarray(visit_table['visit'])
    band = np.asarray(visit_table['band']).astype(str)
    ra = np.asarray(visit_table['ra'], dtype=float)          # deg
    dec = np.asarray(visit_table['dec'], dtype=float)        # deg
    zenith = np.asarray(visit_table['zenithDistance'], dtype=float)  # deg
    mjd = np.asarray(visit_table['expMidptMJD'], dtype=float)

    # tan^2(z)
    tan_z2 = np.tan(np.radians(zenith)) ** 2

    # Parallactic angle q from local sidereal time (astropy), technote method.
    lst = Time(mjd, format='mjd', scale='utc',
               location=CERRO_PACHON).sidereal_time('apparent').deg
    ha = np.radians(lst - ra)          # hour angle (rad)
    dec_r = np.radians(dec)
    lat_r = np.radians(site_lat_deg)
    q = np.arctan2(np.sin(ha),
                   np.tan(lat_r) * np.cos(dec_r) - np.sin(dec_r) * np.cos(ha))

    dcr1 = tan_z2 * np.cos(2.0 * q)
    dcr2 = tan_z2 * np.sin(2.0 * q)

    out = {}
    for i in range(len(visit)):
        out[int(visit[i])] = {
            'dcr1': float(dcr1[i]), 'dcr2': float(dcr2[i]),
            'band': band[i], 'tan_z2': float(tan_z2[i]), 'q': float(q[i]),
        }
    return out


# ----------------------------------------------------------------------------
# Data accumulation: getURI mapping + direct polars read of refit_psf_star
# (same pattern as SNR_vs_dT.py / SkyPlot_vs_secondMoment.py)
# ----------------------------------------------------------------------------
def accumulate(repo, collection, visitMappingFile, bands, frame, color_band1,
               color_band2, mag_column, star_set, n_visits):
    """
    Read the visit->parquet mapping (built by getData.py via butler.getURI on
    ``refit_psf_star``), load ``visit_table`` once for the per-visit DCR, then
    loop visits reading each parquet directly with polars.

    Returns
    -------
    dict of np.ndarray keyed by: band, color, mag, de1, de2, dT_T, dcr1, dcr2, tan_z2
    plus 'n_visits_per_band' dict.
    """
    from lsst.daf.butler import Butler

    # Per-visit parquet URIs (no butler in the loop below)
    with open(visitMappingFile, 'rb') as f:
        visit_mapping = pickle.load(f)

    # Single butler call: visit_table drives the per-visit DCR (z, q).
    butler = Butler(repo, collections=collection)
    visit_table = butler.get('visit_table', instrument='LSSTCam')
    visit_dcr = compute_visit_dcr(visit_table)

    src_cols, psf_cols = FRAME_COLUMNS[frame]

    acc = {k: [] for k in ('band', 'color', 'mag', 'de1', 'de2', 'dT_T', 'dcr1', 'dcr2', 'tan_z2')}
    n_visits_per_band = {}

    for b in bands:
        # Visits of this band, in visit order (from the mapping)
        band_visits = sorted(v for v, info in visit_mapping.items() if info['band'] == b)
        if n_visits is not None:
            band_visits = band_visits[:n_visits]
        n_visits_per_band[b] = len(band_visits)
        print(f"Band {b}: {len(band_visits)} visits", flush=True)

        mag_col = mag_column.format(band=b)
        cols = list(src_cols) + list(psf_cols) + [
            f'fgcm_mag_{color_band1}', f'fgcm_mag_{color_band2}', mag_col,
            'calib_psf_reserved', 'calib_psf_used',
        ]
        cols = list(dict.fromkeys(cols))  # de-dup (e.g. mag_col == fgcm_mag_c1)

        for v in tqdm(band_visits, desc=f"band {b}"):
            if v not in visit_dcr:
                print(f"  visit {v}: absent from visit_table, skipping", flush=True)
                continue
            parquet_path = visit_mapping[v]['parquet_path']
            try:
                tab = polars.scan_parquet(parquet_path).select(cols).collect()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to read columns {cols} from {parquet_path} "
                    f"(visit {v}). Does this refit_psf_star have the alt/az moment "
                    f"(shape_Ialtalt...) and fgcm_mag_* columns? Underlying error: {e}"
                )

            # Star selection
            if star_set == 'reserved':
                sel = tab['calib_psf_reserved'].to_numpy().astype(bool)
            elif star_set == 'used':
                sel = tab['calib_psf_used'].to_numpy().astype(bool)
            else:  # 'all'
                sel = np.ones(len(tab), dtype=bool)
            if not np.any(sel):
                continue

            ixx = tab[src_cols[0]].to_numpy()[sel]
            iyy = tab[src_cols[1]].to_numpy()[sel]
            ixy = tab[src_cols[2]].to_numpy()[sel]
            mixx = tab[psf_cols[0]].to_numpy()[sel]
            miyy = tab[psf_cols[1]].to_numpy()[sel]
            mixy = tab[psf_cols[2]].to_numpy()[sel]

            e1, e2, T = moment2ellipticity(ixx, iyy, ixy)
            me1, me2, mT = moment2ellipticity(mixx, miyy, mixy)
            de1 = e1 - me1
            de2 = e2 - me2
            with np.errstate(divide='ignore', invalid='ignore'):
                dT_T = (T - mT) / T

            mag1 = tab[f'fgcm_mag_{color_band1}'].to_numpy()[sel]
            mag2 = tab[f'fgcm_mag_{color_band2}'].to_numpy()[sel]
            color = mag1 - mag2
            mag = tab[mag_col].to_numpy()[sel]

            info = visit_dcr[v]
            n = int(np.sum(sel))
            acc['band'].append(np.full(n, b))
            acc['color'].append(color.astype(np.float32))
            acc['mag'].append(mag.astype(np.float32))
            acc['de1'].append(de1.astype(np.float32))
            acc['de2'].append(de2.astype(np.float32))
            acc['dT_T'].append(dT_T.astype(np.float32))
            acc['dcr1'].append(np.full(n, info['dcr1'], dtype=np.float32))
            acc['dcr2'].append(np.full(n, info['dcr2'], dtype=np.float32))
            acc['tan_z2'].append(np.full(n, info['tan_z2'], dtype=np.float32))

    out = {}
    for k, val in acc.items():
        out[k] = np.concatenate(val) if len(val) else np.array([])
    out['n_visits_per_band'] = n_visits_per_band
    print(f"Total stars accumulated: {len(out['band']):,}", flush=True)
    return out


# ----------------------------------------------------------------------------
# Color-bin masks (per band), following plots.py tertile scheme
# ----------------------------------------------------------------------------
def color_bin_edges(color, color_edges=None):
    """Return the 3 internal edges [q25, q50, q75] (or user-provided)."""
    if color_edges is not None:
        return np.asarray(color_edges, dtype=float)
    finite = color[np.isfinite(color)]
    return np.quantile(finite, [0.25, 0.5, 0.75])


def color_masks_and_labels(color, edges, colorname):
    """4 series: the 3 tertiles between q25..q75 and the 'all' (q25..q75) band."""
    q0, q1, q2 = edges[0], edges[1], edges[2]
    hi = np.nanmax(color[np.isfinite(color)]) if np.any(np.isfinite(color)) else q2
    finite = np.isfinite(color)
    mask1 = finite & (color > q0) & (color < q1)
    mask2 = finite & (color > q1) & (color < q2)
    mask3 = finite & (color > q2) & (color < hi)
    mask_all = finite & (color > q0) & (color < hi)
    labels = [
        f'{q0:.2f} < {colorname} < {q1:.2f}',
        f'{q1:.2f} < {colorname} < {q2:.2f}',
        f'{q2:.2f} < {colorname} < {hi:.2f}',
        f'{q0:.2f} < {colorname} < {hi:.2f}',
    ]
    return [mask1, mask2, mask3, mask_all], labels


# ----------------------------------------------------------------------------
# Plot builders  (return a dict of binned curves for pickling)
# ----------------------------------------------------------------------------
def _errorbar_series(ax, x0, mean, err, color, marker, label):
    ax.errorbar(x0, mean, yerr=err, ls='None', ms=7, alpha=0.5, capsize=1,
                color=color, marker=marker, label=label)


def plot_vs_dcr(data, band, edges, labels, masks, colorname, dcr_range, nbins,
                repOutPlot, tag):
    """Figs 5/6: de1 vs DCR1 (top), de2 vs DCR2 (bottom), 4 color series."""
    bmask = data['band'] == band
    curves = {'de1_vs_dcr1': {}, 'de2_vs_dcr2': {}}

    fig, ax = plt.subplots(nrows=2, figsize=(8, 8))
    nstars = int(np.sum(masks[3] & bmask))
    fig.suptitle(f'band: {band}\nno. of stars: {nstars}', fontsize=18, y=1.0)

    for (ykey, xkey, axi, panel) in [('de1', 'dcr1', 0, 'de1_vs_dcr1'),
                                     ('de2', 'dcr2', 1, 'de2_vs_dcr2')]:
        for mask, c, m, l in zip(masks, SERIES_COLORS, SERIES_MARKERS, labels):
            sel = mask & bmask
            x0, mean, err, count = bin_1d(data[xkey][sel], data[ykey][sel],
                                          dcr_range[0], dcr_range[1], nbins)
            _errorbar_series(ax[axi], x0, mean, err, c, m, l)
            curves[panel][l] = {'x0': x0, 'mean': mean, 'err': err, 'count': count}
        ax[axi].axhline(0, color='black', linewidth=2, linestyle='-', zorder=-1)
        ax[axi].grid(lw=0.5, alpha=0.5)
        ax[axi].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    ax[0].set_ylabel(r'$\delta e_1$')
    ax[0].set_xlabel(r'$\mathrm{DCR}_1$')
    ax[0].legend(fontsize=9)
    ax[1].set_ylabel(r'$\delta e_2$')
    ax[1].set_xlabel(r'$\mathrm{DCR}_2$')

    path = os.path.join(repOutPlot, f'dcr_residuals_{band}{tag}.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}", flush=True)
    return curves


def plot_vs_x(data, band, edges, labels, masks, colorname, xkey, x_range, nbins,
              xlabel, repOutPlot, tag, fname):
    """Figs 2/3 (xkey='mag') style: dT/T, de1, de2 vs x, 4 color series."""
    bmask = data['band'] == band
    curves = {'dT_T': {}, 'de1': {}, 'de2': {}}

    fig, ax = plt.subplots(ncols=3, figsize=(18, 5))
    nstars = int(np.sum(masks[3] & bmask))
    fig.suptitle(f'band: {band} | no. of stars: {nstars}', fontsize=16, y=1.02)

    for axi, ykey, ylabel in [(0, 'dT_T', r'$\delta T / T$'),
                              (1, 'de1', r'$\delta e_1$'),
                              (2, 'de2', r'$\delta e_2$')]:
        for mask, c, m, l in zip(masks, SERIES_COLORS, SERIES_MARKERS, labels):
            sel = mask & bmask
            x0, mean, err, count = bin_1d(data[xkey][sel], data[ykey][sel],
                                          x_range[0], x_range[1], nbins)
            _errorbar_series(ax[axi], x0, mean, err, c, m, l)
            curves[ykey][l] = {'x0': x0, 'mean': mean, 'err': err, 'count': count}
        ax[axi].axhline(0, color='black', linewidth=1.5, linestyle='--', zorder=-1)
        ax[axi].grid(lw=0.5, alpha=0.5)
        ax[axi].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax[axi].set_ylabel(ylabel)
        ax[axi].set_xlabel(xlabel)
    ax[0].legend(fontsize=9)

    path = os.path.join(repOutPlot, f'{fname}_{band}{tag}.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}", flush=True)
    return curves


def plot_vs_color(data, bands, colorname, color_range, nbins, repOutPlot, tag, bandtag):
    """Fig 4: dT/T, de1, de2 vs color (series = band) + color histogram."""
    band_color_map = {'u': 'b', 'g': 'g', 'r': 'r', 'i': 'k', 'z': 'm', 'y': 'c'}
    curves = {'dT_T': {}, 'de1': {}, 'de2': {}, 'hist': {}}

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, height_ratios=[2, 1])
    axes = [fig.add_subplot(gs[0, j]) for j in range(3)]
    axh = fig.add_subplot(gs[1, :])

    hist_bins = np.linspace(color_range[0], color_range[1], 60)
    for axi, ykey, ylabel in [(0, 'dT_T', r'$\delta T / T$'),
                              (1, 'de1', r'$\delta e_1$'),
                              (2, 'de2', r'$\delta e_2$')]:
        for b in bands:
            bmask = data['band'] == b
            c = band_color_map.get(b, None)
            x0, mean, err, count = bin_1d(data['color'][bmask], data[ykey][bmask],
                                          color_range[0], color_range[1], nbins)
            _errorbar_series(axes[axi], x0, mean, err, c, 'o', b)
            curves[ykey][b] = {'x0': x0, 'mean': mean, 'err': err, 'count': count}
        axes[axi].axhline(0, color='black', linewidth=1.5, linestyle='--', zorder=-1)
        axes[axi].grid(lw=0.5, alpha=0.5)
        axes[axi].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        axes[axi].set_ylabel(ylabel)
        axes[axi].set_xlabel(colorname)
    axes[0].legend(fontsize=9)

    for b in bands:
        bmask = data['band'] == b
        col = data['color'][bmask]
        col = col[np.isfinite(col)]
        h, _ = np.histogram(col, bins=hist_bins, density=True)
        axh.step(0.5 * (hist_bins[1:] + hist_bins[:-1]), h, where='mid',
                 color=band_color_map.get(b, None), label=b)
        curves['hist'][b] = {'edges': hist_bins, 'density': h}
    axh.set_xlabel(colorname)
    axh.set_ylabel('Number density')
    axh.legend(fontsize=9)
    axh.grid(lw=0.5, alpha=0.5)

    path = os.path.join(repOutPlot, f'color_residuals_{bandtag}{tag}.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}", flush=True)
    return curves


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def run(repo='dp2_prep', collection=None, visitMappingFile='data/visit_parquet_mapping.pkl',
        bands='ugrizy', frame='altaz',
        color_band1='g', color_band2='i', mag_column='fgcm_mag_{band}',
        star_set='reserved', n_visits=None, dcr_range=(-1.5, 1.5),
        mag_range=(16., 22.), color_range=(0.0, 3.5), nbins=7,
        do_dcr=True, do_mag=True, do_color=True,
        color_edges=None, repOutPlot='plots/', pklInput=None):

    os.makedirs(repOutPlot, exist_ok=True)
    colorname = f'{color_band1}-{color_band2}'
    tag = f'_{frame}'
    bandtag = ''.join(bands)

    if pklInput is not None:
        with open(pklInput, 'rb') as f:
            saved = pickle.load(f)
        _replot_from_pickle(saved, repOutPlot)
        return

    data = accumulate(repo, collection, visitMappingFile, bands, frame,
                      color_band1, color_band2, mag_column, star_set, n_visits)

    results = {
        'meta': {
            'repo': repo, 'collection': collection, 'visitMappingFile': visitMappingFile,
            'bands': bands, 'frame': frame,
            'colorname': colorname, 'star_set': star_set, 'n_visits': n_visits,
            'dcr_range': dcr_range, 'mag_range': mag_range, 'color_range': color_range,
            'nbins': nbins, 'n_visits_per_band': data['n_visits_per_band'],
        },
        'dcr': {}, 'mag': {}, 'color': {}, 'color_edges': {},
    }

    # Per-band color-bin masks (computed once, over the full accumulated color)
    per_band_masks = {}
    for b in bands:
        bmask = data['band'] == b
        edges = color_bin_edges(data['color'][bmask], color_edges)
        masks, labels = color_masks_and_labels(data['color'], edges, colorname)
        per_band_masks[b] = (edges, masks, labels)
        results['color_edges'][b] = edges

    if do_dcr:
        for b in bands:
            edges, masks, labels = per_band_masks[b]
            results['dcr'][b] = plot_vs_dcr(data, b, edges, labels, masks, colorname,
                                            dcr_range, nbins, repOutPlot, tag)
    if do_mag:
        for b in bands:
            edges, masks, labels = per_band_masks[b]
            results['mag'][b] = plot_vs_x(data, b, edges, labels, masks, colorname,
                                          'mag', mag_range, nbins,
                                          f'{mag_column.format(band=b)} (mag)',
                                          repOutPlot, tag, 'mag_residuals')
    if do_color:
        results['color'] = plot_vs_color(data, bands, colorname, color_range, nbins,
                                         repOutPlot, tag, bandtag)

    # Output pkl schema (for doing your own plotting, e.g. chromatic on vs off) --
    #   results['meta']          : dict of run params, incl. 'collection' (chromatic on/off)
    #   results['color_edges'][band]              : array of the 3 internal color-bin edges
    #   results['dcr'][band]['de1_vs_dcr1'][label]: {'x0','mean','err','count'}  (also de2_vs_dcr2)
    #   results['mag'][band][ykey][label]         : {'x0','mean','err','count'}  ykey in dT_T/de1/de2
    #   results['color'][ykey][band]              : {'x0','mean','err','count'}  ykey in dT_T/de1/de2
    #   results['color']['hist'][band]            : {'edges','density'}
    # 'label' is the color-bin string (e.g. "0.64 < g-i < 0.85"); the 4th is the "all" bin.
    pkl_path = os.path.join(repOutPlot, f'dcr_psf_residuals_{bandtag}{tag}.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved binned results: {pkl_path}", flush=True)


def _replot_from_pickle(saved, repOutPlot):
    """Redraw all figures from a saved results pickle (no butler access)."""
    meta = saved['meta']
    colorname = meta['colorname']
    tag = f"_{meta['frame']}"
    bandtag = ''.join(meta['bands'])

    # DCR figures
    for b, curves in saved.get('dcr', {}).items():
        fig, ax = plt.subplots(nrows=2, figsize=(8, 8))
        fig.suptitle(f'band: {b}', fontsize=18, y=1.0)
        for axi, panel, ylabel, xlabel in [(0, 'de1_vs_dcr1', r'$\delta e_1$', r'$\mathrm{DCR}_1$'),
                                           (1, 'de2_vs_dcr2', r'$\delta e_2$', r'$\mathrm{DCR}_2$')]:
            for (l, cvals), c, m in zip(curves[panel].items(), SERIES_COLORS, SERIES_MARKERS):
                _errorbar_series(ax[axi], cvals['x0'], cvals['mean'], cvals['err'], c, m, l)
            ax[axi].axhline(0, color='black', linewidth=2, linestyle='-', zorder=-1)
            ax[axi].grid(lw=0.5, alpha=0.5)
            ax[axi].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            ax[axi].set_ylabel(ylabel)
            ax[axi].set_xlabel(xlabel)
        ax[0].legend(fontsize=9)
        path = os.path.join(repOutPlot, f'dcr_residuals_{b}{tag}.png')
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {path}", flush=True)

    # Magnitude figures
    for b, curves in saved.get('mag', {}).items():
        fig, ax = plt.subplots(ncols=3, figsize=(18, 5))
        fig.suptitle(f'band: {b}', fontsize=16, y=1.02)
        for axi, ykey, ylabel in [(0, 'dT_T', r'$\delta T / T$'),
                                  (1, 'de1', r'$\delta e_1$'),
                                  (2, 'de2', r'$\delta e_2$')]:
            for (l, cvals), c, m in zip(curves[ykey].items(), SERIES_COLORS, SERIES_MARKERS):
                _errorbar_series(ax[axi], cvals['x0'], cvals['mean'], cvals['err'], c, m, l)
            ax[axi].axhline(0, color='black', linewidth=1.5, linestyle='--', zorder=-1)
            ax[axi].grid(lw=0.5, alpha=0.5)
            ax[axi].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            ax[axi].set_ylabel(ylabel)
            ax[axi].set_xlabel('magnitude')
        ax[0].legend(fontsize=9)
        path = os.path.join(repOutPlot, f'mag_residuals_{b}{tag}.png')
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {path}", flush=True)

    # Color figure
    ccurves = saved.get('color', {})
    if ccurves:
        band_color_map = {'u': 'b', 'g': 'g', 'r': 'r', 'i': 'k', 'z': 'm', 'y': 'c'}
        fig = plt.figure(figsize=(18, 10))
        gs = gridspec.GridSpec(2, 3, height_ratios=[2, 1])
        axes = [fig.add_subplot(gs[0, j]) for j in range(3)]
        axh = fig.add_subplot(gs[1, :])
        for axi, ykey, ylabel in [(0, 'dT_T', r'$\delta T / T$'),
                                  (1, 'de1', r'$\delta e_1$'),
                                  (2, 'de2', r'$\delta e_2$')]:
            for b, cvals in ccurves.get(ykey, {}).items():
                _errorbar_series(axes[axi], cvals['x0'], cvals['mean'], cvals['err'],
                                 band_color_map.get(b, None), 'o', b)
            axes[axi].axhline(0, color='black', linewidth=1.5, linestyle='--', zorder=-1)
            axes[axi].grid(lw=0.5, alpha=0.5)
            axes[axi].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            axes[axi].set_ylabel(ylabel)
            axes[axi].set_xlabel(colorname)
        axes[0].legend(fontsize=9)
        for b, hvals in ccurves.get('hist', {}).items():
            edges = hvals['edges']
            axh.step(0.5 * (edges[1:] + edges[:-1]), hvals['density'], where='mid',
                     color=band_color_map.get(b, None), label=b)
        axh.set_xlabel(colorname)
        axh.set_ylabel('Number density')
        axh.legend(fontsize=9)
        axh.grid(lw=0.5, alpha=0.5)
        path = os.path.join(repOutPlot, f'color_residuals_{bandtag}{tag}.png')
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {path}", flush=True)


def _parse_range(s):
    lo, hi = s.split(',')
    return (float(lo), float(hi))


def main():
    p = argparse.ArgumentParser(description="Reproduce SITCOMTN-174 PSF residual batch (Figs 2-6)")
    p.add_argument('--repo', type=str, default='dp2_prep',
                   help='Butler repo (used only for the single visit_table load)')
    p.add_argument('--collection', type=str, default='LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2',
                   help='Butler collection exposing visit_table (used only for the DCR inputs)')
    p.add_argument('--visitMappingFile', type=str, default='data/visit_parquet_mapping.pkl',
                   help='visit->parquet mapping (getURI on refit_psf_star) with alt/az + fgcm columns')
    p.add_argument('--bands', type=str, default='ugrizy', help='Bands to process, e.g. g or ugrizy')
    p.add_argument('--frame', type=str, default='altaz', choices=['altaz', 'radec'],
                   help='Coordinate frame for the moments')
    p.add_argument('--color_band1', type=str, default='g', help='First band of the color (mag1)')
    p.add_argument('--color_band2', type=str, default='i', help='Second band of the color (mag2)')
    p.add_argument('--mag_column', type=str, default='fgcm_mag_{band}',
                   help="Magnitude column for the vs-mag plot; '{band}' expands to the visit band")
    p.add_argument('--star_set', type=str, default='reserved', choices=['reserved', 'used', 'all'],
                   help='Which PSF stars to use')
    p.add_argument('--n_visits', type=int, default=None, help='Limit visits per band (testing)')
    p.add_argument('--dcr_range', type=_parse_range, default=(-1.5, 1.5), help='min,max for DCR axis')
    p.add_argument('--mag_range', type=_parse_range, default=(16., 22.), help='min,max for magnitude axis')
    p.add_argument('--color_range', type=_parse_range, default=(0.0, 3.5), help='min,max for color axis')
    p.add_argument('--nbins', type=int, default=7, help='Number of bins')
    p.add_argument('--color_edges', type=str, default=None,
                   help='Optional fixed internal color edges "q0,q1,q2" (else per-band quartiles)')
    p.add_argument('--no_dcr', dest='do_dcr', action='store_false')
    p.add_argument('--no_mag', dest='do_mag', action='store_false')
    p.add_argument('--no_color', dest='do_color', action='store_false')
    p.add_argument('--repOutPlot', type=str, default='plots/', help='Output directory')
    p.add_argument('--pklInput', type=str, default=None, help='Replot from a saved results pkl (no butler)')
    args = p.parse_args()

    color_edges = None
    if args.color_edges is not None:
        color_edges = [float(x) for x in args.color_edges.split(',')]

    run(repo=args.repo, collection=args.collection, visitMappingFile=args.visitMappingFile,
        bands=args.bands, frame=args.frame,
        color_band1=args.color_band1, color_band2=args.color_band2, mag_column=args.mag_column,
        star_set=args.star_set, n_visits=args.n_visits, dcr_range=args.dcr_range,
        mag_range=args.mag_range, color_range=args.color_range, nbins=args.nbins,
        do_dcr=args.do_dcr, do_mag=args.do_mag, do_color=args.do_color,
        color_edges=color_edges, repOutPlot=args.repOutPlot, pklInput=args.pklInput)


if __name__ == '__main__':
    main()
