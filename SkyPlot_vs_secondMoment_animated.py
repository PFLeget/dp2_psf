import numpy as np
import treegp
print(treegp.__version__)
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import hpgeom as hpg

import os
os.environ["POLARS_MAX_THREADS"] = "1"
import polars

import pickle
import argparse

# Import skyproj for sky visualization
from skyproj import McBrydeSkyproj
from skyproj.survey import _Survey


class SurveyMcBrydeSkyproj(_Survey, McBrydeSkyproj):
    """McBryde projection with survey footprint drawing capabilities."""
    pass


# Columns to read from parquet files
PARQUET_COLUMNS = [
    'slot_Shape_xx', 'slot_Shape_yy', 'slot_Shape_xy',
    'slot_PsfShape_xx', 'slot_PsfShape_xy', 'slot_PsfShape_yy',
    'coord_ra', 'coord_dec', 'slot_Centroid_x', 'slot_Centroid_y',
    'detector', 'psf_max_value', 'calib_psf_reserved',
]


def load_visit_data(parquet_path):
    """
    Load visit data from parquet file and compute derived columns.

    Parameters
    ----------
    parquet_path : str
        Path to the parquet file

    Returns
    -------
    dict
        Dictionary with all necessary columns including derived ones
    """
    # Read parquet file with polars (fast!)
    table = polars.scan_parquet(parquet_path).select(PARQUET_COLUMNS).collect()

    # Convert to numpy arrays
    slot_Shape_xx = table['slot_Shape_xx'].to_numpy()
    slot_Shape_yy = table['slot_Shape_yy'].to_numpy()
    slot_Shape_xy = table['slot_Shape_xy'].to_numpy()
    slot_PsfShape_xx = table['slot_PsfShape_xx'].to_numpy()
    slot_PsfShape_yy = table['slot_PsfShape_yy'].to_numpy()
    slot_PsfShape_xy = table['slot_PsfShape_xy'].to_numpy()

    # Compute derived quantities
    T_src = slot_Shape_xx + slot_Shape_yy
    e1_src = (slot_Shape_xx - slot_Shape_yy) / T_src
    e2_src = 2 * slot_Shape_xy / T_src

    T_psf = slot_PsfShape_xx + slot_PsfShape_yy
    e1_psf = (slot_PsfShape_xx - slot_PsfShape_yy) / T_psf
    e2_psf = 2 * slot_PsfShape_xy / T_psf

    return {
        'ixx_src': slot_Shape_xx,
        'iyy_src': slot_Shape_yy,
        'ixy_src': slot_Shape_xy,
        'ixx_psf': slot_PsfShape_xx,
        'iyy_psf': slot_PsfShape_yy,
        'ixy_psf': slot_PsfShape_xy,
        'dT_T': (T_src - T_psf) / T_src,
        'de1': e1_src - e1_psf,
        'de2': e2_src - e2_psf,
        'ra': table['coord_ra'].to_numpy(),
        'dec': table['coord_dec'].to_numpy(),
        'xCCD': table['slot_Centroid_x'].to_numpy(),
        'yCCD': table['slot_Centroid_y'].to_numpy(),
        'detector': table['detector'].to_numpy(),
        'psf_max_value': table['psf_max_value'].to_numpy(),
        'calib_psf_reserved': table['calib_psf_reserved'].to_numpy(),
    }


def create_animated_sky_plot(bands='g', visitMappingFile="data/visit_parquet_mapping.pkl",
                              repOutPlot='plots/',
                              key_second_moment='dT_T', bin_spacing=120, colorScale=0.005,
                              psf_max_value=0, fps=10, visits_per_frame=10):
    """
    Create an animated sky plot showing how PSF residuals average evolves over time.

    Parameters
    ----------
    bands : str
        Band(s) to process (e.g., 'g', 'ugrizy')
    visitMappingFile : str
        Path to the visit_parquet_mapping.pkl file
    repOutPlot : str
        Output directory for plots
    key_second_moment : str
        Second moment key to plot (e.g., 'dT_T', 'de1', 'de2')
    bin_spacing : float
        HEALPix bin spacing in arcsec
    colorScale : float
        Color scale range [-colorScale, +colorScale]
    psf_max_value : float
        Exclude PSFs with max pixel value below this threshold
    fps : int
        Frames per second in the output video
    visits_per_frame : int
        Number of visits to add per frame (to speed up video creation)
    """

    CMAP = plt.cm.inferno

    # Load the visit mapping
    with open(visitMappingFile, 'rb') as f:
        visit_mapping = pickle.load(f)

    # Filter visits by band(s) and sort by visit ID (chronological order)
    selected_visits = []
    for visit, info in visit_mapping.items():
        if info['band'] in bands:
            selected_visits.append((visit, info))

    # Sort by visit ID (chronological)
    selected_visits.sort(key=lambda x: x[0])

    print(f"Selected {len(selected_visits)} visits for bands: {bands}")
    print(f"Will create ~{len(selected_visits) // visits_per_frame} frames")

    # Initialize meanify_healpix
    meanifyHealpix = treegp.meanify_healpix(bin_spacing=bin_spacing)

    # Pre-load all visit data and add to meanify
    # We'll store intermediate states for animation
    print("Loading all visits and computing intermediate averages...")

    # Compute nside from bin_spacing to create the healpix map structure
    # We need to process at least one visit first to get nside
    first_visit, first_info = selected_visits[0]
    data = load_visit_data(first_info['parquet_path'])
    filtering = np.ones(len(data["ra"]), dtype=bool)
    if psf_max_value > 0:
        filtering &= (data["psf_max_value"] > psf_max_value)
    coord = np.array([np.degrees(data['ra']), np.degrees(data['dec'])]).T
    meanifyHealpix.add_field(coord[filtering], data[key_second_moment][filtering])

    # Store frames data: list of (n_visits, healpix_map, nside)
    frames_data = []

    # Process visits in batches
    n_visits = 1
    for i, (visit, info) in enumerate(tqdm(selected_visits[1:], desc="Processing visits")):
        # Load and add visit data
        data = load_visit_data(info['parquet_path'])

        filtering = np.ones(len(data["ra"]), dtype=bool)
        if psf_max_value > 0:
            filtering &= (data["psf_max_value"] > psf_max_value)

        coord = np.array([np.degrees(data['ra']), np.degrees(data['dec'])]).T
        meanifyHealpix.add_field(coord[filtering], data[key_second_moment][filtering])
        n_visits += 1

        # Save frame every visits_per_frame visits
        if n_visits % visits_per_frame == 0 or i == len(selected_visits) - 2:
            # Compute current average
            meanifyHealpix.meanify()

            nside = meanifyHealpix.nside
            npix = hpg.nside_to_npixel(nside)
            healpix_map = np.full(npix, hpg.UNSEEN)
            valid_pixels = meanifyHealpix._valid_pixels
            healpix_map[valid_pixels] = meanifyHealpix.params0.copy()

            frames_data.append({
                'n_visits': n_visits,
                'healpix_map': healpix_map.copy(),
                'nside': nside,
            })

    print(f"Created {len(frames_data)} frames")

    # Now create the animation
    print("Creating animation...")

    # Set up the figure
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111)

    # Create skyproj
    sp = SurveyMcBrydeSkyproj(ax=ax, lon_0=0.0)

    # Color scale
    MIN = -colorScale
    MAX = colorScale

    if key_second_moment == 'dT_T':
        ksm = '$\\delta T / T$'
    else:
        ksm = key_second_moment

    # Draw initial empty map
    nside = frames_data[0]['nside']
    npix = hpg.nside_to_npixel(nside)
    empty_map = np.full(npix, hpg.UNSEEN)

    im, lon_raster, lat_raster, values_raster = sp.draw_hpxmap(
        empty_map, nest=True, zoom=False, vmin=MIN, vmax=MAX, cmap=CMAP
    )

    # Draw overlays (these stay constant)
    sp.draw_milky_way(label='Milky Way')
    sp.draw_des(edgecolor='blue', lw=2, label='DES footprint')
    sp.draw_colorbar(label=ksm, fontsize=14, pad=0.02)
    sp.ax.legend(loc='lower right', fontsize=10)

    # Title will be updated
    title = sp.ax.set_title(f"DP2 {ksm} | bands: ({bands}) | N_visits: 0", fontsize=16, y=1.05)

    plt.subplots_adjust(left=0.05, right=0.98, top=0.98, bottom=0.08)

    def update(frame_idx):
        """Update function for animation."""
        frame = frames_data[frame_idx]

        # Clear and redraw the map
        # We need to update the pcolormesh data
        # Unfortunately skyproj doesn't easily support updating, so we redraw
        ax.clear()

        # Recreate skyproj for this frame
        sp_frame = SurveyMcBrydeSkyproj(ax=ax, lon_0=0.0)

        im, _, _, _ = sp_frame.draw_hpxmap(
            frame['healpix_map'], nest=True, zoom=False, vmin=MIN, vmax=MAX, cmap=CMAP
        )

        sp_frame.draw_milky_way(label='Milky Way')
        sp_frame.draw_des(edgecolor='blue', lw=2, label='DES footprint')
        sp_frame.draw_colorbar(label=ksm, fontsize=14, pad=0.02)
        sp_frame.ax.legend(loc='lower right', fontsize=10)
        sp_frame.ax.set_title(f"DP2 {ksm} | bands: ({bands}) | N_visits: {frame['n_visits']}",
                              fontsize=16, y=1.05)

        return []

    # Create animation
    anim = FuncAnimation(fig, update, frames=len(frames_data),
                         interval=1000//fps, blit=False)

    # Save as mp4
    output_file = os.path.join(repOutPlot,
                               f'{key_second_moment}_sky_{bands}_{int(bin_spacing)}_{int(psf_max_value)}_animated.mp4')

    print(f"Saving animation to {output_file}...")
    writer = FFMpegWriter(fps=fps, metadata=dict(artist='LSST/Rubin'), bitrate=1800)
    anim.save(output_file, writer=writer)

    print("Done!")
    plt.close()


def main():

    parser = argparse.ArgumentParser(description="Animated sky map of PSF second moment residuals")
    parser.add_argument('--bands', type=str, required=True,
                        help="The band(s) to process (e.g., y, g, r, i, z, u, ugrizy)")
    parser.add_argument('--visitMappingFile', type=str, required=True,
                        help="Path to visit_parquet_mapping.pkl file")

    parser.add_argument('--key_second_moment', type=str, default='dT_T',
                        help='second moment key')
    parser.add_argument('--bin_spacing', type=float, default=120,
                        help='HEALPix bin size in arcsec')
    parser.add_argument('--psf_max_value', type=float, default=0,
                        help='exclude PSFs with max pixel value below this (e-)')
    parser.add_argument('--colorScale', type=float, default=0.005,
                        help='Min/Max of color scale')
    parser.add_argument('--repOutPlot', type=str, default='plots/',
                        help='Output directory for plots')
    parser.add_argument('--fps', type=int, default=10,
                        help='Frames per second in output video')
    parser.add_argument('--visits_per_frame', type=int, default=10,
                        help='Number of visits to add per frame')

    args = parser.parse_args()

    create_animated_sky_plot(bands=args.bands, visitMappingFile=args.visitMappingFile,
                              repOutPlot=args.repOutPlot,
                              key_second_moment=args.key_second_moment,
                              bin_spacing=args.bin_spacing,
                              colorScale=args.colorScale,
                              psf_max_value=args.psf_max_value,
                              fps=args.fps,
                              visits_per_frame=args.visits_per_frame)


if __name__ == "__main__":
    main()
