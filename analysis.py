# %% Imports

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import io

import MDAnalysis as mda
from MDAnalysis.lib.distances import (
    self_distance_array,
    distance_array,
    capped_distance,
)

from collections import defaultdict
from tqdm.notebook import tqdm
from scipy.stats import bootstrap, pearsonr
from math import ceil

from matplotlib.patches import Patch
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from matplotlib.collections import PolyCollection
from matplotlib import cm

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

sns.set(style="ticks")
# %% Functions

# Periodic Image check. Just run to test if the simulation is valid.


def min_periodic_distance(prot_coords, box_dims):
    """
    Compute the minimum distance (Å) between prot_coords (N×3) and any
    periodic copy of itself, excluding the original.

    Parameters
    ----------
    prot_coords : (N,3) array of float
        Cartesian coordinates of all protein atoms in Å.
    box_dims : array-like of length 3
        (Lx, Ly, Lz) box dimensions in Å.

    Returns
    -------
    float
        The minimum separation (Å) between any atom of the protein and any
        periodic image of a different atom.
    """
    Lx, Ly, Lz = box_dims
    originals = prot_coords.copy()

    # Build all translations in X and Y directions (± one box shift).
    # We do not shift in Z (membrane normal).
    shifts = [
        [+Lx, 0.0, 0.0],
        [-Lx, 0.0, 0.0],
        [0.0, +Ly, 0.0],
        [0.0, -Ly, 0.0],
        [+Lx, +Ly, 0.0],
        [+Lx, -Ly, 0.0],
        [-Lx, +Ly, 0.0],
        [-Lx, -Ly, 0.0],
    ]

    min_dist = np.inf
    for shift in shifts:
        shifted = originals + shift
        d2 = distance_array(originals, shifted, box=None)  # no PBC; already shifted
        min_dist = min(min_dist, d2.min())

    return min_dist


def check_pi_dist(sim_path, plot=False):
    # Load trajectory & compute metrics

    u = mda.Universe(
        sim_path / "step7_production_protein.gro",
        sim_path / "step7_production_protein_nojump.xtc",
    )

    # Extract box lengths (X and Y) in Å for every frame
    box = np.array(
        [ts.dimensions[:2].copy() for ts in u.trajectory]
    )  # shape = (n_frames, 2)
    time_ps = np.array([ts.time for ts in u.trajectory])  # ps

    # Prepare arrays to store protein max‐length and min‐image distance
    maxlen_a = []
    min_dis_a = []

    prot = u.select_atoms("protein")

    for ts in u.trajectory:
        # 1. maximum end‐to‐end distance among all protein atoms (Å)
        dists = self_distance_array(
            prot.positions.copy()
        )  # upper‐triangle distances, Å
        maxlen_a.append(dists.max())

        # 2. minimum distance to any periodic image (Å)
        min_dist = min_periodic_distance(prot.positions.copy(), u.dimensions[:3].copy())
        min_dis_a.append(min_dist)

    maxlen_a = np.asarray(maxlen_a)
    min_dis_a = np.asarray(min_dis_a)

    print(f"Overall maximum protein length = {maxlen_a.max():.2f} Å")
    print(f"Overall minimum box lenght {box[:, 0].min()}")
    print(f"Overall minimum distance to periodic image = {min_dis_a.min():.2f} Å")

    if min_dis_a.min() <= 12:
        raise ValueError(
            f"ERROR: Simulation at destination {sim_path} got to 1.2 nm to its periodic image with a distance of {min_dis_a.min()} Angstrom. Repeat or get rid of this simulation"
        )
    else:
        print(f"Simulation at {sim_path} passed without issues")

    if plot:
        time_ns = time_ps / 1000.0  # convert ps to ns
        Lx = box[:, 0]  # box X length in Å

        plt.figure(figsize=(6, 4))
        plt.plot(time_ns, Lx, label="Box X (Å)")
        plt.plot(time_ns, maxlen_a, label="Max protein length (Å)")
        plt.xlabel("Time (ns)")
        plt.ylabel("Distance (Å)")
        plt.legend(loc="upper right")
        plt.title("Box X length and Protein Max‐Length vs. Time")
        plt.tight_layout()
        plt.show()

        # Plot 2: Minimum periodic‐image distance over time

        plt.figure(figsize=(6, 4))
        plt.plot(
            time_ns,
            min_dis_a,
            label="Min protein–image distance (Å)",
            color="tab:orange",
        )
        plt.axhline(20, ls="--", color="gray", label="20 Å threshold")
        plt.xlabel("Time (ns)")
        plt.ylabel("Distance (Å)")
        plt.ylim(0, max(min_dis_a) * 1.05)
        plt.legend(loc="upper right")
        plt.title("Min Protein–Periodic‐Image Distance vs. Time")
        plt.tight_layout()
        plt.show()
    return


# Numerical analysis of replicates


def analyze_replicate(sim_path, plot=True, update=False):

    outdir = sim_path / "analysis"
    outdir.mkdir(exist_ok=True, parents=True)

    if not update:
        required = [
            "min_dist_nm.dat",
            "total_contacts.dat",
            "residue_occupancy.csv",
            "tot_occupancy.csv",
            "bind_unbind_durations.csv",
            "bind_times.csv",
            "unbind_times.csv",
        ]

        missing = [f for f in required if not (outdir / f).exists()]

        if not missing:  # everything already there
            print(f"[SKIP] {sim_path}  - all analysis files present.")
            return  # exit the function early
        else:
            print(f"[RE-RUN] {sim_path}  - missing: {', '.join(missing)}")
            # fall through and run the full analysis

    u = mda.Universe(
        sim_path / "step7_production_nosolv.gro",
        sim_path / "step7_production_nosolv.xtc",
    )

    contact_cutoff = 4.0  # Å  (heavy-atom contact distance)
    stride = 1  # analyse every frame; increase to speed up
    # -------------------------------------------------------------------------

    bad_H = u.select_atoms("name H* and not type H")
    if len(bad_H) > 0:
        raise ValueError(
            f"Strange atom classification! Atom name: {bad_H[0].name} Atom type: {bad_H[0].type}"
        )

    # selections --------------------------------------------------------------
    prot_all = u.select_atoms("protein and not name H*")  # heavy atoms
    lipids_all = u.select_atoms("resname POPC DOPG and not name H*")  # heavy atoms

    # index mapping protein atom -> residue index (0..Nres-1)
    resids = prot_all.residues.resids
    resid2idx = {resid: i for i, resid in enumerate(resids)}
    nres = len(resids)

    # storage -----------------------------------------------------------------
    n_frames = len(u.trajectory[::stride])
    time_ns = np.empty(n_frames)
    min_d_nm = np.empty(n_frames)
    tot_cont = np.empty(n_frames, dtype=int)
    tot_cont_res = np.empty(n_frames, dtype=int)
    frames_in_contact = np.zeros(nres, dtype=int)

    cutoff_A = contact_cutoff  # Å (MDAnalysis uses Å)
    frame_no = 0

    # main loop ---------------------------------------------------------------
    bind_state = False
    bound_frames = 0
    bind_times = []
    unbind_times = []
    for ts in tqdm(u.trajectory[::stride], desc="Timeframes"):
        time_ns[frame_no] = ts.time / 1000.0  # ps to ns

        # 1. fast neighbour search within cutoff_A
        pairs = capped_distance(
            prot_all.positions,
            lipids_all.positions,
            max_cutoff=cutoff_A,
            box=ts.dimensions,
            return_distances=True,
        )

        if pairs[0].size == 0:
            if bind_state:
                bind_state = False
                unbind_times.append(ts.time)

            # no contacts within cutoff to need global minimum
            dmin_nm = (
                distance_array(
                    prot_all.positions, lipids_all.positions, box=ts.dimensions
                ).min()
                / 10.0
            )
            min_d_nm[frame_no] = dmin_nm
            tot_cont[frame_no] = 0
        else:
            if not bind_state:
                bind_state = True
                bind_times.append(ts.time)
            # contacts exist within cutoff
            pair_idx, dists = pairs  # unpack 2-tuple (idx_pairs, distances)
            min_d_nm[frame_no] = dists.min() / 10.0
            tot_cont[frame_no] = len(pair_idx)  # number of contacting prot atoms

            # mark residues that have ≥1 contact
            contact_resids = np.unique(
                [prot_all[atom_idx].resid for atom_idx in np.unique(pair_idx[:, 0])]
            )
            tot_cont_res[frame_no] = len(contact_resids)

            for resid in np.unique(prot_all.resids):
                if resid in contact_resids:
                    frames_in_contact[resid2idx[resid]] += 1
                else:
                    frames_in_contact[resid2idx[resid]] += 0
            bound_frames += 1

        frame_no += 1

    # save to text

    # Distance
    np.savetxt(
        outdir / "min_dist_nm.dat",
        np.column_stack((time_ns, min_d_nm)),
        header="time(ns)  min_dist(nm)",
    )

    # Contacts per frame
    np.savetxt(
        outdir / "total_contacts.dat",
        np.column_stack((time_ns, tot_cont, tot_cont_res)),
        header="time(ns)  n_contacts  n_res_contacs",
    )

    # occupancy
    print(
        f"Max value in frames_in_contact per residue: {np.max(frames_in_contact)}, Number of frames: {n_frames}"
    )
    occupancy = frames_in_contact / n_frames  # fraction for each residue
    print(
        f"Max value in frames_in_contact per residue: {np.max(frames_in_contact)}, Number of frames: {n_frames}"
    )
    tot_occupancy = bound_frames / n_frames  # fraction for total protein
    pd.DataFrame({"tot_occupancy": [tot_occupancy]}).to_csv(
        outdir / "tot_occupancy.csv", index=False
    )

    occ_df = pd.DataFrame(
        {
            "resid": resids,
            "resname": [res.resname for res in prot_all.residues],
            "occupancy": occupancy,  # fraction 0-1
        }
    )
    csv_path = outdir / "residue_occupancy.csv"
    occ_df.to_csv(csv_path, index=False)
    print(f"Per-residue occupancy table -> {csv_path}")

    # association/dissociation times
    if len(bind_times) == 0:
        asso_times = [np.nan]
        disso_times = []
        print("The protein did not make contact with the membrane")
    else:
        asso_times = [bind_times[0]]
        disso_times = []

        if len(bind_times) > len(unbind_times) and len(unbind_times) > 0:
            for i, time in enumerate(unbind_times):
                disso_times.append(time - bind_times[i])
                asso_times.append(bind_times[i + 1] - time)

        if len(bind_times) == len(unbind_times):
            for i, time in enumerate(bind_times):
                if i == 0:
                    disso_times.append(unbind_times[i] - time)
                else:
                    disso_times.append(unbind_times[i] - time)
                    asso_times.append(time - unbind_times[i - 1])

        if len(disso_times) > 0:
            print(
                f"Protein associated after {asso_times[0]/1000} ns and dissociated after {disso_times[0]/1000} ns"
            )
            print(
                f"Subsequently the protein associated {len(asso_times)-1} more times and dissociated {len(disso_times)-1} more times"
            )
        else:
            print(
                f"Protein associated after {asso_times[0]/1000} ns and remained bound"
            )
    print(f"The fractional occupancy is {tot_occupancy * 100}% of the whole simulation")

    # convert ps -> ns
    asso_ns = np.asarray(asso_times) / 1000.0
    disso_ns = np.asarray(disso_times) / 1000.0

    event_df = pd.DataFrame(
        {
            "event": ["association"] * len(asso_ns) + ["dissociation"] * len(disso_ns),
            "duration": np.concatenate([asso_ns, disso_ns]),  # ns
        }
    )
    event_df.to_csv(outdir / "bind_unbind_durations.csv", index=False)

    pd.DataFrame({"bind_time_ns": np.asarray(bind_times) / 1000.0}).to_csv(
        outdir / "bind_times.csv", index=False
    )
    pd.DataFrame({"unbind_time_ns": np.asarray(unbind_times) / 1000.0}).to_csv(
        outdir / "unbind_times.csv", index=False
    )

    if plot:

        # quick plots
        plt.figure(figsize=(5, 3))
        plt.plot(time_ns, min_d_nm)
        plt.ylabel("Min dist (nm)")
        plt.xlabel("Time (ns)")
        plt.tight_layout()
        # plt.savefig(outdir / "min_dist.png", dpi=300)
        plt.show()
        plt.close()

        plt.figure(figsize=(5, 3))
        plt.plot(time_ns, tot_cont, color="tab:orange")
        plt.ylabel("n_contacts")
        plt.xlabel("Time (ns)")
        plt.tight_layout()
        # plt.savefig(outdir / "contacts.png", dpi=300)
        plt.show()
        plt.close()

        print(f"Written results in {outdir}/")

        # Convert min distances back to Å for histogram
        min_d_A = min_d_nm * 10.0

        # 7. Histogram
        plt.figure(figsize=(4.5, 3.5))
        bins = np.linspace(0, 15, 60)  # 0 … 15 Å, 0.25 Å bin width
        plt.hist(min_d_A, bins=bins, density=True, alpha=0.7, color="tab:blue")

        plt.axvline(
            contact_cutoff,
            color="red",
            ls="--",
            label=f"chosen cutoff = {contact_cutoff:.1f} Å",
        )

        plt.xlabel("Min protein–lipid distance (Å)")
        plt.ylabel("Probability density")
        plt.title("Distribution of min distances\n(use valley to choose cutoff)")
        plt.legend(frameon=False)
        plt.tight_layout()
        hist_png = outdir / "min_dist_hist.png"
        # plt.savefig(hist_png, dpi=300)
        plt.show()
        plt.close()

        print(f"\nHistogram saved to {hist_png}")

        color_mapping = {
            0: ["ASP", "GLU"],
            1: ["ARG", "HIS", "LYS"],
            2: ["SER", "THR", "ASN", "GLN", "CYS", "GLY", "TYR"],
            3: [
                "ALA",
                "LEU",
                "ILE",
                "VAL",
                "MET",
                "PRO",
                "TRP",
                "PHE",
            ],
        }

        # map residue name -> category -> colour
        category_pal = {
            0: "tab:red",  # acidic  (ASP/GLU)
            1: "tab:blue",  # basic   (ARG/HIS/LYS)
            2: "tab:green",  # polar   (SER/…/TYR)
            3: "tab:orange",
        }  # hydrophobics

        # build fast lookup dictionaries
        cat_of_res = {}
        for cat, aalist in color_mapping.items():
            for aa in aalist:
                cat_of_res[aa] = cat

        bar_colors = [
            category_pal[cat_of_res.get(rn, 3)]  # default -> cat 3
            for rn in occ_df.resname
        ]

        # 2. bar-plot
        xlabels = [f"{rid} {rname}" for rid, rname in zip(occ_df.resid, occ_df.resname)]
        plt.figure(figsize=(max(6, len(occupancy) * 0.08), 3.5))
        plt.bar(
            range(len(occupancy)),
            occupancy,
            width=0.9,
            color=bar_colors,
            edgecolor="k",
            linewidth=0.2,
        )

        plt.xticks(range(len(occupancy)), xlabels, rotation=90, fontsize=6)
        plt.ylabel("Fraction of frames\nin contact (≤4 Å)")
        plt.xlabel("Residue  (id name)")
        plt.title("Per-residue contact occupancy")

        # legend — one entry per category

        nice_label = {
            0: "Negative",  # ASP, GLU
            1: "Positive",  # ARG, HIS, LYS
            2: "Polar",  # SER, THR, ASN, …
            3: "Apolar",
        }  # hydrophobics

        legend_handles = [
            Patch(
                facecolor=category_pal[c], label=nice_label[c]
            )  # use the pretty label
            for c in category_pal
        ]
        plt.legend(handles=legend_handles, frameon=False, fontsize=6)

        plt.tight_layout()
        bar_png = outdir / "residue_occupancy.png"
        # plt.savefig(bar_png, dpi=300)
        plt.show()
        plt.close()
        print(f"Occupancy bar-plot saved to {bar_png}")

    return


def print_stats(name, series):
    print(
        f"{name}:  min={series.min():.3f}  max={series.max():.3f}  "
        f"mean={series.mean():.3f}  ± SD={series.std(ddof=1):.3f}"  # Because on can never sample all populations its ddof=1
    )


# --------------------- Helper functions -------------------------


def read_xvg(path, names, ps2ns=False):
    """Return a DataFrame from a .xvg file (no header lines)."""
    with open(path) as fh:
        numeric = [ln for ln in fh if not ln.startswith(("#", "@"))]

    df = pd.read_csv(
        io.StringIO("".join(numeric)),
        delim_whitespace=True,
        names=names,
        engine="python",
    )  # tolerant of ragged spacing

    if ps2ns:
        df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors="coerce")  # force float
        df.iloc[:, 0] /= 1000.0
        df.rename(columns={df.columns[0]: "time_ns"}, inplace=True)

    return df


def bound_mask(series: pd.Series, bind_thr=5, win=250, unbind_thr=5, unbind_win=25):
    """
    Boolean mask of 'bound' frames using *two* hysteresis windows.

    enter  bound: contacts ≥ bind_thr for win consecutive frames
    leave  bound: contacts < unbind_thr for unbind_win consecutive frames

    Parameters
    ----------
    series      : pd.Series of n_contacts
    bind_thr    : int  (default 5)
    win         : int  frames required for binding   (default 250 ≃ 5 ns)
    unbind_thr  : int  (default 5)
    unbind_win  : int  frames required for unbinding (default 25 ≃ 0.5 ns)
    """
    n = len(series)
    mask = pd.Series(False, index=series.index)

    i = 0
    while i < n - win:
        # look for a binding window
        if (series.iloc[i : i + win] >= bind_thr).all():
            j = i + win
            # stay bound until an *unbinding* window appears
            while j < n:
                if j + unbind_win > n:  # nothing left to remain bound
                    j = n
                    break
                # contacts below unbind_thr for unbind_win frames?
                if (series.iloc[j : j + unbind_win] < unbind_thr).all():
                    break  # end of bound segment at j
                j += 1
            mask.iloc[i:j] = True
            i = j  # continue search after segment
        else:
            i += 1
    return mask


def read_xvg_proj(
    path: Path,
    contacts: pd.DataFrame,  # <‑‑ pass the *contacts* table
    n_pc: int = 2,
    skip_header: int = 21,
    stride: int = 1,
) -> pd.DataFrame:
    """
    Parse a multi‑block   gmx anaeig -proj   file and attach
    both the matching min‑distance and the bound/unbound flag
    (after the same replica‑wise striding).
    """
    #  1. PCs
    raw = path.read_text()
    blocks = re.split(r"(?m)^[&]|^@ *title", raw)[1:]
    if len(blocks) < n_pc:
        raise ValueError(f"{path.name}: only {len(blocks)} blocks, need ≥{n_pc}")

    pc_tables = []
    for i, blk in enumerate(blocks[:n_pc], start=1):
        lines = blk.splitlines()[skip_header:]
        numeric = "\n".join(l for l in lines if _NUMERIC_RE.match(l))
        df = pd.read_csv(
            io.StringIO(numeric),
            delim_whitespace=True,
            usecols=[1],
            names=[f"PC{i}"],
            header=None,
            engine="python",
        )
        pc_tables.append(df)

    pca_df = pd.concat(pc_tables, axis=1)  # full, un‑strided PCs

    #  2. sort contacts
    rep_order = [f"frame_{k:02d}" for k in range(6, 26)]  # 06 … 25
    sort_key = lambda col: col.map({r: i for i, r in enumerate(rep_order)})

    contacts_sorted = contacts.sort_values(
        ["replica_id", "time_ns"], key=sort_key
    ).reset_index(drop=True)

    # 3. stride inside each replica
    n_rep = len(rep_order)
    n_rows_total = len(pca_df)
    n_per_rep = n_rows_total // n_rep

    keep = []
    for r in range(n_rep):
        start = r * n_per_rep
        stop = start + n_per_rep
        keep.extend(range(start, stop, stride))

    pca_df = pca_df.iloc[keep].reset_index(drop=True)
    bound_series = contacts_sorted["bound"].iloc[keep].reset_index(drop=True)

    # sanity
    if len(pca_df) != len(bound_series):
        raise ValueError("Row mismatch after striding")

    pca_df["bound"] = bound_series
    return pca_df


def read_xvg_sasa(
    path: Path,
    contacts: pd.DataFrame,
    stride: int = 1,  # keep every frame by default
) -> pd.DataFrame:
    """
    Read a gmx sasa output that has one numeric block with three columns:
        time (ps)   SASA_total   SASA_TRP71
    Attach the bound/unbound flag and stride replica‑wise
    so it stays aligned with the concatenated trajectory.
    """
    #  1. numeric table
    numeric = "\n".join(
        l for l in path.read_text().splitlines() if _NUMERIC_RE.match(l)
    )
    sasa_df = pd.read_csv(
        io.StringIO(numeric),
        delim_whitespace=True,
        names=["time_ps", "SASA_total", "SASA_trp71"],
        header=None,
        engine="python",
    )

    #  2. align with contacts
    rep_order = [f"frame_{k:02d}" for k in range(6, 26)]
    contacts_sorted = (
        contacts.assign(
            replica_id=pd.Categorical(contacts["replica_id"], rep_order, ordered=True)
        )
        .sort_values(["replica_id", "time_ns"])
        .reset_index(drop=True)
    )

    # 3. stride inside each replica
    n_rep = len(rep_order)
    n_rows_total = len(sasa_df)  # 200 020
    n_per_rep = n_rows_total // n_rep  # 10 001

    keep = [
        idx
        for r in range(n_rep)
        for idx in range(r * n_per_rep, (r + 1) * n_per_rep, stride)
    ]

    sasa_df = sasa_df.iloc[keep].reset_index(drop=True)
    bound_col = contacts_sorted["bound"].iloc[keep].reset_index(drop=True)

    if len(sasa_df) != len(bound_col):
        raise ValueError("Row mismatch after striding")

    sasa_df["bound"] = bound_col
    return sasa_df


# --------------------- Plotting ----------------------------


def one_kde(ax, data, color, label, ls="-", smooth=1.0, norm=True):
    line = sns.kdeplot(  # pass with x= so kwargs are tidy-form
        x=data,
        ax=ax,
        linewidth=1.6,
        color=color,
        label=label,
        linestyle=ls,
        fill=False,
        bw_adjust=smooth,
        clip_on=False,
    )

    if norm:
        # --- rescale curve so its *area* = len(data)/N_TOTAL -------------
        curve = line.get_lines()[-1]
        y = curve.get_ydata()
        curve.set_ydata(y * len(data) / N_TOTAL)


def new_panel(width=5.2, height=3.6):
    return plt.subplots(
        figsize=(width, height),
        constrained_layout=True,  # keep the inner axes equal everywhere
    )


def kde_panel(x, y, xlab, ylab, title, outpng, ymin, ymax, xmax=20):
    """
    2-D kernel–density map + contour lines.
    `x`, `y` are 1-D NumPy / pandas arrays.
    """
    plt.figure(figsize=(5.5, 4))
    sns.kdeplot(
        x=x,
        y=y,
        fill=True,  # colour-filled
        thresh=0,  # show the full support
        levels=12,  # number of contour levels
        cmap="rocket_r",  # perceptually-uniform colormap
        linewidths=0.8,  # draw contour lines
    )
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.xlim(0, xmax)
    plt.ylim(ymin, ymax)
    plt.tight_layout()
    # plt.savefig(outpng, dpi=300)
    plt.show()
    plt.close()
    print("written", outpng)


def split_violin_boxes(
    data_dict,
    ylabel,
    fname,
    bw_adj=1,
    order=None,
    ylim=(None, None),
    box_width=0.28,
    dpi=300,
    legend=True,
    special_case=False,
):
    # -------- reshape input --------------------------------------------------
    rows = []
    for lab, (bnd, unb) in data_dict.items():
        rows.append(pd.DataFrame({"membrane": lab, "value": bnd, "state": "bound"}))
        rows.append(pd.DataFrame({"membrane": lab, "value": unb, "state": "unbound"}))
    df = pd.concat(rows, ignore_index=True)
    if order is None:
        order = list(data_dict.keys())

    # -------- plot -----------------------------------------------------------
    fig, ax = new_panel()

    # 1) split violins (no inner box)
    sns.violinplot(
        data=df,
        x="membrane",
        y="value",
        hue="state",
        hue_order=["unbound", "bound"],
        split=True,
        order=order,
        palette=STATE_PALETTE,
        bw_method="scott",
        bw_adjust=bw_adj,
        cut=0,
        inner=None,
        ax=ax,
        alpha=0.6,
    )

    # 2) overlay small black boxplots, one per half
    # sns.boxplot(
    #    data=df,
    #    x="membrane",
    #    y="value",
    #    hue="state",
    #    hue_order=["unbound", "bound"],
    #    order=order,
    #    dodge=True,
    #    width=box_width,
    #    showcaps=False,
    #    showfliers=False,
    #    whis=1.5,
    #    boxprops=dict(facecolor="black", edgecolor="black"),
    #    whiskerprops=dict(color="black", linewidth=1.2),
    #    medianprops=dict(color="white", linewidth=1.2),
    #    ax=ax,
    # )
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    # -------------------------------------------------------------
    # Build a minimalist custom legend
    handles = [
        Patch(
            facecolor=STATE_PALETTE["unbound"], edgecolor="black", label="unbound"
        ),  # left halves
        Patch(
            facecolor=STATE_PALETTE["bound"], edgecolor="black", label="bound"
        ),  # right halves
    ]

    ax.legend(
        handles=handles,
        frameon=False,  # ← removes the border box
        loc="upper right",  # or wherever you like
        title="",
        ncols=2,
    )  # no legend title

    # keep only our custom legend
    if not legend:
        ax.legend_.remove()
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(ylim)
    # plt.ylabel("")
    plt.xlabel("")
    # Optional: special-case horizontal line for 50% DOPG mean
    if special_case and "0% DOPG" in data_dict:
        unbound_vals = data_dict["0% DOPG"][1]  # ← just the unbound series
        mean_val = unbound_vals.mean()
        ax.axhline(mean_val, ls="--", color="black", lw=1.2)

    fig.tight_layout()
    fig.savefig(plot_dir / fname, dpi=dpi)
    plt.show()
    plt.close(fig)


# %% Running full numerical analysis:

# Recommended membrane by membrane to comply with wall-times <24 h
run_dir = Path("/.../replicates/0_POPC_100_DOPG")

replicates = run_dir.iterdir()
replicates = [r for r in run_dir.iterdir() if r.name != "analysis_master"]

min_dists = []
contacts = []
tot_occupancies = []
res_occupancies = []
init_asso_times = []


for rep in replicates[::-1]:
    if rep.is_dir():
        # Check if protein does not come within non-bonded cutoff distance
        required = [
            "min_dist_nm.dat",
            "total_contacts.dat",
            "residue_occupancy.csv",
            "tot_occupancy.csv",
            "bind_unbind_durations.csv",
            "bind_times.csv",
            "unbind_times.csv",
        ]

        missing = [f for f in required if not (rep / "analysis" / f).exists()]

        if not missing:  # everything already there
            print(f"[SKIP] {rep}  - all analysis files present.")
        else:
            print(f"[RE-RUN] {rep}  - missing: {', '.join(missing)}")
            # fall through and run full anaysis
            check_pi_dist(rep, plot=False)
        # run analysis
        # If update = True will recompute and overwrite existing results and plots
        # If update = False will skip runs with computed results
        analyze_replicate(rep, plot=True, update=False)

        # Load and collect data
        aid = rep.name
        a_dir = rep / "analysis"

        # 1. time-series min distance  &  contacts
        df_min = pd.read_csv(
            a_dir / "min_dist_nm.dat",
            comment="#",
            delim_whitespace=True,
            names=["time_ns", "min_dist_nm"],
        )
        df_cnt = pd.read_csv(
            a_dir / "total_contacts.dat",
            comment="#",
            delim_whitespace=True,
            names=["time_ns", "n_contacts", "n_res_contacts"],
        )

        df_min["replica_id"] = aid
        df_cnt["replica_id"] = aid
        min_dists.append(df_min)
        contacts.append(df_cnt)

        # 2. total fractional occupancy
        tot_occ = pd.read_csv(a_dir / "tot_occupancy.csv")  # column: tot_occupancy
        tot_occ["replica_id"] = aid
        tot_occupancies.append(tot_occ)

        # 3. per-residue occupancy
        res_occ = pd.read_csv(
            a_dir / "residue_occupancy.csv"
        )  # resid, resname, occupancy
        res_occ["replica_id"] = aid
        res_occupancies.append(res_occ)

        # 4. association/dissociaton times
        events = pd.read_csv(a_dir / "bind_unbind_durations.csv")  # event, duration
        first_assoc = events.loc[events["event"] == "association", "duration"].iloc[
            0
        ]  # numeric scalar (ns)

        init_asso_times.append({"replica_id": aid, "first_assoc_ns": first_assoc})

min_dist_df = pd.concat(min_dists, ignore_index=True)
tot_occupancy_df = pd.concat(tot_occupancies, ignore_index=True)
contacts_df = pd.concat(contacts, ignore_index=True)
res_occupancy_df = pd.concat(res_occupancies, ignore_index=True)
init_asso_times_df = pd.DataFrame(init_asso_times)

master_dir = run_dir / "analysis_master"
master_dir.mkdir(exist_ok=True, parents=True)
min_dist_df.to_csv(master_dir / "all_min_dist.csv", index=False)

max_residues = 132  # total residues in protein +1
mask_bad = contacts_df["n_res_contacts"] > max_residues
n_bad = mask_bad.sum()

# For some reason empty entries are filled with large random numbers from memory, if 0 contacts would be assigned
if n_bad:
    print(
        f"[clean-up] found {n_bad} frames with senseless n_res_contacts "
        f"(>{max_residues}); setting them to 0"
    )
    contacts_df.loc[mask_bad, "n_res_contacts"] = 0

contacts_df.to_csv(master_dir / "all_contacts.csv", index=False)
tot_occupancy_df.to_csv(master_dir / "all_tot_occupancy.csv", index=False)
res_occupancy_df.to_csv(master_dir / "all_residue_occupancy.csv", index=False)
init_asso_times_df.to_csv(master_dir / "all_init_asso_times.csv", index=False)

print(f"Master CSV files written to {master_dir}")

# %% Aggregated Plotting and Analysis

root = Path("/.../replicates")  # parent folder of membrane folders
mem_dirs = sorted(root.glob("*_POPC_*_DOPG"))  # 3 membrane composition folders

for mem in mem_dirs:
    master_dir = mem / "analysis_master"
    plot_dir = master_dir / "plots"
    plot_dir.mkdir(exist_ok=True, parents=True)

    # ------- 1.  load master tables -----------------

    min_df = pd.read_csv(master_dir / "all_min_dist.csv")
    cnt_df = pd.read_csv(master_dir / "all_contacts.csv")
    occ_df = pd.read_csv(master_dir / "all_tot_occupancy.csv")
    res_df = pd.read_csv(master_dir / "all_residue_occupancy.csv")
    asso_df = pd.read_csv(master_dir / "all_init_asso_times.csv")

    # -------- 2. simple stats tables -------------

    print_stats("Total occupancy", occ_df["tot_occupancy"])

    # --------- 3. bootstrap SD / CI for per-residue occupancy ---------

    # Build a 2-D array  shape = (n_replicas, n_residues)

    pivot = res_df.pivot_table(index="replica_id", columns="resid", values="occupancy")

    # rows = replica, columns = residue
    data_mat = pivot.to_numpy()  # NaN if a replica lacks that residue

    # bootstrap along the replica axis (axis=0)
    res = bootstrap(
        (data_mat,),  # the data, for some reasons needs to be a tuple
        np.nanmean,  # statistic that ignores nan
        axis=0,  # resample rows
        vectorized=True,
        n_resamples=1000,
        confidence_level=0.95,
        method="BCa",
    )  # bias-corrected accelerated

    boot_mean = res.bootstrap_distribution.mean(axis=-1)
    ci_low = res.confidence_interval.low
    ci_high = res.confidence_interval.high
    boot_sd = res.standard_error

    # assemble table for plotting and value storage
    res_unique = res_df[["resid", "resname"]].drop_duplicates().sort_values("resid")
    boot_table = res_unique.copy()
    boot_table["occ_mean"] = boot_mean
    boot_table["occ_sd"] = boot_sd
    boot_table["ci_low"] = ci_low
    boot_table["ci_high"] = ci_high
    boot_table.to_csv(master_dir / "residue_occupancy_bootstrap.csv", index=False)

    print("\nBootstrap summary saved to residue_occupancy_bootstrap.csv")
    print("Columns: resid, resname, occ_mean, occ_sd, ci_low, ci_high")

    #  Plot bootstrap-averaged per-residue occupancy   (mean ± SD)
    color_mapping = {
        0: ["ASP", "GLU"],
        1: ["ARG", "HIS", "LYS"],
        2: ["SER", "THR", "ASN", "GLN", "CYS", "GLY", "TYR"],
        3: [
            "ALA",
            "LEU",
            "ILE",
            "VAL",
            "MET",
            "PRO",
            "TRP",
            "PHE",
        ],
    }

    # map residue name to category to colour
    category_pal = {
        0: "tab:red",  # acidic  (ASP/GLU)
        1: "tab:blue",  # basic   (ARG/HIS/LYS)
        2: "tab:green",  # polar   (SER/…/TYR)
        3: "tab:orange",
    }  # hydrophobics

    cat_of_res = {}
    for cat, aalist in color_mapping.items():
        for aa in aalist:
            cat_of_res[aa] = cat

    boot_df = pd.read_csv(master_dir / "residue_occupancy_bootstrap.csv")
    xlabels = [f"{rid} {rname}" for rid, rname in zip(boot_df.resid, boot_df.resname)]
    y_mean = boot_df.occ_mean

    Nrep = len(res_df["replica_id"].unique())  # number of independent replicas
    y_err = boot_df.occ_sd  # lower error  # upper error

    # colours re-used from amino-acid classes
    bar_colors = [category_pal[cat_of_res.get(rn, 3)] for rn in boot_df.resname]

    plt.figure(figsize=(max(6, len(boot_df) * 0.08), 3.8))
    plt.bar(
        range(len(y_mean)),
        y_mean,
        yerr=y_err,
        capsize=2,
        linewidth=0.2,
        color=bar_colors,
        edgecolor="k",
    )

    plt.xticks(range(len(y_mean)), xlabels, rotation=90, fontsize=6)
    plt.ylabel("Mean occupancy")
    plt.xlabel("Residue")
    plt.title("0% DOPG - Fractional occupancy")
    plt.ylim(0, 0.3)

    nice_label = {
        0: "Negative",  # ASP, GLU
        1: "Positive",  # ARG, HIS, LYS
        2: "Polar",  # SER, THR, ASN, …
        3: "Apolar",
    }  # hydrophobics

    legend_handles = [
        Patch(facecolor=category_pal[c], label=nice_label[c])  # use the pretty label
        for c in category_pal
    ]
    plt.legend(handles=legend_handles, frameon=False, fontsize=6, loc="upper left")

    plt.tight_layout()
    # plt.savefig(plot_dir / "residue_occupancy_bootstrap.png", dpi=300)
    plt.show()
    plt.close()

    # ----- 4. total occupancy -----

    # ---------- total occupancy ---------------------------------------
    occ_vals = occ_df["tot_occupancy"].values
    plt.figure(figsize=(4, 3))
    plt.hist(occ_vals, bins=np.linspace(0, 1, 10), alpha=0.8, edgecolor="k")
    plt.ylim(0, 16)
    plt.yticks([0, 2, 4, 6, 8, 10, 12, 14, 16])
    plt.xlabel("Total fractional occupancy")
    plt.ylabel("Count (replicas)")
    plt.title("Distribution of overall occupancy")
    plt.tight_layout()
    # plt.savefig(plot_dir / "hist_tot_occupancy.png", dpi=300)
    plt.show()
    plt.close()

    print(
        f"{mem.name} Total occupancy  mean = {occ_vals.mean():.3f}  ± SD = {occ_vals.std(ddof=1):.3f}"  # Because on can never sample all populations its ddof=1
    )
# %% Strucural analysis (rmsd and dssp)
root = Path("/.../replicates")  # parent folder of membrane folders
mem_dirs = sorted(root.glob("*_POPC_*_DOPG"))  # 3 membrane composition folders

for mem in mem_dirs:
    print(f"Processing {mem.name} …")
    rmsd_frames, ss_frames = [], []

    for frame in mem.glob("frame_*"):
        rmsd_xvg = frame / "rmsd_7o3y.xvg"
        ss_xvg = frame / "ss_counts.xvg"
        if not (rmsd_xvg.exists() and ss_xvg.exists()):
            print(f"  [skip] {frame.name}: missing xvg")
            continue

        # --- RMSD of protein ----------------------------------------------------------
        rmsd = read_xvg(rmsd_xvg, ["time_ns", "RMSD_nm"])  # rmsd computed with gmx rms
        rmsd["replica_id"] = frame.name
        rmsd_frames.append(rmsd)

        # --- secondary structure counts -----------------------------------
        cols = [
            "time_ps",
            "Loop",
            "Break",
            "Bend",
            "Turn",
            "PP",
            "Pi",
            "310",
            "Beta",
            "Bridge",
            "Alpha",
        ]
        ss = read_xvg(ss_xvg, cols, ps2ns=True)  # dssp counts computed with gmx dssp
        ss["replica_id"] = frame.name
        ss_frames.append(ss)

    # nothing found?
    if not rmsd_frames and not ss_frames:
        print(f"  No usable replicas in {mem.name}")
        continue

    master_dir = mem / "analysis_master"
    # master_dir.mkdir(exist_ok=True, parents=True)

    if rmsd_frames:
        pd.concat(rmsd_frames, ignore_index=True).to_csv(
            master_dir / "all_rmsd.csv", index=False
        )
        print(f"RMSD -> {master_dir/'all_rmsd.csv'}")

    if ss_frames:
        pd.concat(ss_frames, ignore_index=True).to_csv(
            master_dir / "all_ss_counts.csv", index=False
        )
        print(f"SS counts -> {master_dir/'all_ss_counts.csv'}")

# Plotting
for mem_dir in sorted(root.glob("*_POPC_*_DOPG")):
    mname = mem_dir.name
    master = mem_dir / "analysis_master"
    rmsd_csv = master / "all_rmsd.csv"
    ss_csv = master / "all_ss_counts.csv"
    if not (rmsd_csv.exists() and ss_csv.exists()):
        print(f"[skip] {mname}: missing master CSVs")
        continue

    # ------------------------------------------------- RMSD overlay
    rmsd_df = pd.read_csv(rmsd_csv)
    plt.figure(figsize=(6, 3.2))
    for rid, g in rmsd_df.groupby("replica_id"):
        plt.plot(g["time_ns"], g["RMSD_nm"], lw=0.8, alpha=0.8)
    plt.xlabel("Time (ns)")
    plt.ylabel("RMSD (nm)")
    plt.title(f"{mname}  –  backbone RMSD (all replicas)")
    plt.tight_layout()
    # plt.savefig(master / "overlay_rmsd.png", dpi=250)
    plt.show()
    plt.close()

    # ------------------------------------------------- SS overlay
    ss_df = pd.read_csv(ss_csv)
    ss_df["Total_SS"] = ss_df.drop(columns=["time_ns", "replica_id"]).sum(axis=1)

    plt.figure(figsize=(6, 3.2))
    for rid, g in ss_df.groupby("replica_id"):
        plt.plot(g["time_ns"], g["Alpha"], lw=0.8, alpha=0.8)
    plt.xlabel("Time (ns)")
    plt.ylabel("# structured residues")
    plt.title(f"{mname}  –  secondary-structure content (all replicas)")
    plt.tight_layout()
    # plt.savefig(master / "overlay_sscount.png", dpi=250)
    plt.show()
    plt.close()

    print(f"[done] {mname}: overlay_rmsd.png & overlay_sscount.png")


# will append all new plots to the same analysis_master folder
for mem_dir in sorted(root.glob("*_POPC_*_DOPG")):
    mname = mem_dir.name
    master_dir = mem_dir / "analysis_master"
    plot_dir = master_dir / "plots"
    # plot_dir.mkdir(exist_ok=True, parents=True)

    paths = {
        "rmsd": master_dir / "all_rmsd.csv",
        "ss": master_dir / "all_ss_counts.csv",
        "dist": master_dir / "all_min_dist.csv",
    }
    if not all(p.exists() for p in paths.values()):
        print(f"[skip] {mname}: some master CSVs missing to {paths}")
        continue

    # ------------- merge ------------
    rmsd = pd.read_csv(paths["rmsd"])
    ss = pd.read_csv(paths["ss"])
    dist = pd.read_csv(paths["dist"])

    # keep only α-helix count from ss table (column ‘Alpha’)
    ss_small = ss[["time_ns", "replica_id", "Alpha"]]

    merged = (
        rmsd.merge(dist, on=["time_ns", "replica_id"])
        .merge(ss_small, on=["time_ns", "replica_id"])
        .rename(columns={"min_dist_nm": "d_nm", "Alpha": "n_helix"})
    )  # columns: time_ns, RMSD_nm, d_nm, n_helix, replica_id

    # -------- RMSD vs distance ----------
    kde_panel(
        x=merged["d_nm"] * 10,
        y=merged["RMSD_nm"],
        xlab="Min protein–membrane distance (Å)",
        ylab="Backbone RMSD (nm)",
        title=f"{mname} – RMSD vs distance",
        outpng=plot_dir / "rms_vs_dist_kde.png",
        ymin=0,
        ymax=0.8,
    )

    # -------- Helix vs distance -----------------
    kde_panel(
        x=merged["d_nm"] * 10,
        y=merged["n_helix"],
        xlab="Min protein–membrane distance (Å)",
        ylab="# residues in α-helix",
        title=f"{mname} – Helix content vs distance",
        outpng=plot_dir / "helix_vs_dist_kde.png",
        ymin=105,
        ymax=130,
    )

    # -------------------- DSSP visualisation --------------------

    # DSSP one-letter codes to colours
    dssp_palette = {
        "H": "red",  # α-helix
        "G": "gold",  # 3_10
        "I": "darkred",  # π-helix
        "E": "royalblue",  # β-strand
        "B": "lightskyblue",  # β-bridge
        "T": "limegreen",  # turn
        "S": "palegreen",  # bend
        "P": "plum",  # poly-pro
        "~": "grey",  # loop
        "=": "white",  # break
    }

    # Ensure consistent order
    dssp_codes_ordered = list(dssp_palette.keys())  # e.g., ['H', 'E', 'C', ...]

    code2i = {code: i for i, code in enumerate(dssp_codes_ordered)}
    i2code = {i: code for code, i in code2i.items()}
    lut = ListedColormap([dssp_palette[code] for code in dssp_codes_ordered])

    # ---------- read all dssp.dat files -------------------------
    dssp_strings = {}  # frame to [str,str,…]  (one str per time-step)
    all_rows = []  # for the logo

    for frame in sorted(mem_dir.glob("frame_*")):
        dat = frame / "dssp.dat"
        if not dat.exists():
            continue
        with open(dat) as fh:
            rows = [ln.strip() for ln in fh if ln and ln[0] not in "#@"]
        if rows:
            dssp_strings[frame.name] = rows
            all_rows.extend(rows)

    if not dssp_strings:
        print(f"[skip] {mname}: no dssp.dat files")
    else:
        # ---------- GRID of heat-maps (max 20 replicas) ----------
        ncols, nrep = 5, len(dssp_strings)
        nrows = ceil(nrep / ncols)

        fig, axarr = plt.subplots(
            nrows, ncols, figsize=(ncols * 4, nrows * 3), squeeze=False
        )
        i = 1
        for idx, (rid, rows) in enumerate(dssp_strings.items()):
            r, c = divmod(idx, ncols)

            # build the colour image -------------------------------------------
            img = np.vectorize(code2i.get)([list(s) for s in rows])
            ax = axarr[r, c]
            ax.imshow(
                img.T,
                aspect="auto",
                cmap=lut,
                origin="lower",
                interpolation="none",
                vmin=0,
                vmax=len(code2i) - 1,
            )
            ax.set_title(f"rep. {i}", fontsize=14)
            i += 1

            # ------------------------------------------------------------------
            # Y-labels (show only on the first column)
            if c == 0:  # panels in left-most column
                n_res = img.shape[0]  # 131 residues
                ax.set_yticks([0, 131])
                ax.set_yticklabels(["26", "156"], fontsize=14)
            else:
                ax.set_yticks([])

            # X-labels (show only on the bottom row)
            if r == nrows - 1:  # panels in bottom row
                n_frames = img.shape[1]  # number of time points in this replica
                ax.set_xticks([0, 10000])
                ax.set_xticklabels(["0", "2"], fontsize=14)
            else:
                ax.set_xticks([])

        # blank unused panels
        for k in range(nrep, nrows * ncols):
            r, c = divmod(k, ncols)
            axarr[r, c].axis("off")

        mname_dict = {
            "0_POPC_100_DOPG": "100% DOPG",
            "50_POPC_50_DOPG": "50% DOPG",
            "100_POPC_0_DOPG": "0% DOPG",
        }
        fig.suptitle(f"{mname_dict[str(mname)]} – DSSP heat-maps", y=1.03, fontsize=14)
        # ---------- add legend (outside, centre-right) ----------

        handles = [
            Patch(facecolor=col, edgecolor="k", label=code)
            for code, col in dssp_palette.items()
        ]
        # Place on the right outside the grid
        # fig.legend(
        #    handles=handles,
        #    loc="center left",
        #    bbox_to_anchor=(1.02, 0.5),  # (x, y) in figure coords
        #    borderaxespad=0.0,
        #    title="DSSP code",
        #    fontsize=7,
        # )
        # plt.xlabel("time (ns)", fontsize=14)
        # plt.ylabel("Residue", fontsize=14)

        fig.tight_layout()
        # plot_dir.mkdir(exist_ok=True, parents=True)

        fig.savefig(plot_dir / "dssp_heatmaps_grid.png", dpi=600, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        # ---------- LOGO (stacked fractions) --------------------
        arr = np.array([list(r) for r in all_rows])
        n_res = arr.shape[1]
        fractions = {c: (arr == c).mean(0) for c in dssp_palette}

        bottom = np.zeros(n_res)
        plt.figure(figsize=(8, 3.2))
        for c, color in dssp_palette.items():
            h = fractions[c]
            plt.bar(range(n_res), h, bottom=bottom, color=color, width=1.0, lw=0)
            bottom += h

        tick_pos = np.arange(n_res)  # 0 … 130  (x–positions of bars)
        tick_labels = tick_pos + 26  # show 26 … 156 instead
        plt.xticks(
            tick_pos[::10],  # every 10th residue to avoid clutter
            tick_labels[::10],
            rotation=0,
            fontsize=7,
        )
        plt.yticks(fontsize=7)
        plt.ylim(0, 1)
        plt.xlim(0, n_res)
        plt.xlabel("Residue")
        plt.ylabel("Fraction")
        plt.title(f"{mname_dict[str(mname)]} – per-residue DSSP distribution")
        handles = [
            Patch(facecolor=col, edgecolor="k", label=code)
            for code, col in dssp_palette.items()
        ]
        plt.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            title="DSSP code",
            fontsize=7,
            frameon=False,
        )
        plt.tight_layout()
        # plt.savefig(plot_dir / "dssp_logo.png", dpi=300)
        plt.show()
        plt.close()

        print(f"[extra] {mname}: DSSP grid & logo saved in {plot_dir}")

# %% Overlays of binding


root = Path("/.../replicates")
plot_dir = root / "global_analysis" / "plots"
plot_dir.mkdir(parents=True, exist_ok=True)

cdf_data = {}  # mem_name to (x_full, y_full)
hist_data = {}  # mem_name to 1‑D ndarray of distances (Å)
cnt_hist_data = {}


mem_dirs = [
    root / "0_POPC_100_DOPG",
    root / "50_POPC_50_DOPG",
    root / "100_POPC_0_DOPG",
]
# ---------------------------------------------------------------------
for mem_dir in mem_dirs:
    mname = mem_dir.name
    label_map = {
        "0_POPC_100_DOPG": "100% DOPG",
        "50_POPC_50_DOPG": "50% DOPG",
        "100_POPC_0_DOPG": "0% DOPG",
    }
    legend_label = label_map[mem_dir.name]
    master_dir = mem_dir / "analysis_master"

    # ---------- load per‑membrane master tables ----------------------
    min_df = pd.read_csv(master_dir / "all_min_dist.csv")
    cnt_df = pd.read_csv(master_dir / "all_contacts.csv")

    # ---------- store histogram input --------------------------------
    hist_data[legend_label] = (min_df["min_dist_nm"] * 10).values  # Å
    cnt_hist_data[legend_label] = cnt_df["n_contacts"]
    # ---------- first‑passage CDF ------------------------------------
    window = 250  # 250 frames  to 5 ns (0.02 ns/frame)
    contact_thr = 4

    first_passage = []
    for rid, grp in cnt_df.groupby("replica_id"):
        hits = grp["n_contacts"] > contact_thr
        sustained = hits.rolling(window, min_periods=window).sum() == window
        if sustained.any():
            first_idx = sustained.idxmax()
            first_time = grp.loc[first_idx, "time_ns"]
            first_passage.append(first_time)

    fp = np.sort(first_passage)  # ascending times (ns)
    x = np.concatenate(([0.0], fp, [200.0]))  # pad for pretty step
    y = np.concatenate(([0], np.arange(1, len(fp) + 1), [len(fp)]))
    cdf_data[legend_label] = (x, y)  # keep for later


colours = {
    "100% DOPG": "#d62728",  # red
    "50% DOPG": "#1f77b4",  # blue
    "0% DOPG": "#2ca02c",  # green
}

#  ---- overlay cdf-plot -----
plt.figure(figsize=(5.2, 3.6), constrained_layout=True)
for mname, (x, y) in cdf_data.items():
    plt.step(x, y, where="post", label=mname, color=colours[mname])
plt.xlabel("First sustained‑contact time (ns)")
plt.ylabel("Cumulative counts")
# plt.title("First‑passage CDF")
plt.xlim(0, 200)
plt.ylim(0, 20)
plt.yticks([0, 5, 10, 15, 20])
plt.legend(frameon=False, loc="upper left")
plt.tight_layout()
# plt.savefig(plot_dir / "first_passage_cdf_OVERLAY.png", dpi=300)
plt.show()
plt.close()


#  --------------- overlay min-dist histogram ----------------

plt.figure(figsize=(5.2, 3.6), constrained_layout=True)
bins = np.linspace(0, 80, 60)
for mname, d in hist_data.items():
    plt.hist(d, bins=bins, density=True, alpha=0.5, label=mname, color=colours[mname])
plt.xlabel("Protein–membrane distance (Å)")
plt.ylabel("Probability density")
# plt.title("Distribution of min distances – all membranes")
plt.legend(frameon=False)
plt.ylim(0, 0.25)
plt.tight_layout()
# plt.savefig(plot_dir / "min_dist_hist_OVERLAY.png", dpi=300)
plt.show()
plt.close()


plt.figure(figsize=(5.2, 3.6), constrained_layout=True)

for mname, d in hist_data.items():
    sns.kdeplot(
        d,
        bw_adjust=0.7,  # ‑‑> a bit less smoothing than default
        label=mname,
        linewidth=2,
        fill=None,
        color=colours[mname],
    )

plt.xlabel("Protein–membrane distance (Å)")
plt.ylabel("Kernel density estimate")
# plt.title("Smoothed distribution of min distances – all membranes")
plt.xlim(-5, 80)
plt.ylim(0, 0.14)
plt.legend(frameon=False)
plt.tight_layout()
# plt.savefig(plot_dir / "min_dist_KDE_OVERLAY.png", dpi=300)
plt.show()
plt.close()

#  ---------------------------- Contacts   ---------------------


plt.figure(figsize=(5.2, 3.6), constrained_layout=True)
bins = np.linspace(0, 200, 60)
for mname, d in cnt_hist_data.items():
    plt.hist(d, bins=bins, density=True, alpha=0.5, label=mname, color=colours[mname])
plt.xlabel("Protein-membrane heavy atom contacts")
plt.ylabel("Probability density")
# plt.title("Distribution of min distances – all membranes")
plt.legend(frameon=False)
# plt.ylim(0, 1)
# plt.yscale("log")
plt.tight_layout()
# plt.savefig(plot_dir / "contacts_hist_OVERLAY.png", dpi=300)
plt.show()
plt.close()

# ------------------ Violin Plot of contacts-----------------------

# Create long-form dataframe for seaborn
violin_df = pd.DataFrame(
    [(label, val) for label, data in cnt_hist_data.items() for val in data],
    columns=["membrane", "contacts"],
)

# List of membranes in the desired order
membrane_order = ["0% DOPG", "50% DOPG", "100% DOPG"]

plt.figure(figsize=(5.2, 3.6), constrained_layout=True)

sns.violinplot(
    data=violin_df,
    x="membrane",
    y="contacts",
    palette=colours,
    linewidth=1,
    cut=0,
    order=membrane_order,
    inner=None,  # Order the membranes from 0% DOPG to 100% DOPG
)

# Custom legend
handles = [mpatches.Patch(color=col, label=label) for label, col in colours.items()]
# plt.legend(handles=handles, frameon=False, loc="upper left")

plt.ylabel("Protein–membrane contacts")
plt.xlabel("")
plt.tight_layout()
# plt.savefig(plot_dir / "contacts_violin_OVERLAY.png", dpi=300)
plt.show()
plt.close()


# %% Overlay of secondary

root = Path("/.../replicates")
plot_dir = root / "global_analysis" / "plots"
plot_dir.mkdir(parents=True, exist_ok=True)


_NUMERIC_RE = re.compile(r"^[ \t]*[+-]?\d")

# ---- main loop ------------------------------------------------------------
mem_results = {}  # store the six DF per membrane if you wish

mem_dirs = [
    root / "0_POPC_100_DOPG",
    root / "50_POPC_50_DOPG",
    root / "100_POPC_0_DOPG",
]
# ---------------------------------------------------------------------
for mem_dir in mem_dirs:
    mname = mem_dir.name
    label_map = {
        "0_POPC_100_DOPG": "100% DOPG",
        "50_POPC_50_DOPG": "50% DOPG",
        "100_POPC_0_DOPG": "0% DOPG",
    }
    label_name = label_map[mname]
    master = mem_dir / "analysis_master"
    ss_df = pd.read_csv(master / "all_ss_counts.csv")
    contacts = pd.read_csv(master / "all_contacts.csv")

    for df in (ss_df, contacts):
        df["time_ns"] = df["time_ns"].round(2)
    # add a column that says whether each frame is bound or not
    contacts["bound"] = False
    for rid, grp in contacts.groupby("replica_id"):
        contacts.loc[grp.index, "bound"] = bound_mask(grp["n_contacts"])

    #   split secondary‑structure counts -----------------------------------
    ss_join = ss_df.merge(
        contacts[["time_ns", "replica_id", "bound"]], on=["time_ns", "replica_id"]
    )
    ss_bound = ss_join[ss_join["bound"]].drop(columns="bound").reset_index(drop=True)
    ss_unbound = ss_join[~ss_join["bound"]].drop(columns="bound").reset_index(drop=True)

    pca_xvg = master / "pca_short_1-6_proj.xvg"
    if pca_xvg.exists():
        pca_df = read_xvg_proj(pca_xvg, contacts)

    sasa_xvg = master / "sasa_prot.xvg"
    if sasa_xvg.exists():
        sasa_df = read_xvg_sasa(sasa_xvg, contacts)
        print(sasa_df.shape)
        print(sasa_df.head())
        print(sasa_df.tail())
    else:
        raise FileNotFoundError(sasa_xvg)

    # store or process further …
    mem_results[label_name] = dict(
        ss_bound=ss_bound,
        ss_unbound=ss_unbound,
        pca_bound=pca_df[pca_df["bound"]].drop(columns="bound").reset_index(drop=True),
        pca_unbound=pca_df[~pca_df["bound"]]
        .drop(columns="bound")
        .reset_index(drop=True),
        sasa_bound=sasa_df[sasa_df["bound"]]
        .drop(columns="bound")
        .reset_index(drop=True),
        sasa_unbound=sasa_df[~sasa_df["bound"]]
        .drop(columns="bound")
        .reset_index(drop=True),
    )

    print(
        f"{label_name:18s}:  bound frames = {len(ss_bound):6d}   "
        f"unbound = {len(ss_unbound):6d}"
    )


colours = {"bound": "tab:red", "unbound": "tab:blue"}
for lname, res in mem_results.items():

    # ---------- α-helix counts histogram ---------------------------------------
    b_alpha = res["ss_bound"]["Alpha"].values
    u_alpha = res["ss_unbound"]["Alpha"].values

    N = 200020  # or compute it once: len(all_frames_across_both_sets)

    plt.figure(figsize=(4.5, 3.2))
    bins = np.arange(105, 130, 1)  # 0 … 132 residues, step 4
    plt.hist(
        u_alpha,
        bins=bins,
        weights=np.ones_like(u_alpha) / N,
        alpha=0.6,
        color=colours["unbound"],
        label="unbound",
    )
    plt.hist(
        b_alpha,
        bins=bins,
        weights=np.ones_like(b_alpha) / N,
        alpha=0.6,
        color=colours["bound"],
        label="bound",
    )

    plt.xlabel("Residues in α-helix")
    plt.ylabel("Kernel density estimate")
    # plt.title(f"{lname} – α-helix distribution")
    plt.legend(frameon=False)
    plt.tight_layout()
    # plt.savefig(plot_dir / f"{lname}_alpha_hist.png", dpi=300)
    plt.show()
    plt.close()

    print(f"[saved]  {lname}:  RMSD & α-helix histograms")

    sns.set(style="ticks")

colours = {
    "100% DOPG": "#d62728",  # red
    "50% DOPG": "#1f77b4",  # blue
    "0% DOPG": "#2ca02c",  # green
}

STATE_COLOURS = {
    "unbound": (0.12, 0.47, 0.71, 0.55),  #  to rgba blues
    "bound": (0.85, 0.37, 0.01, 0.55),
}  #  to rgba oranges
BOX_SCALE = 1.4  # horizontal scaling factor for inner boxes


STATE_PALETTE = {
    "unbound": (0.41, 0.24, 0.60, 0.55),  # purple 55 % α
    "bound": (1.00, 0.50, 0.00, 0.55),
}  # orange 55 % α

#  2. ----- α‑Helix counts violin ----
# build {membrane: (bound_array, unbound_array)} for α‑helix counts

alpha_dict = {
    lab: (bags["ss_bound"]["Alpha"], bags["ss_unbound"]["Alpha"])
    for lab, bags in mem_results.items()
}

split_violin_boxes(
    alpha_dict,
    ylabel="Residues in α‑helix",
    fname="violin_alpha_split.png",
    order=["0% DOPG", "50% DOPG", "100% DOPG"],
    bw_adj=2,
    ylim=[100, 135],
    legend=True,
    special_case=True,
)


# %% PCA overlay -------------------------------------------------------

colours = {"100% DOPG": "#d62728", "50% DOPG": "#1f77b4", "0% DOPG": "#2ca02c"}
fig, ax = plt.subplots(
    figsize=(5.2, 3),
    constrained_layout=True,  # keep the inner axes equal everywhere
)


for mkey, bags in mem_results.items():
    col = colours[mkey]
    lab = mkey
    N_TOTAL = len(bags["pca_bound"]) + len(bags["pca_unbound"])

    one_kde(ax, bags["pca_bound"]["PC1"], col, f"{lab} (bound)", ls="-", norm=False)
    one_kde(
        ax, bags["pca_unbound"]["PC1"], col, f"{lab} (unbound)", ls="--", norm=False
    )

ax.set_xlabel("PC1 projection")
ax.set_xlim(-20, 13.15)
ax.set_ylim(0, 0.25)
ax.set_yticks([0, 0.05, 0.1, 0.15, 0.2, 0.25])
ax.set_ylabel("Kernel‑density estimate")
# ax.set_title("PC1 distributions – bound vs. unbound")
# ax.legend(frameon=False, fontsize=8, ncol=1, loc="upper right")

# inset zoom
x0, x1 = -15, -5  # shoulder window
y0, y1 = 0, 0.02

axins = inset_axes(
    ax,
    width="30.2%",
    height="40%",
    loc="upper left",
    bbox_to_anchor=(0.108, -0.25, 1, 1),
    bbox_transform=ax.transAxes,
    borderpad=1.2,
)

for spine in axins.spines.values():  # top, bottom, left, right
    spine.set_color("black")

for mkey, bags in mem_results.items():
    col = colours[mkey]
    NTOT = len(bags["pca_bound"]) + len(bags["pca_unbound"])

    # ---- bound
    sns.kdeplot(
        x=bags["pca_bound"]["PC1"],
        weights=np.full(len(bags["pca_bound"]), 1 / NTOT),
        ax=axins,
        color=col,
        linestyle="-",
        linewidth=1.4,
        fill=False,
        bw_adjust=1.0,
        clip=(x0, x1),  # ← plot only within the zoom window
    )

    # ---- unbound
    sns.kdeplot(
        x=bags["pca_unbound"]["PC1"],
        weights=np.full(len(bags["pca_unbound"]), 1 / NTOT),
        ax=axins,
        color=col,
        linestyle="--",
        linewidth=1.4,
        fill=False,
        bw_adjust=1.0,
        clip=(x0, x1),
    )

axins.set_xlim(x0, x1)
axins.set_ylim(y0, y1)
# axins.set_xticks([x0,x1])
axins.set_yticks([y0, y1])
axins.set_xticks([])
axins.tick_params(axis="both", labelsize=10)
axins.set_xticklabels([])
# axins.set_yticklabels([])

axins.set_xlabel("")  # hide x‑label
axins.set_ylabel("")  # hide y‑label

mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="black", lw=0.8)

# Create custom legend handles
custom_lines = [
    Line2D([0], [0], color="black", lw=1.6, linestyle="-", label="bound"),
    Line2D([0], [0], color="black", lw=1.6, linestyle="--", label="unbound"),
]

# Add a minimal legend inside the plot
ax.legend(
    handles=custom_lines,
    loc="upper right",
    frameon=False,
    fontsize=10,
)

fig.tight_layout()
fig.savefig(plot_dir / "overlay_PC1_bound_unbound_zoom_norm_white.png", dpi=300)
plt.show()
plt.close(fig)


# %% SASA violins

STATE_COLOURS = {
    "unbound": (0.12, 0.47, 0.71, 0.55),  #  to rgba blues
    "bound": (0.85, 0.37, 0.01, 0.55),
}  #  to rgba oranges
BOX_SCALE = 1.4  # horizontal scaling factor for inner boxes


STATE_PALETTE = {
    "unbound": (0.41, 0.24, 0.60, 0.55),  # purple 55 % α
    "bound": (1.00, 0.50, 0.00, 0.55),
}  # orange 55 % α

# TOTAL‑protein SASA
sasa_total = {
    lab: (bags["sasa_bound"]["SASA_total"], bags["sasa_unbound"]["SASA_total"])
    for lab, bags in mem_results.items()
}

split_violin_boxes(
    sasa_total,
    ylabel="Total SASA (nm$^2$)",
    fname="violin_SASA_total_split.png",
    order=["0% DOPG", "50% DOPG", "100% DOPG"],
    ylim=[90, 120],
    legend=False,
    special_case=True,
)
# TRP‑71 SASA
sasa_trp = {
    lab: (bags["sasa_bound"]["SASA_trp71"], bags["sasa_unbound"]["SASA_trp71"])
    for lab, bags in mem_results.items()
}

split_violin_boxes(
    sasa_trp,
    ylabel="TRP 71 SASA (nm$^2$)",
    fname="violin_SASA_TRP71_split.png",
    order=["0% DOPG", "50% DOPG", "100% DOPG"],
    ylim=[0, 1.4],
    legend=False,
    special_case=True,
)
