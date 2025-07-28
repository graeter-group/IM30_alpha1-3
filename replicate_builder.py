#!/usr/bin/env python3

# %%
"""
Build replica systems:
  • from 3 pre-equilibrated membranes
  • and 20 protein conformations
  • shift membrane so lowest atom sits 1.5 nm above box-bottom
  • place protein 2.0 nm above membrane top, add 1.5 nm solvent head-room
  • rotate protein
  • combine systems
  • copy all forcefield and mdp files
  • adjust topology
  • solvate and ionise (150 mM KCl, neutralize with K))

Requires: MDAnalysis, GROMACS (gmx)
"""

from pathlib import Path
import shutil, os, subprocess, sys, numpy as np, MDAnalysis as mda
from MDAnalysis.core.universe import Merge
from MDAnalysis.transformations import unwrap, center_in_box
from MDAnalysis.lib.transformations import rotation_matrix
from MDAnalysis.analysis.rms import RMSD
from MDAnalysis.analysis import align, distances
from MDAnalysis.analysis.rms import RMSF
from MDAnalysis.lib.distances import self_distance_array
from MDAnalysis.lib.distances import distance_array
import matplotlib.pyplot as plt

# from sklearn_extra.cluster import KMedoids

from collections import Counter
from collections import defaultdict

# %%

membrane_paths = [
    Path(
        "/.../pure_membranes/100_POPC_0_DOPG/gromacs/step6.7_equilibration_nosolv.gro"
    ),
    Path(
        "/.../pure_membranes/50_POPC_50_DOPG/gromacs/step6.7_equilibration_nosolv.gro"
    ),
    Path(
        "/.../pure_membranes/0_POPC_100_DOPG/gromacs/step6.7_equilibration_nosolv.gro"
    ),
]  # equilibrated final membrane coordinate files

# For saving the turned proteins for visualiziaton and more
protein_out_dir = Path("/.../templates/final_proteins")

# For inputting the protein conformations
protein_dir = Path("/.../IM30/gromacs")
protein_paths = list(
    protein_dir.glob("frame*")
)  # 25 frames total were extracted from the 200 ns trajectory
protein_paths.sort()
protein_paths = protein_paths[6:]  # Only extract structures after 40 ns (6 to 25)

templ_dir = Path("/.../templates")

scripts = list(templ_dir.glob("*.sh"))
forcefield_top = Path(templ_dir / "topol.top")
forcefield_dir = Path(templ_dir / "toppar")  # folder with .itp files
mdp_files = list(templ_dir.glob("*.mdp"))
water_box = Path(templ_dir / "tip3_216.gro")  # solvent template
output_root = Path("/.../replicates")

# sanity check if all files are present
for p in (
    membrane_paths
    + protein_paths
    + mdp_files
    + scripts
    + [forcefield_top, forcefield_dir, water_box]  # add mpds
):
    if not p.exists():
        sys.exit(f"ERROR: file/folder not found – {p}")

output_root.mkdir(exist_ok=True)


# helper functions
def run(cmd, **kw):
    """subprocess.run wrapper that aborts on non-zero exit"""
    print(">>", " ".join(cmd) if isinstance(cmd, list) else cmd)
    subprocess.run(cmd, check=True, **kw)


def write_topology(
    u: mda.Universe,
    template_top: Path,
    output_top: Path,
    lipid_names=("POPC", "DOPG"),
    protein_name="PROA",
):
    """
    Build a complete topol.top:
      • keeps header from template_top
      • adds one protein line (protein_name 1)
      • counts lipid_names in Universe and adds one line per lipid
    """
    # count lipids
    counts = Counter(res.resname for res in u.residues if res.resname in lipid_names)

    # build [ molecules ] block
    mol_block = ""
    for name in sorted(counts):
        mol_block += f"{name:<18}{counts[name]}\n"

    # write new topology
    header = Path(template_top).read_text()
    output_top = Path(output_top)

    # keep a backup if the file exists
    if output_top.exists():
        shutil.copy(output_top, output_top.with_suffix(".bak"))

    output_top.write_text(header.rstrip() + "\n" + mol_block)
    print(f"{output_top} written")
    for res, n in counts.items():
        print(f"  {res:>5}  {n} lipids")


def enforce_tilt(universe, maxdeg=45, protein_sel="protein"):
    """Turn protein within the universe so its principal axis is <= 45 degrees against the xy-plane"""
    sel = universe.select_atoms(protein_sel)

    # 1. longest principal axis
    long_axis = sel.principal_axes()[0]  # already unit vector

    # 2. current tilt
    z = np.array([0.0, 0.0, 1.0])
    theta = np.degrees(np.arccos(abs(long_axis @ z)))
    print(f"initial tilt: {theta:.2f}°")

    if theta <= maxdeg + 1e-3:
        print("within limit, no rotation applied")
        return theta

    # 3. build rotation that reduces tilt to exactly maxdeg
    rotvec = np.cross(long_axis, z)
    rotvec /= np.linalg.norm(rotvec)
    dtheta = theta - maxdeg  # degrees
    R = rotation_matrix(np.radians(dtheta), rotvec)[:3, :3]

    # 4. rotate rigidly about protein COM
    com = sel.center_of_mass()
    sel.translate(-com)
    sel.rotate(R, point=np.zeros(3))  # rotate about origin
    sel.translate(com)

    # 5. recompute axis (force fresh calc by re-calling with reset=True)
    new_axis = sel.principal_axes()[0]
    new_theta = np.degrees(np.arccos(abs(new_axis @ z)))
    print(f"rotated down, tilt now {new_theta:.2f}°")

    return new_theta


def random_xy_rotation(
    universe, seed=12345, protein_sel="protein and backbone", angle_range=(0.0, 360.0)
):
    """
    Rotate the protein rigidly around the box z-axis by a pseudo-random
    angle (uniform in angle_range), reproducible via seed.

    Parameters
    ----------
    universe     : MDAnalysis.Universe  (already contains protein + membrane)
    seed         : int        Seed for reproducibility
    protein_sel  : str        Atom selection to rotate
    angle_range  : (min,max)  Degrees; default 0–360
    """
    rng = np.random.default_rng(seed)
    angle_deg = rng.uniform(*angle_range)
    angle_rad = np.radians(angle_deg)

    # 3×3 rotation matrix about z
    Rz = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad), 0.0],
            [np.sin(angle_rad), np.cos(angle_rad), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    sel = universe.select_atoms(protein_sel)
    com = sel.center_of_mass()
    sel.translate(-com)  # move COM to origin
    sel.rotate(Rz, point=np.zeros(3))  # apply rotation
    sel.translate(com)  # move back

    print(f"Rotated protein by {angle_deg:.2f}° in XY plane")


def min_periodic_distance(prot_coords, box_dims):
    """
    Compute the minimum distance (Å) between prot_coords (N×3) and any
    periodic copy of itself, excluding the original (i.e. no i=i).
    This helps ensure we have no self-interaction across pbcs.

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
        periodic image of a *different* atom.
    """
    Lx, Ly, Lz = box_dims
    originals = prot_coords.copy()

    # 1) Build all translations in X and Y directions (±1 box shift).
    #    We do not shift in Z, assuming the membrane normal is Z:
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

    # 2) For each shift, compute distances between originals and shifted coords
    min_dist = np.inf
    for shift in shifts:
        shifted = originals + shift  # (N,3) array
        # Compute all pairwise distances: originals[i] vs shifted[j]
        d2 = distance_array(originals, shifted, box=None)  # no box; we already shifted
        # We do not need to mask i=j, because shifted = originals + shift,
        # so diagonal would be |orig_i - (orig_i+shift)| = |shift| >> 0.
        min_dist = min(min_dist, d2.min())

    return min_dist


max_prot_top = 240 - 15  # Arbitary value that can be hand selected as control
max_mem_top = 0

# workflow ----------------------------------------------------------------
for mem_path in membrane_paths:
    mem_tag = mem_path.parent.parent.name  # e.g. mem1
    mem_dir = output_root / mem_tag
    mem_dir.mkdir(exist_ok=True)

    seed = 1234

    for prot_path in protein_paths:
        prot_tag = prot_path.stem  # e.g. prot1
        sys_dir = mem_dir / prot_tag
        sys_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n### Building replica: {mem_tag} + {prot_tag}")

        # ---- 1. MDAnalysis: load & shift membrane -----------------------
        mem_u = mda.Universe(str(mem_path))
        # Get lowest membrane atom coordinate
        z_min_mem = mem_u.atoms.positions[:, 2].min()
        # Shfit all atoms so lowest atom is 1.5 nanometrs (15 angstom) above bottom edge
        mem_u.atoms.translate([0, 0, 15 - z_min_mem])
        # Get highest membrane atoms postions post translation
        mem_top = mem_u.atoms.positions[:, 2].max()
        if mem_top > max_mem_top:
            max_mem_top = mem_top

        # ---- 2. load & shift protein -----------------------------------
        prot_u = mda.Universe(str(prot_path))

        # wrap protein (make pbc whole) and Put protein in middle of box
        # 2.1. unwrap the molecule → no more PBC breaks, recommended to do this with gmx traj, if desired
        # prot_u.trajectory.add_transformations(unwrap(prot_u.atoms))

        # --- 3. give it a fresh 150 Å cubic box (15 nm) thats bigger than some unfolding if it happens -----------------
        cube_edge = 150.0  # Å
        prot_u.dimensions = np.array([cube_edge, cube_edge, cube_edge, 90, 90, 90])

        # --- 4. center the COM in that new box (XY and Z) -----------------
        prot_u.trajectory.add_transformations(
            center_in_box(prot_u.atoms, center="mass", wrap=False)
        )

        # Ensure the angle between the protein and the membrane (xy plane) is less than 45 degrees, otherwise rotate to that
        enforce_tilt(maxdeg=45, universe=prot_u)

        # Rotate protein randomy in xy
        random_xy_rotation(
            prot_u, seed=seed, protein_sel="protein", angle_range=(0.0, 360.0)
        )

        # 4.1. centre the protein COM in the membrane XY plane
        box_x = mem_u.dimensions[0]  #  (membrane box length in X)
        box_y = mem_u.dimensions[1]  #   (membrane box length in Y)

        prot_com_xy = prot_u.atoms.center_of_mass()[:2]  #  x and y only
        shift_xy = np.array(
            [box_x / 2.0 - prot_com_xy[0], box_y / 2.0 - prot_com_xy[1], 0.0]
        )  # z-shift is already done
        prot_u.atoms.translate(shift_xy)

        # Similar
        z_min_prot = prot_u.atoms.positions[:, 2].min()
        # Shift it 2 nm above the membrane
        prot_u.atoms.translate([0, 0, (mem_top + 20) - z_min_prot])
        prot_top = prot_u.atoms.positions[:, 2].max()
        if (
            prot_top > max_prot_top
        ):  # If the highest atom exceeds your manually set buffer space
            max_prot_top = prot_top
            print("New Max Z-coordinate!")
            print(prot_top + 15)
            raise RuntimeError(
                f"[ABORT] Protein top ({prot_top:.2f} nm) + 15  buffer "
                f"exceeds allowed box height of {max_prot_top + 15} Angstrom. "
                "Choose a different frame or increase Lz."
            )

        # ---- 5. merge & define box -------------------------------------
        combined = Merge(prot_u.atoms, mem_u.atoms)

        box_z = 240  # If max_prot + 15 is ever larger than this, change it
        combined.dimensions = np.array([box_x, box_y, box_z, 90, 90, 90])
        print(combined.dimensions)

        # 5.1 Renumber residue ids
        # continuous renumber: 1..N for *all* residues in the combined system
        # Relevant for printing to gromacs files
        for new_id, res in enumerate(combined.residues, start=1):
            res.resid = new_id

        combined_gro = sys_dir / "combined.gro"
        combined.atoms.write(str(combined_gro))
        print(
            f" -> combined.gro written  (box {box_x:.2f}×{box_y:.2f}×{box_z:.2f} Angstrom)"
        )

        box_x, box_y = combined.dimensions[:2]  # Å
        prot_xyz = combined.select_atoms("protein").positions  # Å

        # Save protein configs
        prot_save = combined.select_atoms("protein")
        outfile = protein_out_dir / f"{prot_path.stem}_final.gro"

        prot_save.write(str(outfile))

        # 5.2. check 15-Å safety margin in X and Y on insertion
        # If any atom is close to the wall, asserts its still over 20 Angstrom away from its periodic image
        margin = 15.0  # Å
        hits = (
            (prot_xyz[:, 0] < margin)
            | (prot_xyz[:, 0] > box_x - margin)
            | (prot_xyz[:, 1] < margin)
            | (prot_xyz[:, 1] > box_y - margin)
        )

        if hits.any():
            nbad = hits.sum()
            min_image_dist = min_periodic_distance(
                prot_save.positions.copy(), (box_x, box_y, box_z)
            )
            if min_image_dist < 20:
                raise RuntimeError(
                    f"[ABORT] {nbad} protein atoms lie closer than {margin} Å "
                    f"to the XY box boundary (box {box_x:.1f} × {box_y:.1f} Å). "
                    f"the protein got to only {min_image_dist} Angstrom dstiacne to its periodic image"
                    # f"Choose another frame or enlarge box. The protein unfolded to a total of {max_len} Angstrom"
                )

        # ---- 6. copy and edit shared input files --------------------------------
        shutil.copy(forcefield_top, sys_dir / "topol.top")
        shutil.copytree(forcefield_dir, sys_dir / "toppar", dirs_exist_ok=True)
        for mdp in mdp_files:
            shutil.copy(mdp, sys_dir / mdp.name)

        # Edit mdp file wiht gen-seed
        mdp_path = Path(sys_dir / "step6.1_equilibration.mdp")
        lines = mdp_path.read_text().splitlines()

        # 6.1. Replace the gen-seed line for velocity generation
        new_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("gen-seed"):
                # preserve indentation, but replace the value
                indent = line[: line.find("gen-seed")]
                new_lines.append(f"{indent}gen-seed = {seed}")
            else:
                new_lines.append(line)

        # 6.2. Write back to the same file (or a new one)
        mdp_path.write_text("\n".join(new_lines) + "\n")

        for scirpt in scripts:
            shutil.copy(scirpt, sys_dir / scirpt.name)

        write_topology(mem_u, sys_dir / "topol.top", sys_dir / "topol.top")

        # ---- 7. solvate & ionise with gmx---------------------------------------
        cwd = Path.cwd()
        try:
            os.chdir(sys_dir)
            run(
                [
                    "gmx",
                    "solvate",
                    "-quiet",
                    "-cp",
                    "combined.gro",
                    "-cs",
                    str(water_box),
                    "-o",
                    "solvated.gro",
                    "-p",
                    "topol.top",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )

            run(
                [
                    "gmx",
                    "grompp",
                    "-quiet",
                    "-f",
                    "dummy.mdp",
                    "-c",
                    "solvated.gro",
                    "-p",
                    "topol.top",
                    "-o",
                    "ions.tpr",
                    "-maxwarn",
                    "1",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )

            # echo through a pipe so we select the TIP3 group automatically
            run(
                "echo TIP3 | gmx genion -quiet -s ions.tpr -o final.gro -p topol.top "
                "-pname POT -nname CLA -neutral -conc 0.12",
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
            print("   → final.gro with 150 mM KCl written")

            run(
                "printf '%s\\n' "
                "'del 2-18' "
                "'name 0 SYSTEM' "
                "'name 1 SOLU' "
                "'r POT | r CLA | r TIP3' "
                "'name 2 SOLV' "
                "'r POPC | r DOPG' "
                "'r DOPG | r POPC' "
                "'name 3 MEMB' "
                "'del 4' "
                "'1 | 3' "
                "'name 4 SOLU_MEMB' "
                "'q' "
                "| gmx make_ndx -quiet -f final.gro -o index.ndx",
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
        finally:
            os.chdir(cwd)

        seed += 1

print("\nAll replicas built ✔")
print("maximum z dimenison:")
print(max_prot_top + 15)
print("maximum mem top:")
print(max_mem_top + 15)


# %% Protein Strucutre analysis
# Some plotting functionalities to investigate the membranes, to choose appropriate box sizes and more

from MDAnalysis.lib.distances import self_distance_array


prot_folder = Path("/.../IM30/gromacs")

u = mda.Universe(
    prot_folder / "step5_production_protein.gro",
    prot_folder / "step5_production_protein_whole.xtc",
)

protein = u.select_atoms("protein")

# again these transformation should be preferably done via gmx
# u.trajectory.add_transformations(
#    unwrap(protein),  # join fragments across PBC
#    center_in_box(protein, center="mass"),  # put COM in box centre
# )
tail = u.select_atoms("resid 140-156 and backbone")
bb = protein.select_atoms("backbone")
print(bb)


# --- 1. RMSD -------------------------------------------------------------
rmsd = RMSD(bb).run()
time, rmsd_values = rmsd.rmsd.T[1], rmsd.rmsd.T[2] / 10  # A -> nm

# --- 2. Radius of gyration ----------------------------------------------
rg = np.array([bb.radius_of_gyration() for ts in u.trajectory])

# --- 3. End–to–end distance (first & last CA) ----------------------------

# holder for max length per frame
maxlen_nm = []

for ts in u.trajectory:  # iterates over every frame
    dists = self_distance_array(protein.positions)  # Å,  (N*(N-1)/2) elements
    maxlen_nm.append(dists.max())  # convert Å → nm

maxlen_nm = np.asarray(maxlen_nm)  # shape (n_frames,)
print(f"overall maximum length = {maxlen_nm.max():.2f} Angstrom")

# --- 4. RMSF -------------------------------------------------------------
align.AlignTraj(u, bb, select="backbone", in_memory=True).run()
rmsf = RMSF(bb).run().rmsf

# --- 5. plots  -------------------------------------------------
plt.figure()
plt.plot(time, rmsd_values)
plt.ylabel("Backbone RMSD (nm)")
plt.tight_layout()
plt.savefig(prot_folder / "bb_rmsd.png", dpi=300, bbox_inches="tight")
plt.show()
plt.figure()
plt.plot(time, rg)
plt.ylabel("Rg (Angstrom)")
plt.tight_layout()
plt.savefig(prot_folder / "bb_rmsd.png", dpi=300, bbox_inches="tight")
plt.show()
plt.figure()
plt.plot(time, maxlen_nm)
plt.ylabel("Max. dist between any protein atoms (Angstrom)")
plt.tight_layout()
plt.savefig(prot_folder / "protein_max_dist.png", dpi=300, bbox_inches="tight")
plt.show()


# %% Membrane anaysis
# Some plotting functionalities to investigate the membranes, to choose appropriate box sizes and more

membrane_paths = [
    Path("/.../pure_membranes/100_POPC_0_DOPG/gromacs/"),
    Path("/.../pure_membranes/50_POPC_50_DOPG/gromacs/"),
    Path("/.../pure_membranes/0_POPC_100_DOPG/gromacs/"),
]

# To investigate compression/expansion of membranes during equilibration
for mem in membrane_paths:
    u = mda.Universe(
        mem / "step6.7_equilibration_nosolv.gro",
        mem / "step6.7_equilibration_nosolv.xtc",
    )
    # --------  XY box lengths over time  -----------------------------------
    box = np.array([ts.dimensions[:2].copy() for ts in u.trajectory])  # Å
    time_ps = np.array([u.trajectory.time for ts in u.trajectory])  # ps

    Lx = box[:, 0]
    Ly = box[:, 1]

    plt.figure()
    plt.ylim(140, 155)
    plt.plot(time_ps / 1000, Lx, label="Box-XY")
    plt.xlabel("Time (ns)")
    plt.ylabel(f"Box length (nm) {mem.parent.name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(mem / "box_xy.png", dpi=300, bbox_inches="tight")
    plt.show()

# ------------ To investigate lipid distribution ----------
import MDAnalysis as mda, numpy as np, matplotlib.pyplot as plt
from MDAnalysis.lib.distances import capped_distance, self_capped_distance

u = mda.Universe(
    membrane_paths[1] / "step6.7_equilibration_nosolv.gro",
    membrane_paths[1] / "step6.7_equilibration_nosolv.xtc",
)


POPC = u.select_atoms("resname POPC and name P")  # 298 atoms
DOPG = u.select_atoms("resname DOPG and name P")  # 298 atoms
N_A, N_B = len(POPC), len(DOPG)
x_B_bulk = N_B / (N_A + N_B)  # 0.50

rcut = 10.0  # Å  (larger than half-box → everyone is a neighbour)
stride = 10

frac_B = []

# ---------- loop over frames -------------------------------------------

time_ns = time_ps[::stride]  # ns

for ts in u.trajectory[::stride]:

    # POPC–DOPG pairs  (each appears *once*)
    pairs_AB = capped_distance(
        POPC.positions,
        DOPG.positions,
        max_cutoff=rcut,
        box=ts.dimensions,
        return_distances=False,
    )
    hetero = np.bincount(pairs_AB[:, 0], minlength=N_A)  # POPC index

    # POPC–POPC pairs (self_capped_distance returns upper triangle only)
    pairs_AA = self_capped_distance(
        POPC.positions, max_cutoff=rcut, box=ts.dimensions, return_distances=False
    )
    # add the pair to both partners
    neighbours_A = np.bincount(pairs_AA[:, 0], minlength=N_A)
    neighbours_B = np.bincount(pairs_AA[:, 1], minlength=N_A)
    homo = neighbours_A + neighbours_B

    total = hetero + homo
    frac_B.append((hetero[total > 0] / total[total > 0]).mean())

frac_B = np.asarray(frac_B)


# ---------- plot -------------------------------------------------------
plt.figure()
plt.plot(time_ns, frac_B, label="observed POPC→DOPG")
plt.axhline(x_B_bulk, c="k", ls="--", label="random mix (0.5)")
plt.xlabel("time (ns)")
plt.ylabel("fraction DOPG neighbours")
plt.legend()
plt.tight_layout()
plt.savefig(membrane_paths[1] / "lipid_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

# %% How many lipids per leaftet?

PC_PG_membrane_path = Path("/.../pure_membranes/50_POPC_50_DOPG/gromacs/")


u = mda.Universe(
    PC_PG_membrane_path / "step6.7_equilibration_nosolv.gro",
    PC_PG_membrane_path / "step6.7_equilibration_nosolv.xtc",
)


# 2. Select P atoms to mark each lipid species
popc_P = u.select_atoms("resname POPC and name P")
dopg_P = u.select_atoms("resname DOPG and name P")

# build a dict mapping resid -> resname (“POPC” or “DOPG”).
resname_map = {
    res.resid: res.resname for res in u.residues if res.resname in ("POPC", "DOPG")
}


# 3. Loop over frames, assing midplane and define leaflets
# store, for each frame, counts of (upper/lower) × (POPC/DOPG).
# Using a list of dicts to plot vs. time if desired.

time_ps = []
counts_per_frame = []

for ts in u.trajectory:
    time_ps.append(ts.time)

    # 3a. Collect all phosphate z-coordinates (Å)
    all_P = u.select_atoms("resname POPC DOPG and name P")
    z_all = all_P.positions[:, 2]  # array of shape (n_lipids,)
    z_cut = z_all.mean()  # midplane = mean z of all P

    # 3b. For each phosphate, check if above or below z_cut
    # Build counters
    cnt = defaultdict(int)
    for atom in all_P:
        resid = atom.resid
        resnm = resname_map[resid]  # "POPC" or "DOPG"
        z_P = atom.position[2]
        leaflet = "upper" if (z_P > z_cut) else "lower"
        cnt[(resnm, leaflet)] += 1

    # 3c. Save counts for this frame
    counts_per_frame.append(
        {
            "time": ts.time,
            "POPC_upper": cnt.get(("POPC", "upper"), 0),
            "POPC_lower": cnt.get(("POPC", "lower"), 0),
            "DOPG_upper": cnt.get(("DOPG", "upper"), 0),
            "DOPG_lower": cnt.get(("DOPG", "lower"), 0),
            "z_cut": z_cut,
        }
    )

# Convert to NumPy arrays for easy plotting
time_ns = np.array(time_ps) / 1000.0  # ps→ns
p_up = np.array([f["POPC_upper"] for f in counts_per_frame])
p_lo = np.array([f["POPC_lower"] for f in counts_per_frame])
d_up = np.array([f["DOPG_upper"] for f in counts_per_frame])
d_lo = np.array([f["DOPG_lower"] for f in counts_per_frame])
z_cuts = np.array([f["z_cut"] for f in counts_per_frame])

# 4. Plot Results

# 4a. “Trace over time” of how many lipids are in each leaflet
plt.figure(figsize=(5, 4))
plt.plot(time_ns, p_up, label="POPC (upper)")
plt.plot(time_ns, p_lo, label="POPC (lower)")
plt.plot(time_ns, d_up, label="DOPG (upper)")
plt.plot(time_ns, d_lo, label="DOPG (lower)")
plt.xlabel("Time (ns)")
plt.ylabel("Number of lipids")
plt.legend(loc="upper right", frameon=False)
plt.tight_layout()
plt.title("Leaflet populations vs. time")
plt.show()

# 4b. Optional: plot midplane z to check if it drifts
plt.figure(figsize=(5, 3))
plt.plot(time_ns, z_cuts, c="k")
plt.xlabel("Time (ns)")
plt.ylabel("Mean phosphate z (Å)")
plt.title("Bilayer midplane vs. time")
plt.tight_layout()
plt.show()

# 4c. If you just want the average over all frames:
avg_popc_up = p_up.mean()
avg_popc_lo = p_lo.mean()
avg_dopg_up = d_up.mean()
avg_dopg_lo = d_lo.mean()

print("Average leaflet counts (over entire trajectory):")
print(f"  POPC – upper: {avg_popc_up:.1f}, lower: {avg_popc_lo:.1f}")
print(f"  DOPG – upper: {avg_dopg_up:.1f}, lower: {avg_dopg_lo:.1f}")
