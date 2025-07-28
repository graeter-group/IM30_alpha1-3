# %%
#!/usr/bin/env python3
"""
make_pymol_views.py and write three .pml scripts for PyMOL

usage
-----
python make_pymol_views.py
pymol protein_charge.pml
pymol protein_rescat.pml
pymol protein_occ.pml
"""

from pathlib import Path
import sys, numpy as np, MDAnalysis as mda, pandas as pd

pdb_in = Path("/.../7o3y_chainA_r26-156_capped.pdb")  # original protein PDB
boot_csv = Path(
    "/.../replicates/0_POPC_100_DOPG/analysis_master/residue_occupancy_bootstrap.csv"
)  # bootstrap table, do for every membrane setup

work_dir = Path("/.../replicates/global_analysis/renders/scripts")  # Save directory
work_dir.mkdir(exist_ok=True)

# ---------- 1. read bootstrap table (resid, occ_mean) ------------------
boot = pd.read_csv(boot_csv)
occ_dict = dict(zip(boot.resid, boot.occ_mean))  # resid -> 0–1

# ---------- 2. write PDB with occupancy in B-factor --------------------
u = mda.Universe(pdb_in)
tf = np.zeros(len(u.atoms))
for res in u.residues:
    tf[res.atoms.indices] = occ_dict.get(res.resid, 0.0)
u.add_TopologyAttr("tempfactors", tf)
pdb_occ = work_dir / "0PC_100PG_protein_occ_bfac.pdb"
u.atoms.write(pdb_occ)
print("PDB with occupancy in B-factor  →", pdb_occ)

# ---------- 3. residue category dictionaries ---------------------------
res_cat = {
    0: ["ASP", "GLU"],
    1: ["ARG", "HIS", "LYS"],
    2: ["SER", "THR", "ASN", "GLN", "CYS", "GLY", "TYR"],
    3: ["ALA", "LEU", "ILE", "VAL", "MET", "PRO", "TRP", "PHE"],
}

charge_pal = {0: "red", 1: "blue", 2: "green", 3: "grey"}
cat_pal = {0: "red", 1: "blue", 2: "green", 3: "orange"}


def write_pml(fname, cmds):
    Path(fname).write_text("bg_color white\n" + "\n".join(cmds) + "\n")
    print("wrote", fname)


# ---------- 3a. charge colouring script --------------------------------
cmds = [f"load {pdb_in}, prot", "hide everything, prot", "show cartoon, prot"]
for cat, colour in charge_pal.items():
    sel = " or ".join(f"resn {aa}" for aa in res_cat[cat])
    cmds += [f"select cat{cat}, prot and ({sel})", f"color {colour}, cat{cat}"]
write_pml(work_dir / "protein_charge.pml", cmds)

# ---------- 3b. residue-class colouring script -------------------------
cmds = [f"load {pdb_in}, prot", "hide everything, prot", "show cartoon, prot"]
for cat, colour in cat_pal.items():
    sel = " or ".join(f"resn {aa}" for aa in res_cat[cat])
    cmds += [f"select cat{cat}, prot and ({sel})", f"color {colour}, cat{cat}"]
write_pml(work_dir / "protein_rescat.pml", cmds)

# ---------- 3c. occupancy colouring + labels ---------------------------
cmds = [
    f"load {pdb_occ}, prot",
    "hide everything, prot",
    "show cartoon, prot",
    # blue-white-red spectrum on B-factor (0-1)
    "spectrum b, grey_yellow_red, prot, minimum=0, maximum=0.1",
    # numerical label on every C-alpha
    'label prot and name CA, "%s %s %.2f" % (resn,resi,b)',
    # ─────────────────  new view / render settings  ──────────────────
    # turn OFF depth-cue & specular highlights
    "set depth_cue, 0",  # no fog
    "set specular, off",  # flat shading
    # ► camera orientation copied from “get_view”
    r"""set_view (\
     0.759469748,    0.526663065,   -0.381884217,\
     0.506365240,   -0.110064447,    0.855256557,\
     0.408400446,   -0.842913628,   -0.350278139,\
     0.000000000,    0.000000000, -276.168914795,\
   -75.736999512,   58.494400024,  -84.796897888,\
   217.733703613,  334.604187012,  -20.000000000 )""",
    # ► high-quality ray-traced image (2400×2400 px, ~300 dpi)
    "ray 2800, 2000",
    "png protein_occ.png",
]
write_pml(work_dir / "protein_occ.pml", cmds)
