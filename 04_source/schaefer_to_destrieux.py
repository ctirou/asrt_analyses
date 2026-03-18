"""
Map Schaefer 200-parcel (7 Networks) labels to Destrieux (a2009s) atlas regions
on the fsaverage surface, using majority-vote per hemisphere.

Annotation files required in FREESURFER_DIR/fsaverage/label/:
  - lh/rh.Schaefer2018_200Parcels_7Networks_order.annot
    (download from: github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/
     brain_parcellation/Schaefer2018_LocalGlobal)
  - lh/rh.a2009s.annot  (ships with FreeSurfer fsaverage)
"""

import csv

import nibabel as nib
import numpy as np

from config import FREESURFER_DIR

SUBJECT      = 'fsaverage2'
LABEL_DIR    = FREESURFER_DIR / SUBJECT / 'label'       # where .annot files live
SCHAEFER_ANNOT  = 'Schaefer2018_200Parcels_7Networks'   # Schaefer annotation stem
DESTRIEUX_ANNOT = 'aparc.a2009s'                        # Destrieux annotation stem

# ── helpers ──────────────────────────────────────────────────────────────────

def _decode(name):
    # nibabel may return bytes for label names; ensure they are plain strings
    return name.decode() if isinstance(name, bytes) else name


def _read_annot(path):
    # read_annot returns (per-vertex label indices, colortable, label name list)
    labels, _, names = nib.freesurfer.read_annot(str(path))
    names = [_decode(n) for n in names]
    return labels, names  # labels: (n_vertices,) int array; names: list of strings

# ── main mapping ──────────────────────────────────────────────────────────────

rows = []  # will hold one dict per Schaefer parcel

for hemi in ('lh', 'rh'):
    # load per-vertex parcel indices and parcel name list for each atlas
    schaefer_labels, schaefer_names = _read_annot(
        LABEL_DIR / f'{hemi}.{SCHAEFER_ANNOT}.annot'
    )
    destrieux_labels, destrieux_names = _read_annot(
        LABEL_DIR / f'{hemi}.{DESTRIEUX_ANNOT}.annot'
    )

    # pial surface coordinates in RAS ≈ MNI space, used to compute parcel centroids
    coords, _ = nib.freesurfer.read_geometry(
        str(FREESURFER_DIR / SUBJECT / 'surf' / f'{hemi}.pial')
    )

    for p_idx in np.unique(schaefer_labels):  # iterate over each unique parcel index
        if p_idx < 0 or p_idx >= len(schaefer_names):  # skip out-of-range indices
            continue
        schaefer_name = schaefer_names[p_idx]  # full name, e.g. '7Networks_LH_SomMot_1'
        if schaefer_name.lower() in ('unknown', 'corpuscallosum', '???'):  # skip non-cortical labels
            continue

        mask = schaefer_labels == p_idx          # boolean mask: vertices belonging to this parcel
        dest_at_parcel = destrieux_labels[mask]  # Destrieux index at each of those vertices

        # exclude unassigned (-1) and "unknown" (index 0) destrieux vertices
        valid = dest_at_parcel[dest_at_parcel > 0]
        if valid.size == 0:
            print(f"  WARNING: no valid destrieux vertices for {hemi}.{schaefer_name}")
            continue

        counts = np.bincount(valid)               # count occurrences of each Destrieux label
        majority_idx = int(counts.argmax())       # index of the most common Destrieux label
        majority_name = destrieux_names[majority_idx]          # its string name
        match_pct = counts[majority_idx] / valid.size * 100    # % of parcel vertices it covers

        centroid = coords[mask].mean(axis=0)  # mean MNI coordinate of all parcel vertices

        network = schaefer_name.split('_')[2]               # e.g. 'SomMot' from '7Networks_LH_SomMot_1'
        schaefer_short = '_'.join(schaefer_name.split('_')[3:])  # strip prefix: '1' or 'Post_1'

        rows.append({
            'hemi':             hemi,
            'schaefer_parcel':  schaefer_short,   # short parcel label, e.g. '1' or 'Post_1'
            'network':          network,           # parent network name
            'destrieux_label':  majority_name,     # best-matching Destrieux region
            'match_pct':        round(match_pct, 1),
            'mni_x':            round(float(centroid[0]), 1),
            'mni_y':            round(float(centroid[1]), 1),
            'mni_z':            round(float(centroid[2]), 1),
        })

        print(
            f"{hemi}.{schaefer_name:<55s}  →  "
            f"{majority_name:<30s}  ({match_pct:.1f}% of {mask.sum()} vertices)"
        )

# ── save CSV ──────────────────────────────────────────────────────────────────

out_path = FREESURFER_DIR / 'schaefer_to_destrieux.csv'
fieldnames = ['hemi', 'schaefer_parcel', 'network', 'destrieux_label', 'match_pct', 'mni_x', 'mni_y', 'mni_z']

with open(out_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()   # write column names
    writer.writerows(rows) # write one row per parcel

print(f"\nSaved {len(rows)} parcels → {out_path}")
