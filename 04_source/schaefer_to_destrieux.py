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
LABEL_DIR    = FREESURFER_DIR / SUBJECT / 'label'
SCHAEFER_ANNOT  = 'Schaefer2018_200Parcels_7Networks'
DESTRIEUX_ANNOT = 'aparc.a2009s'

# ── helpers ──────────────────────────────────────────────────────────────────

def _decode(name):
    return name.decode() if isinstance(name, bytes) else name


def _read_annot(path):
    labels, _, names = nib.freesurfer.read_annot(str(path))
    names = [_decode(n) for n in names]
    return labels, names

# ── main mapping ──────────────────────────────────────────────────────────────

rows = []

for hemi in ('lh', 'rh'):
    schaefer_labels, schaefer_names = _read_annot(
        LABEL_DIR / f'{hemi}.{SCHAEFER_ANNOT}.annot'
    )
    destrieux_labels, destrieux_names = _read_annot(
        LABEL_DIR / f'{hemi}.{DESTRIEUX_ANNOT}.annot'
    )

    for p_idx in np.unique(schaefer_labels):
        if p_idx < 0 or p_idx >= len(schaefer_names):
            continue
        schaefer_name = schaefer_names[p_idx]
        if schaefer_name.lower() in ('unknown', 'corpuscallosum', '???'):
            continue

        mask = schaefer_labels == p_idx
        dest_at_parcel = destrieux_labels[mask]

        # exclude unassigned (-1) and "unknown" (index 0) destrieux vertices
        valid = dest_at_parcel[dest_at_parcel > 0]
        if valid.size == 0:
            print(f"  WARNING: no valid destrieux vertices for {hemi}.{schaefer_name}")
            continue

        counts = np.bincount(valid)
        majority_idx = int(counts.argmax())
        majority_name = destrieux_names[majority_idx]
        match_pct = counts[majority_idx] / valid.size * 100

        rows.append({
            'hemi':             hemi,
            'schaefer_parcel':  schaefer_name,
            'destrieux_label':  majority_name,
            'match_pct':        round(match_pct, 1),
            'n_vertices':       int(mask.sum()),
        })

        print(
            f"{hemi}.{schaefer_name:<55s}  →  "
            f"{majority_name:<30s}  ({match_pct:.1f}% of {mask.sum()} vertices)"
        )

# ── save CSV ──────────────────────────────────────────────────────────────────

out_path = FREESURFER_DIR / 'schaefer_to_destrieux.csv'
fieldnames = ['hemi', 'schaefer_parcel', 'destrieux_label', 'match_pct', 'n_vertices']

with open(out_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved {len(rows)} parcels → {out_path}")
