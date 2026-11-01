"""Anatomical feature groups for the CC200 atlas.

Each CC200 ROI is assigned to a lobe using its maximum-overlap AAL label (from the
PCP CC200 label file). An edge belongs to a group if at least one endpoint lies in
the group ("incident" definition), which measures the connectivity profile of the
group with the rest of the brain."""
import re
import numpy as np
import pandas as pd

LOBES = {
    "Frontal": r"^(Frontal|Precentral|Supp_Motor|Olfactory|Rectus|Paracentral)",
    "Temporal": r"^(Temporal|Heschl|Fusiform)",
    "Parietal": r"^(Parietal|Postcentral|SupraMarginal|Angular|Precuneus)",
    "Occipital": r"^(Occipital|Calcarine|Cuneus|Lingual)",
    "Limbic": r"^(Cingulum|Hippocampus|ParaHippocampal|Amygdala|Insula)",
    "Subcortical": r"^(Caudate|Putamen|Pallidum|Thalamus)",
    "Cerebellum": r"^(Cerebelum|Vermis)",
}


def roi_lobes(label_csv):
    df = pd.read_csv(label_csv, skipinitialspace=True)
    col = [c for c in df.columns if c.strip() == "AAL"][0]
    lobes = []
    for s in df[col].astype(str):
        m = re.findall(r'\["([^"]+)":\s*([0-9.]+)\]', s)
        names = [n for n, _ in m if n != "None"]
        lab = names[0] if names else "None"
        lobe = next((L for L, pat in LOBES.items() if re.match(pat, lab)), "Unassigned")
        lobes.append(lobe)
    return np.array(lobes)


def edge_masks(lobes):
    R = len(lobes); i, j = np.triu_indices(R, 1)
    masks = {L: (lobes[i] == L) | (lobes[j] == L) for L in LOBES}
    masks["Whole-brain"] = np.ones(len(i), bool)
    return masks
