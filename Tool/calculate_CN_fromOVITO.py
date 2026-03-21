import os
import pandas as pd
from ovito.io import *
from ovito.modifiers import *
import numpy as np
import torch
import glob

def calculate_CN_fromOVITO(sample_path, results_path=False):
    pipeline = import_file(sample_path, sort_particles=True)

    pipeline.modifiers.append(VoronoiAnalysisModifier(
        compute_indices=True,
        edge_threshold=0.1,
    ))

    pipeline_result = pipeline.compute()
    data = pipeline_result.particles


    try:
        df = pd.DataFrame({
            'Coordination': data['Coordination'],
        })
    except KeyError as e:
        raise ValueError(f"Error")
    if results_path:
        df.to_csv(results_path, index=False)

    return df