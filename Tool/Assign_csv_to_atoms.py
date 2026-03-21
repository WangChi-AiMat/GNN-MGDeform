from ovito.io import import_file, export_file
from ovito.data import DataCollection
import os
import numpy as np


def assign_csv_to_atoms(data_path, y_true,y_pred, output_path, index, prop_name_true: str = "y_true",
                        prop_name_pred: str = "y_pred"):

    os.makedirs(output_path, exist_ok=True)

    pipeline = import_file(data_path, sort_particles=True)
    data: DataCollection = pipeline.compute()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_true.ndim == 2 and y_true.shape[1] == 1:
        y_true = y_true.flatten()
    if y_pred.ndim == 2 and y_pred.shape[1] == 1:
        y_pred = y_pred.flatten()

    n_atoms = data.particles.count
    if len(y_true) != n_atoms or len(y_pred) != n_atoms:
        raise ValueError(
            f"Error")

    data.particles_.create_property(name=prop_name_true, data=y_true)
    data.particles_.create_property(name=prop_name_pred, data=y_pred)

    save_path = os.path.join(output_path, f"AfterGNN_graph{index}.dump")

    export_file(data, save_path, "lammps/dump",
                columns=["Particle Identifier", "Particle Type",
                         "Position.X", "Position.Y", "Position.Z", prop_name_true, prop_name_pred],
                restricted_triclinic=True
                )

