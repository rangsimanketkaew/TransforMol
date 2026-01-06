"""
Generate dataset for neural network training from AQM HDF5 files

Usage: python py1_generate_dataset.py \
    --gas AQM-gas.hdf5 \
    --sol AQM-sol.hdf5 \
    [--n-samples 100] \
    [--output-csv molecule_dataset.csv]

Updates:
    02.11.2025 Initial script [Rangsiman Ketkaew]
"""

import sys
import argparse
import numpy as np
import pandas as pd
import h5py

from natsort import natsorted
from pathlib import Path


def extract_molecule_data(gas_file, sol_file, max_molecules, output_csv):
    """
    Extract molecular geometries from gas and solvent phase HDF5 files

    args:
        gas_file: Path to HDF5 file containing gas phase geometries
        sol_file: Path to HDF5 file containing solvent phase geometries
        max_molecules: Maximum number of molecules to process (int, -1 for all molecules)
        output_csv: Path to output CSV file

    Returns:
        df: DataFrame containing the extracted data
    """

    gas_h5 = h5py.File(gas_file, "r")
    sol_h5 = h5py.File(sol_file, "r")

    gas_mol_ids = set(gas_h5.keys())
    sol_mol_ids = set(sol_h5.keys())
    common_mol_ids = gas_mol_ids.intersection(sol_mol_ids)

    print(f"Found {len(gas_mol_ids)} molecules in gas phase")
    print(f"Found {len(sol_mol_ids)} molecules in solvent phase")
    print(f"Found {len(common_mol_ids)} molecules in both phases\n")

    if max_molecules != -1:
        common_mol_ids = list(common_mol_ids)[:max_molecules]
        print(f"Limiting to {max_molecules} molecules for processing\n")

    data_records = []
    processed = 0
    skipped = 0

    common_mol_ids = list(natsorted(common_mol_ids))

    # for water
    dielectric_const = 78.4

    data_records = []
    data_records_append = data_records.append

    # Loop over molecules
    for i, mol_id in enumerate(common_mol_ids):
        print(
            f"\rProcessing molecule: {i+1} / {len(common_mol_ids)}", end="", flush=True
        )

        # Get gas and solvent groups per molecule
        try:
            gas_mol_group = gas_h5[mol_id]
            sol_mol_group = sol_h5[mol_id]
            gas_conf_ids = list(gas_mol_group.keys())
            sol_conf_ids = list(sol_mol_group.keys())
        except Exception as e:
            print(f"\nError accessing molecule {mol_id}: {e}")
            skipped += 1
            continue

        if gas_conf_ids != sol_conf_ids:
            print(f"\nMismatch in conformation IDs for {mol_id}")
            skipped += len(gas_conf_ids)
            continue

        # Loop over configurations
        for gas_conf in gas_conf_ids:
            try:
                gas_group = gas_mol_group[gas_conf]
                sol_group = sol_mol_group[gas_conf]
            except KeyError:
                skipped += 1
                continue

            try:
                gas_xyz = np.array(gas_group["atXYZ"])
                gas_atomic_nums = np.array(gas_group["atNUM"])
                sol_xyz = np.array(sol_group["atXYZ"])
                sol_atomic_nums = np.array(sol_group["atNUM"])

                n_atoms = len(gas_atomic_nums)
                if n_atoms != len(sol_atomic_nums):
                    skipped += 1
                    continue

                if not np.array_equal(gas_atomic_nums, sol_atomic_nums):
                    skipped += 1
                    continue

                gas_energy = float(gas_group["ePBE0+MBD"][0])
                sol_energy = float(sol_group["ePBE0+MBD"][0])

                gas_hlgap = float(gas_group["HLgap"][0])
                sol_hlgap = float(sol_group["HLgap"][0])

                gas_dipole = float(gas_group["DIP"][0])
                sol_dipole = float(sol_group["DIP"][0])

                data_records_append(
                    {
                        "molecule_id": mol_id,
                        "gas_conformation": gas_conf,
                        "sol_conformation": gas_conf,
                        "num_atoms": n_atoms,
                        "atomic_numbers": gas_atomic_nums.tolist(),
                        "gas_xyz": gas_xyz.flatten().tolist(),
                        "sol_xyz": sol_xyz.flatten().tolist(),
                        "dielectric_const": dielectric_const,
                        "gas_energy": gas_energy,
                        "sol_energy": sol_energy,
                        "solvation_energy": sol_energy - gas_energy,
                        "gas_hlgap": gas_hlgap,
                        "sol_hlgap": sol_hlgap,
                        "gas_dipole": gas_dipole,
                        "sol_dipole": sol_dipole,
                    }
                )
                processed += 1

            except Exception as e:
                print(f"\nError processing {mol_id}/{gas_conf}: {e}")
                skipped += 1

    gas_h5.close()
    sol_h5.close()

    df = pd.DataFrame(data_records)
    df.to_csv(output_csv, index=False)

    print(f"\nNumber of unique molecules: {df['molecule_id'].nunique()}")
    print(f"Total processed: {processed}")
    print(f"Total skipped: {skipped}")

    return df


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="Generate dataset from AQM HDF5 files")
    parser.add_argument("--gas", default="AQM-gas.hdf5", help="Gas phase HDF5 file")
    parser.add_argument("--sol", default="AQM-sol.hdf5", help="Solvent phase HDF5 file")
    parser.add_argument("--n-samples", type=int, default=-1, help="Number of samples for sample dataset (optional)")
    parser.add_argument("--output-csv", default="molecule_dataset.csv", help="Output CSV file")
    args = parser.parse_args()
    # fmt: on

    if not Path(args.gas).exists():
        print(f"Error: Gas phase file not found: {args.gas}")
        sys.exit(1)

    if not Path(args.sol).exists():
        print(f"Error: Solvent phase file not found: {args.sol}")
        sys.exit(1)

    df = extract_molecule_data(args.gas, args.sol, args.n_samples, args.output_csv)

    print("\nComplete!")
    print(f"Dataset saved to: {args.output_csv}")
