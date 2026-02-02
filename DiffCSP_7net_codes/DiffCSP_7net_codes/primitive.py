import os
import glob
import time
import argparse
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from joblib import Parallel, delayed


def get_primitive(structure, symprec):
    sga = SpacegroupAnalyzer(structure, symprec=symprec)
    return sga.get_primitive_standard_structure()


def _process_cif_to_primitive(cif_path, out_folder, idx, symprec):
    try:
        struct = CifParser(cif_path).get_structures(primitive=False)[0]
        prim = get_primitive(struct, symprec)
        prim.to(filename=os.path.join(out_folder, f"primitive_{idx}.cif"))
        print(f"[✓] {cif_path} → primitive_{idx}.cif")
    except Exception as e:
        print(f"[!] Failed {cif_path}: {e}")


def convert_folder_to_primitive(folder, symprec, n_jobs):
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"Folder not found: {folder}")

    cif_paths = sorted(glob.glob(os.path.join(folder, "*.cif")))
    print(f"Found {len(cif_paths)} CIFs in '{folder}'")

    if not cif_paths:
        return

    out_folder = folder

    Parallel(n_jobs=n_jobs)(
        delayed(_process_cif_to_primitive)(cif, out_folder, idx, symprec)
        for idx, cif in enumerate(cif_paths, 1)
    )

    print(f"[✓] Done. Generated primitive CIFs in '{out_folder}'.")


def main():
    t0 = time.time()
    parser = argparse.ArgumentParser(
        description="Convert all CIFs in a folder to primitive structures."
    )
    parser.add_argument(
        "folder",
        help="Folder containing CIF files (e.g. LiCdP_1/)"
    )
    parser.add_argument(
        "--symprec", type=float, default=5e-2,
        help="Symmetry precision for primitive finding (default: 0.05)"
    )
    parser.add_argument(
        "--n_jobs", type=int, default=-1,
        help="Number of parallel jobs for conversion (default: -1, use all cores)"
    )
    args = parser.parse_args()

    print(f"Converting CIFs in folder '{args.folder}' → primitive structures")
    convert_folder_to_primitive(
        folder=args.folder,
        symprec=args.symprec,
        n_jobs=args.n_jobs
    )
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

