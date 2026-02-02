#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import time
import argparse
from typing import List
from joblib import Parallel, delayed

from ase.io import read  # ASE
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

adaptor = AseAtomsAdaptor()

def atoms_to_primitive_structure(atoms, symprec: float) -> Structure:
    """
    ASE Atoms -> pymatgen Structure -> primitive standard structure
    """
    s: Structure = adaptor.get_structure(atoms)
    sga = SpacegroupAnalyzer(s, symprec=symprec)
    prim = sga.get_primitive_standard_structure()
    return prim

def _process_one(atoms, out_dir: str, idx: int, symprec: float, basename: str):
    try:
        prim = atoms_to_primitive_structure(atoms, symprec)
        fname = f"primitive_{basename}_{idx}.cif" if basename else f"primitive_{idx}.cif"
        prim.to(filename=os.path.join(out_dir, fname), fmt="cif")
    except Exception as e:
        print(f"[!] Failed index {idx} in '{basename}': {e}")

def convert_extxyz_to_primitives(extxyz_path: str, out_dir: str, symprec: float, n_jobs: int):
    os.makedirs(out_dir, exist_ok=True)
    basename = os.path.basename(os.path.dirname(extxyz_path)) 
    try:
        atoms_list = read(extxyz_path, index=":")  
    except Exception as e:
        print(f"[!] Cannot read {extxyz_path}: {e}")
        return

    n = len(atoms_list)
    print(f"  - {basename}: {n} structures found in '{os.path.basename(extxyz_path)}'")

    Parallel(n_jobs=n_jobs)(
        delayed(_process_one)(atoms_list[i], out_dir, i + 1, symprec, basename)
        for i in range(n)
    )

def find_extxyz_files(input_roots: List[str], filename: str) -> List[str]:
    found = []
    for root in input_roots:
        for folder in glob.glob(root):
            candidate = os.path.join(folder, filename)
            if os.path.isfile(candidate):
                found.append(os.path.abspath(candidate))
    return found

def main():
    parser = argparse.ArgumentParser(
        description="Convert generated_crystals.extxyz (multi-structure) to primitive CIFs (many files)."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
    )
    parser.add_argument(
        "-f", "--filename",
        default="generated_crystals.extxyz",
    )
    parser.add_argument(
        "-o", "--out",
        default=None,
    )
    parser.add_argument(
        "--symprec",
        type=float,
        default=5e-2,
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=-1,
    )
    args = parser.parse_args()

    t0 = time.time()
    extxyz_files = find_extxyz_files(args.inputs, args.filename)
    if not extxyz_files:
        print("[!] No generated_crystals.extxyz found under given inputs.")
        return

    print(f"[✓] Found {len(extxyz_files)} extxyz file(s).")
    for extxyz in extxyz_files:
        parent = os.path.basename(os.path.dirname(extxyz))
        out_dir = args.out if args.out else f"{parent}"
        print(f"→ Converting '{extxyz}' → primitives in '{out_dir}' (symprec={args.symprec})")
        convert_extxyz_to_primitives(extxyz, out_dir, args.symprec, args.jobs)

    print(f"\n[Done] Total time: {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()

