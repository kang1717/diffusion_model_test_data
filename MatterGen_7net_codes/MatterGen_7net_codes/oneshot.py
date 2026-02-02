#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, glob, argparse, gc
from collections import defaultdict
from typing import Optional

from ase.io import read, write
from sevenn.calculator import SevenNetCalculator

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Composition
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from pymatgen.ext.matproj import MPRester

CALC_MODEL = '7net-mf-ompa'
CALC_MODAL = 'mpa'
THRESHOLD_EV_PER_ATOM = 0.06   
LOG_FILENAME = "energy_log_hull.txt"

API_KEY = ""  #### Materials Project key
CIF_GLOB = "primitive_*.cif"
CPU_FALLBACK_N_ATOMS = 600     

HAS_TORCH = False
try:
    import torch
    HAS_TORCH = True
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
except Exception:
    pass

adaptor = AseAtomsAdaptor()
compat = MaterialsProject2020Compatibility()
mp_cache = {}  

def free_cuda_memory():
    gc.collect()
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()

def sorted_cif_list(pattern: str):
    def keyfunc(f):
        m = re.match(r".*?(\d+)\.cif$", f)
        return int(m.group(1)) if m else 10**9
    return sorted(glob.glob(pattern), key=keyfunc)

def safe_get_energy(atoms, calc_gpu, use_cpu_fallback=True, cpu_threshold=CPU_FALLBACK_N_ATOMS):
    free_cuda_memory()

    if cpu_threshold is not None and len(atoms) >= cpu_threshold:
        calc_cpu = SevenNetCalculator(CALC_MODEL, modal=CALC_MODAL)  # CPU
        atoms.calc = calc_cpu
        E = atoms.get_potential_energy()
        atoms.calc = None
        free_cuda_memory()
        return E

    atoms.calc = calc_gpu
    try:
        E = atoms.get_potential_energy()
    except RuntimeError as e:
        if 'CUDA out of memory' in str(e) and use_cpu_fallback:
            print("[WARN] CUDA OOM → fall back to CPU for this structure")
            free_cuda_memory()
            calc_cpu = SevenNetCalculator(CALC_MODEL, modal=CALC_MODAL)
            atoms.calc = calc_cpu
            E = atoms.get_potential_energy()
        else:
            atoms.calc = None
            free_cuda_memory()
            raise
    atoms.calc = None
    free_cuda_memory()
    return E

def build_my_corrected_entry_from_atoms(atoms, total_energy_eV) -> Optional[ComputedStructureEntry]:
    struct: Structure = adaptor.get_structure(atoms)

    mpset = MPRelaxSet(struct)
    incar = mpset.incar
    ldauu = incar.get("LDAUU", [])
    elems = [sp.symbol for sp in struct.composition.elements]
    hubbards = {el: float(U) for el, U in zip(elems, ldauu) if float(U) > 0}
    is_hubbard = bool(hubbards)
    run_type = "GGA+U" if is_hubbard else "GGA"
    potcar_symbols = [f"PBE {lbl}" for lbl in mpset.potcar_symbols]

    params = {
        "run_type": run_type,
        "is_hubbard": is_hubbard,
        "hubbards": hubbards,
        "potcar_symbols": potcar_symbols,
        "pseudo_potential": {
            "functional": "PBE",
            "labels": [s.split()[1] for s in potcar_symbols],
            "pot_type": "paw",
        },
    }
    entry_raw = ComputedStructureEntry(structure=struct, energy=float(total_energy_eV), parameters=params)
    return compat.process_entry(entry_raw) 

# ---------- MP PD ----------
def get_pd_for_chemsys(chemsys: str, my_entries_for_this_chemsys):
    if chemsys not in mp_cache:
        with MPRester(API_KEY) as mpr:
            raw = mpr.get_entries_in_chemsys(chemsys.split("-"), compatible_only=False)
        mp_cache[chemsys] = compat.process_entries(raw)
    base = mp_cache[chemsys]
    return PhaseDiagram(base + my_entries_for_this_chemsys)

def elements_set_from_formula(formula: str):
    return {el.symbol for el in Composition(formula).elements}

def main():
    parser = argparse.ArgumentParser(
        description="Hull-based selection with MP2020 corrections, exact chemsys match, relative window to min(Ehull)."
    )
    parser.add_argument("--target_formula", required=True,
                        help="ex: La2PtI2 → chemesys: La-Pt-I")
    parser.add_argument("--threshold", type=float, default=THRESHOLD_EV_PER_ATOM,
                        help="(relative) min(Ehull)+threshold window (eV/atom)")
    parser.add_argument("--pattern", default=CIF_GLOB,
                        help="input CIF pattern (default: primitive_*.cif)")
    parser.add_argument("--log", default=LOG_FILENAME,
                        help="log file name (default: energy_log_hull.txt)")
    args = parser.parse_args()

    target_elems = elements_set_from_formula(args.target_formula)
    target_chemsys = "-".join(sorted(target_elems))

    cif_files = sorted_cif_list(args.pattern)
    if not cif_files:
        print(f"No CIF files matching '{args.pattern}'.")
        return

    with open(args.log, "w") as log:
        log.write("idx\tname\ttarget_chemsys\tcand_chemsys\tformula\tn_atoms\tE_tot(eV)\tE/atom(eV)\tEhull(eV/atom)\tselected\treason\n")

    calc_gpu = SevenNetCalculator(CALC_MODEL, modal=CALC_MODAL)

    my_entries_by_chemsys = defaultdict(list)  # chemsys -> [(idx,fname,natoms,Etot,Epa,entry_like,rf,reason)]
    for fname in cif_files:
        m = re.match(r".*?(\d+)\.cif$", fname)
        idx = int(m.group(1)) if m else -1

        free_cuda_memory()

        atoms = read(fname, format="cif")
        struct = adaptor.get_structure(atoms)
        cand_elems = {sp.symbol for sp in struct.composition.elements}
        cand_chemsys = "-".join(sorted(cand_elems))

        if cand_elems != target_elems:
            with open(args.log, "a") as log:
                log.write(f"{idx}\t{os.path.basename(fname)}\t{target_chemsys}\t{cand_chemsys}\t"
                          f"{struct.composition.reduced_formula}\t{len(struct)}\tNA\tNA\tNA\tFalse\tnon-matching chemsys\n")
            print(f"{os.path.basename(fname):20s} -> SKIP (chemsys {cand_chemsys} != target {target_chemsys})")
            free_cuda_memory()
            continue

        Etot = safe_get_energy(atoms, calc_gpu, use_cpu_fallback=True)
        natoms = len(atoms)
        Epa = Etot / natoms

        my_corr = build_my_corrected_entry_from_atoms(atoms, Etot)
        if my_corr is None:
            entry_like = PDEntry(Composition(struct.formula), float(Etot))
            reason = "filtered_by_MP2020->PDEntry"
        else:
            entry_like = my_corr
            reason = ""

        my_entries_by_chemsys[cand_chemsys].append(
            (idx, fname, natoms, Etot, Epa, entry_like, entry_like.composition.reduced_formula if hasattr(entry_like, "composition") else struct.composition.reduced_formula, reason)
        )

        free_cuda_memory()

    results_by_chemsys = {}
    for chemsys, rows in my_entries_by_chemsys.items():
        if not rows:
            continue
        pd = get_pd_for_chemsys(chemsys, [t[5] for t in rows])  
        res = []
        for (idx, fname, natoms, Etot, Epa, entry_like, rf, reason) in rows:
            ehull = pd.get_e_above_hull(entry_like, allow_negative=True)
            res.append({"idx": idx, "fname": fname, "natoms": natoms, "Etot": Etot,
                        "Epa": Epa, "ehull": ehull, "form": rf, "chemsys": chemsys,
                        "reason": reason})
        results_by_chemsys[chemsys] = res

        free_cuda_memory()

    selected_count = 0
    for chemsys, res in results_by_chemsys.items():
        if not res:
            continue
        print(f"== {chemsys}: {len(res)} candidates")
        min_eh = min(r["ehull"] for r in res)
        cutoff = min_eh + args.threshold

        kept_in_cs = 0
        for r in res:
            keep = (r["ehull"] <= cutoff)
            if keep:
                atoms_out = read(r["fname"], format="cif")
                try:
                    write(f"POSCARm_{r['idx']}", atoms_out, format="vasp", direct=True, vasp5=True)
                finally:
                    del atoms_out
                    free_cuda_memory()
                selected_count += 1
                kept_in_cs += 1

            with open(args.log, "a") as log:
                log.write(f"{r['idx']}\t{os.path.basename(r['fname'])}\t{target_chemsys}\t{chemsys}\t"
                          f"{r['form']}\t{r['natoms']}\t{r['Etot']:.6f}\t{r['Epa']:.6f}\t"
                          f"{r['ehull']:.6f}\t{keep}\tmin+window(cutoff={cutoff:.6f}) {r['reason']}\n")

            print(f"{os.path.basename(r['fname']):20s} -> Ehull = {r['ehull']:.6f} eV/atom "
                  f"(min={min_eh:.6f}, cutoff={cutoff:.6f}){' [KEPT]' if keep else ''}")

        print(f"== {chemsys}: kept {kept_in_cs} / {len(res)} (min={min_eh:.6f}, cutoff={cutoff:.6f})")

        results_by_chemsys[chemsys] = None
        free_cuda_memory()

    print(f"\nDone. Selection by relative window to min Ehull per chemsys.")
    print(f"- Threshold window: +{args.threshold:.3f} eV/atom from min(Ehull)")
    print(f"- Log: {args.log}")
    print(f"- Saved POSCARm_i for {selected_count} structures within window.")

if __name__ == "__main__":
    main()

