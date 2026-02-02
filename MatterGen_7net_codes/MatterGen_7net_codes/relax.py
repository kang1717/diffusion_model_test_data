#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, glob, gc
from collections import defaultdict
from typing import Optional

from ase.io import read, write
from ase.constraints import ExpCellFilter
from ase.optimize import LBFGS
from ase.calculators.singlepoint import SinglePointCalculator  # ★ 추가
from sevenn.calculator import SevenNetCalculator

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Composition
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.ext.matproj import MPRester

# ---- (선택) PyTorch 메모리 조각화 완화 ----
HAS_TORCH = False
try:
    import torch
    HAS_TORCH = True
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
except Exception:
    pass

def free_cuda_memory():
    """파이썬 참조 정리 + PyTorch CUDA 캐시 비우기."""
    gc.collect()
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()

# ------------------ 사용자 설정 ------------------
CALC_MODEL = "7net-mf-ompa"
CALC_MODAL = "mpa"
FMAX = 0.02            # eV/Å
MAX_STEPS = 200
MP_API_KEY = "qsk5Q2eACLEDaeLAaMdilGoRL6mLT2Tg"   # MP API 키

INPUT_PATTERN = "unique_*.cif"
RELAX_LOG = "relax.log"
LOG = "final_hull_log.txt"

WINDOW_EV_PER_ATOM = 0.03  # 상대 윈도우: Ehull ≤ (min_Ehull + window) 만 유지
# -------------------------------------------------

adaptor = AseAtomsAdaptor()
calc = SevenNetCalculator(CALC_MODEL, modal=CALC_MODAL)
compat = MaterialsProject2020Compatibility()

# MP PhaseDiagram 캐시 (보정된 엔트리)
mp_cache = {}

def iter_sorted_indices(pattern: str):
    cif_files = glob.glob(pattern)
    return sorted(
        (int(m.group(1)), f)
        for f in cif_files
        if (m := re.match(r"unique_(\d+)\.cif", os.path.basename(f)))
    )

def relax_atoms(infile: str):
    """
    구조 relax → SevenNet으로 총에너지 평가 → 무거운 calc 제거,
    에너지만 담긴 SinglePointCalculator 부착 → CONTCAR{i} 저장.
    반환: atoms(에너지 포함), idx, Etot
    """
    free_cuda_memory()  # 계산 직전 캐시 정리

    atoms = read(infile, format="cif")
    atoms.calc = calc
    ecf = ExpCellFilter(atoms)
    opt = LBFGS(ecf, logfile=RELAX_LOG)
    opt.run(fmax=FMAX, steps=MAX_STEPS)

    # 총에너지는 무거운 calc가 붙어 있을 때 한 번만 계산
    Etot = atoms.get_potential_energy()

    # 무거운 SevenNet calc 제거 + 가벼운 SinglePointCalculator로 교체 (에너지 보존)
    atoms.calc = SinglePointCalculator(atoms, energy=Etot)

    # 저장(중간 산물)
    idx = int(re.match(r"unique_(\d+)\.cif", os.path.basename(infile)).group(1))
    write(f"CONTCAR{idx}", atoms, format="vasp")

    # 메모리 정리
    free_cuda_memory()
    return atoms, idx, Etot

def build_corrected_entry_from_atoms(atoms, total_e: float) -> Optional[ComputedStructureEntry]:
    """ASE Atoms → Structure → MP 메타(+U/POTCAR) → ComputedStructureEntry → MP2020 보정
       (총에너지는 호출자에서 전달받아 사용)
    """
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

    entry_raw = ComputedStructureEntry(structure=struct, energy=float(total_e), parameters=params)
    return compat.process_entry(entry_raw)  # None이면 보정 규칙에서 탈락

def get_pd_for_chemsys(chemsys: str, my_entries_for_this_chemsys):
    """MP DB(compatible_only=False → 직접 보정) + 내 엔트리로 PD 생성 (캐시 사용)"""
    if chemsys not in mp_cache:
        with MPRester(MP_API_KEY) as mpr:
            raw = mpr.get_entries_in_chemsys(chemsys.split("-"), compatible_only=False)
        mp_cache[chemsys] = compat.process_entries(raw)
    base = mp_cache[chemsys]
    return PhaseDiagram(base + my_entries_for_this_chemsys)

def main():
    # 0) 대상 목록
    pairs = iter_sorted_indices(INPUT_PATTERN)
    if not pairs:
        print(f"No inputs matching {INPUT_PATTERN}")
        return

    # 1) Relax → 내 보정 엔트리 만들기
    #    chemsys -> [(idx, atoms, entry_corr, formula, nat, Etot, Epa)]
    my_entries_by_chemsys = defaultdict(list)
    with open(LOG, "w") as log:
        log.write("idx\tformula\tchemsys\tnatoms\tE_tot(eV)\tE/atom(eV)\tEhull(eV/atom)\tkeep\tinfo\n")

        for idx, infile in pairs:
            atoms = None
            try:
                # 구조 relax (+ SinglePointCalculator로 교체)
                atoms, _, Etot = relax_atoms(infile)  # CONTCAR{idx} 저장됨

                # 보정 엔트리 생성 (총에너지는 인자로 전달)
                entry_corr = build_corrected_entry_from_atoms(atoms, Etot)

                struct = adaptor.get_structure(atoms)
                formula = struct.composition.reduced_formula
                chemsys = Composition(formula).chemical_system
                nat = len(atoms)
                Epa = Etot / nat

                if entry_corr is None:
                    log.write(f"{idx}\t{formula}\t{chemsys}\t{nat}\t{Etot:.6f}\t{Epa:.6f}\tNA\tFalse\tMP2020_filtered\n")
                    print(f"[{idx}] {formula} → filtered by MP2020 (skipped)")
                else:
                    my_entries_by_chemsys[chemsys].append((idx, atoms, entry_corr, formula, nat, Etot, Epa))

            except Exception as e:
                print(f"[{idx}] relax/entry failed: {e}")
            finally:
                # 이 구조 처리 끝: 캐시 정리 (atoms는 저장/선별에서 사용되므로 여기선 유지)
                free_cuda_memory()

    # 2) 화학계별 PD 구성 → Ehull 1pass 계산(저장)
    #    chemsys -> [{'idx','atoms','natoms','Etot','Epa','ehull','form'}]
    results_by_chemsys = {}
    for chemsys, rows in my_entries_by_chemsys.items():
        if not rows:
            continue

        free_cuda_memory()
        pd = get_pd_for_chemsys(chemsys, [r[2] for r in rows])  # entry_corr 리스트

        res = []
        for (idx, atoms_saved, entry_corr, formula, nat, Etot, Epa) in rows:
            ehull = pd.get_e_above_hull(entry_corr, allow_negative=True)
            res.append({"idx": idx, "atoms": atoms_saved, "natoms": nat,
                        "Etot": Etot, "Epa": Epa, "ehull": ehull,
                        "form": formula, "chemsys": chemsys})
        results_by_chemsys[chemsys] = res

        free_cuda_memory()

    # 3) chemsys별 min Ehull → min+window 이하만 KEEP (2pass)
    kept = 0
    with open(LOG, "a") as log:
        for chemsys, res in results_by_chemsys.items():
            if not res:
                continue
            min_eh = min(r["ehull"] for r in res)
            cutoff = min_eh + WINDOW_EV_PER_ATOM

            for r in res:
                keep = (r["ehull"] <= cutoff)
                if keep:
                    write(f"CONTCARf_{r['idx']}", r["atoms"], format="vasp")
                    kept += 1

                # 구조 참조 해제 (저장 여부와 무관)
                try:
                    del r["atoms"]
                except Exception:
                    pass
                free_cuda_memory()

                log.write(f"{r['idx']}\t{r['form']}\t{chemsys}\t{r['natoms']}\t"
                          f"{r['Etot']:.6f}\t{r['Epa']:.6f}\t{r['ehull']:.6f}\t{keep}\t"
                          f"min={min_eh:.6f}, cutoff={cutoff:.6f}\n")

                print(f"[{r['idx']}] {r['form']:12s} | Ehull = {r['ehull']:.6f} eV/atom "
                      f"(min={min_eh:.6f}, cutoff={cutoff:.6f}){' <- KEEP' if keep else ''}")

    print(f"\nDone. Kept {kept} structures with Ehull ≤ min(Ehull)+{WINDOW_EV_PER_ATOM:.3f} eV/atom.")
    print(f"Log written to: {LOG}")

if __name__ == "__main__":
    main()

