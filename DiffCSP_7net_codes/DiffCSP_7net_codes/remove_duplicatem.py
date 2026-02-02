import os
import re
import glob
import time
import argparse
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher
from joblib import Parallel, delayed

def multiply_formula(formula, n):
    matches = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
    expanded = []
    for el, count in matches:
        cnt = int(count) if count else 1
        total = cnt * n
        expanded.append(f"{el}{total if total != 1 else ''}")
    return ''.join(expanded)

def get_primitive(structure, symprec):
    sga = SpacegroupAnalyzer(structure, symprec=symprec)
    return sga.get_primitive_standard_structure()

def _process_cif_to_primitive(cif_path, out_folder, idx, symprec):
    try:
        struct = CifParser(cif_path).get_structures(primitive=False)[0]
        prim = get_primitive(struct, symprec)
        prim.to(filename=os.path.join(out_folder, f"primitive_{idx}.cif"))
    except Exception as e:
        print(f"[!] Failed {cif_path}: {e}")

def collect_and_convert_to_primitive_parallel(matter, n, out_folder, symprec, n_jobs):
    os.makedirs(out_folder, exist_ok=True)
    all_files = []
    for i in range(1, n+1):
        scaled = multiply_formula(matter, i)
        all_files += glob.glob(f"{scaled}/{scaled}_*.cif")
    all_files.sort()
    print(f"Found {len(all_files)} CIFs to convert.")
    Parallel(n_jobs=n_jobs)(
        delayed(_process_cif_to_primitive)(cif, out_folder, idx, symprec)
        for idx, cif in enumerate(all_files, 1)
    )

def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

def dedup_list(structs, matcher, volume_tol, lattice_tol):
    unique = []
    for s in structs:
        dup = False
        a1, b1, c1 = s.lattice.abc
        for u in unique:
            # 1) volume filter
            vol_diff = abs(s.volume - u.volume) / max(s.volume, u.volume)
            if vol_diff > volume_tol:
                continue
            # 2) lattice-parameter filter
            a2, b2, c2 = u.lattice.abc
            if (abs(a1 - a2) / max(a1, a2) > lattice_tol or
                abs(b1 - b2) / max(b1, b2) > lattice_tol or
                abs(c1 - c2) / max(c1, c2) > lattice_tol):
                continue
            # 3) full structure match
            if matcher.fit(s, u):
                dup = True
                break
        if not dup:
            unique.append(s)
    return unique

def remove_duplicates_parallel(input_folder, matcher_kwargs,
                               volume_tol, lattice_tol,
                               chunk_size, n_jobs, output_prefix):
    # 1) Load primitives
    prim_paths = sorted(glob.glob(f"{input_folder}/POSCARm_*"))
    structs = []
    for p in prim_paths:
        try:
            s = Structure.from_file(p).get_sorted_structure()
            s.remove_oxidation_states()
            structs.append(s)
        except Exception:
            pass

    matcher = StructureMatcher(**matcher_kwargs)
    chunks = list(chunked(structs, chunk_size))
    print(f"Total: {len(structs)} structs → {len(chunks)} chunks")

    # 2) Chunked parallel dedup
    survivors_per_chunk = Parallel(n_jobs=n_jobs)(
        delayed(dedup_list)(chunk, matcher, volume_tol, lattice_tol)
        for chunk in chunks
    )
    # 3) Mid-output
    for ci, (chunk, surv) in enumerate(zip(chunks, survivors_per_chunk), 1):
        print(f" Chunk {ci}: {len(chunk)} → {len(surv)} survivors")

    # 4) Flatten survivors and final dedup
    survivors = [s for grp in survivors_per_chunk for s in grp]
    print(f" After chunked: {len(structs)} → {len(survivors)} total survivors")
    final = dedup_list(survivors, matcher, volume_tol, lattice_tol)
    print(f" Final: {len(survivors)} → {len(final)} unique")

    # 5) Save
    for idx, u in enumerate(final, 1):
        u.to(filename=os.path.join(input_folder, f"{output_prefix}{idx}.cif"), fmt="cif")
    print(f"[✓] Saved {len(final)} unique structures.")

def main():
    t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("matter", help="Base formula, e.g. LiCdP")
    parser.add_argument("-n", type=int, default=4, help="Max expansion multiple")
    parser.add_argument("--symprec", type=float, default=5e-2, help="Symprec for primitive")
    parser.add_argument("--volume_tol", type=float, default=0.15, help="Volume diff tol")
    parser.add_argument("--lattice_tol", type=float, default=0.05, help="Lattice abc diff tol")
    parser.add_argument("--chunk_size", type=int, default=100, help="Chunk size")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Parallel jobs")
    args = parser.parse_args()

    out_folder = f"{args.matter}"

    matcher_kwargs = {
        "ltol": 0.5,
        "stol": 0.9,
        "angle_tol": 30,
        "primitive_cell": False,
        "scale": False,
        "attempt_supercell": False
    }
    print("Removing duplicates (chunked + lattice screening)...")
    remove_duplicates_parallel(
        out_folder, matcher_kwargs,
        volume_tol=args.volume_tol,
        lattice_tol=args.lattice_tol,
        chunk_size=args.chunk_size,
        n_jobs=args.n_jobs,
        output_prefix="unique_"
    )
    print(f"Total time: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()

