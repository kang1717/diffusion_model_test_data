import glob
import re
from ase.io import read, write
from sevenn.calculator import SevenNetCalculator

CALC_MODEL = '7net-mf-ompa'
CALC_MODAL = 'mpa'
THRESHOLD_EV_PER_ATOM = 0.06  # 100 meV per atom
LOG_FILENAME = "energy_log.txt"

calc = SevenNetCalculator(CALC_MODEL, modal=CALC_MODAL)

cif_files = sorted(
    glob.glob('primitive_*.cif'),
    key=lambda f: int(re.match(r'primitive_(\d+)\.cif', f).group(1))
)

energies = []  # list of (idx, total_e, per_atom_e, natoms)
with open(LOG_FILENAME, "w") as log:
    log.write("idx\ttotal_energy(eV)\tn_atoms\tenergy_per_atom(eV)\n")
    for fname in cif_files:
        idx = int(re.match(r'primitive_(\d+)\.cif', fname).group(1))
        atoms = read(fname, format="cif")
        atoms.calc = calc
        total_e = atoms.get_potential_energy()
        natoms = len(atoms)
        e_per_atom = total_e / natoms
        energies.append((idx, total_e, natoms, e_per_atom))
        log.write(f"{idx}\t{total_e:.6f}\t{natoms}\t{e_per_atom:.6f}\n")

min_per_atom = min(e[3] for e in energies)

for idx, total_e, natoms, e_per_atom in energies:
    if e_per_atom <= min_per_atom + THRESHOLD_EV_PER_ATOM:
        atoms = read(f'primitive_{idx}.cif', format="cif")
        atoms.calc = calc  # not strictly needed for write
        write(f"POSCARm_{idx}", atoms, format="vasp")

print(f"Done. Energies logged in '{LOG_FILENAME}'.")
print(f"Structures within {THRESHOLD_EV_PER_ATOM:.3f} eV/atom of min saved as POSCARm_i.")

