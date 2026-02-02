from ase.io import read, write
from sevenn.calculator import SevenNetCalculator
from ase.constraints import ExpCellFilter
from ase.optimize import LBFGS
import glob, re

calc = SevenNetCalculator('7net-mf-ompa', modal='mpa')

cif_files = glob.glob('unique_*.cif')
indices = sorted(
    int(m.group(1))
    for f in cif_files
    if (m := re.match(r'unique_(\d+)\.cif', f))
)

energies = {}
threshold = 0.03  # eV per atom (30 meV)

with open('final_log', 'w') as log_file:
    for i in indices:
        atoms = read(f'unique_{i}.cif', format='cif')
        atoms.calc = calc

        cell_filter = ExpCellFilter(atoms)
        opt = LBFGS(cell_filter, logfile='relax.log')
        opt.run(fmax=0.02, steps=200)

        total_energy = atoms.get_potential_energy()
        epa = total_energy / len(atoms)  # per‑atom energy
        energies[i] = epa

        log_file.write(
            f"[#{i}] E_tot = {total_energy:.6f} eV, "
            f"E/atom = {epa:.6f} eV\n"
        )

        write(f'CONTCAR{i}', atoms, format='vasp')

min_epa = min(energies.values())

for i, epa in energies.items():
    if epa <= min_epa + threshold:
        atoms = read(f'CONTCAR{i}', format='vasp')
        write(f'CONTCARf_{i}', atoms, format='vasp')

