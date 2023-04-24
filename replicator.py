# makes artificial replicates of PDB

import copy
import parmed as pmd
import numpy as np
import openmm.app as app

fn = "/home/yppatel/misc/chignolin_idp/disordered_chignolin_eval/GYDPETGTWG.pdb"
pdb = pmd.load_file(fn)

num_replicates = 25

pdb_positions = np.array(pdb.positions._value)
pdbs = []
for n in range(num_replicates):
    pdb_offset = copy.deepcopy(pdb)
    pdb_offset.positions = pdb_positions + 100 * n
    pdbs.append(pdb_offset)

combined_pdb_fn = "combined.pdb"
combined = pdb
for replicate in pdbs[1:]:
    combined = combined + replicate
combined.save(combined_pdb_fn, overwrite=True)

combined_pdb = app.PDBFile(combined_pdb_fn)
print(combined_pdb.topology)