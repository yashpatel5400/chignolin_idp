# makes artificial replicates of PDB

import parmed as pmd
import copy
import openmm.app as app

fn = "/home/yppatel/misc/chignolin_idp/disordered_chignolin_eval/GYDPETGTWG.pdb"
pdb = pmd.load_file(fn)

num_replicates = 2

pdbs = []
for n in range(num_replicates):
    pdb_offset = copy.deepcopy(pdb)
    pdb_offset.positions *= (1 + .05 * n)
    pdbs.append(pdb_offset)

combined_pdb_fn = "combined.pdb"
combined = pdb
for replicate in pdbs[1:]:
    combined = combined + replicate
combined.save(combined_pdb_fn, overwrite=True)

combined_pdb = app.PDBFile(combined_pdb_fn)
print(combined_pdb.topology)