# makes artificial replicates of PDB

import parmed as pmd
import copy

fn = "/home/yppatel/misc/chignolin_idp/disordered_chignolin_eval/GYDPETGTWG.pdb"
pdb = pmd.load_file(fn)

num_replicates = 5

pdbs = []
for n in range(num_replicates):
    pdb_offset = copy.deepcopy(pdb)
    pdb_offset.positions *= (1 + .05 * n)
    pdbs.append(pdb_offset)

combined = pdb
for replicate in pdbs[1:]:
    combined = combined + replicate
combined.save('combined.pdb')