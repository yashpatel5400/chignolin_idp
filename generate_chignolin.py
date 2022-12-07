import time
from rdkit import Chem
from rdkit.Chem import AllChem, TorsionFingerprints
import json
import tqdm
import numpy as np

import multiprocessing
from multiprocessing.pool import Pool

import random
from concurrent.futures import ProcessPoolExecutor

from main.utils import *

def create_chignolin(mol_fn, out_dir):
    result_fn = os.path.join(out_dir, f'{os.path.basename(mol_fn).split(".")[0]}.json')
    if os.path.exists(result_fn):
        print(f"Already have: {result_fn}! Skipping...")
        return
    print(f"Creating: {result_fn}...")

    m = Chem.rdmolfiles.MolFromPDBFile(mol_fn, removeHs=False)
    num_torsions = len(TorsionFingerprints.CalculateTorsionLists(m)[0])
    print(f"{result_fn} -> {num_torsions}")
    
    Chem.SanitizeMol(m)
    AllChem.EmbedMultipleConfs(m, numConfs=1000, numThreads=-1)
    Chem.AllChem.MMFFOptimizeMoleculeConfs(m, numThreads=-1)


    confgen = ConformerGeneratorCustom(max_conformers=1,
                        rmsd_threshold=None,
                        force_field='mmff',
                        pool_multiplier=1)

    m = prune_conformers(m, 0.05)

    energys = confgen.get_conformer_energies(m)
    standard = energys.min()
    total = np.sum(np.exp(-(energys-standard)))

    nonring, ring = Chem.TorsionFingerprints.CalculateTorsionLists(m)
    rbn = len(nonring)
    out = {
        'mol': Chem.MolToSmiles(m, isomericSmiles=False),
        'standard': standard,
        'total': total
    }

    with open(result_fn, 'w') as fp:
        json.dump(out, fp)

def create_chignolin_wrapper(args):
    return create_chignolin(*args)

if __name__ == "__main__":
    in_dir = "/home/yppatel/misc/chignolin_idp/chignolin"
    out_dir = "/home/yppatel/misc/chignolin_idp/chignolin_out"

    fns = os.listdir(in_dir)
    full_fns = [(os.path.join(in_dir, fn), out_dir,) for fn in fns]

    multiprocessing.set_start_method('spawn')
    p = Pool(multiprocessing.cpu_count())
    p.map(create_chignolin_wrapper, full_fns)
    p.close()
    p.join()