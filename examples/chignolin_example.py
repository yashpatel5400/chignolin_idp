import sys
sys.path.append("/home/yppatel/misc/chignolin_idp/src/")

import numpy as np
import copy
import torch
import os
import pickle

import rdkit.Chem as Chem

from conformer_rl import utils
from conformer_rl.config import Config
from conformer_rl.environments import Task
from conformer_rl.models import RTGNRecurrent

from conformer_rl.molecule_generation.generate_molecule_config import config_from_rdkit
from conformer_rl.agents import PPORecurrentExternalCurriculumAgent
from conformer_rl.utils import MDSimulator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                                                                                                                                                                                                     
import logging
logging.basicConfig(level=logging.DEBUG)

import argparse

if __name__ == '__main__':
    utils.set_one_thread()

    # Create mol_configs for the curriculum
    chignolin_fasta = "YYDPETGTWY"
    curriculum_lens = [3, 5, 7, 10]
    curricula_num_conformers = [300, 500, 700, 1000]
    
    curriculum_fastas = [chignolin_fasta[:curriculum_len] for curriculum_len in curriculum_lens]
    chignolin_pdb_fns = [f"src/conformer_rl/molecule_generation/chignolin/{curriculum_fasta}.pdb" for curriculum_fasta in curriculum_fastas]
    simulator = MDSimulator(chignolin_pdb_fns)

    mol_configs = []
    for idx, chignolin_pdb_fn in enumerate(chignolin_pdb_fns):
        cached_filename = f"{curriculum_fastas[idx]}.pkl"
        if os.path.exists(cached_filename):
            with open(cached_filename, "rb") as f:
                print(f"Loading {cached_filename}...")
                mol_config = pickle.load(f)
        else:
            mol_config = config_from_rdkit(chignolin_pdb_fn, num_conformers=curricula_num_conformers[idx], calc_normalizers=True, simulator=simulator) 
            with open(cached_filename, "wb") as f:
                pickle.dump(mol_config, f)
        mol_config.mol_fn = chignolin_pdb_fn
        mol_configs.append(mol_config)
    
    eval_mol_config = copy.deepcopy(mol_configs[-1]) 

    config = Config()
    config.network = RTGNRecurrent(6, 128, edge_dim=6, node_dim=5).to(device)

    # Batch Hyperparameters
    config.max_steps = 200001

    # training Hyperparameters
    lr = 5e-6 * np.sqrt(10)
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=lr, eps=1e-5)
    config.discount = 0.9999
    config.use_gae = True
    config.gae_lambda = 0.95
    config.entropy_weight = 0.001
    config.value_loss_weight = 0.25
    config.gradient_clip = 0.5
    config.ppo_ratio_clip = 0.2

    # curriculum Hyperparameters
    tag = f"curriculum_chignolin_overall_min"
    config.curriculum_stable_conformers = [10, 10, 15, 20] # number of "true" stable conformers: used for assessing the agent per level
    config.curriculum_agent_buffer_len = 20
    config.curriculum_agent_reward_thresh = 0.4
    config.curriculum_agent_success_rate = 0.7
    config.curriculum_agent_fail_rate = 0.2
    config.tag = tag

    # Task Settings
    config.train_env = Task('GibbsScoreLogPruningCurriculumEnv-v0', concurrency=True, num_envs=10, seed=np.random.randint(0,1e5), mol_configs=mol_configs, tag=tag)
    config.eval_env = Task('GibbsScoreLogPruningEnv-v0', seed=np.random.randint(0,7e4), mol_config=eval_mol_config, tag=tag)
    config.eval_interval = 20000
    config.eval_episodes = 2

    agent = PPORecurrentExternalCurriculumAgent(config)
    agent.run_steps()