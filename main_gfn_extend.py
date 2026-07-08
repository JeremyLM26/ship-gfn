import torch
import numpy as np
import hydra
import randomname
import random
import matplotlib.pyplot as plt
import pickle
import gzip
import scipy.io as sco
import os # needed to keep track of directories
import sys
import subprocess # needed to compile and execute Fortran code from here

# importing ToyTokenizer as well here, needed to dynamically determine values for vocab_size and num_actions parameters
# change for extending GFN - need combo_to_index here as well
# this is because this GFN is not outputting full pathway indices, but indices for spin to append to pathways
from lib.utils.tokenizers_modified import str_to_tokens, tokens_to_str, ToyTokenizer, combo_to_index
from omegaconf import OmegaConf, DictConfig
from collections.abc import MutableMapping

from torch.distributions import Categorical
from tqdm import tqdm

#from comet_ml import Experiment

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def init_run(cfg, instance):
    trial_id = cfg.trial_id
    if cfg.job_name is None:
        cfg.job_name = '_'.join(randomname.get_name().lower().split('-') + [str(trial_id)])
    cfg.seed = random.randint(0, 100000) if cfg.seed is None else cfg.seed
    set_seed(cfg.seed)
    cfg = OmegaConf.to_container(cfg, resolve=True)  # Resolve config interpolations
    cfg = DictConfig(cfg)
    # logger.write_hydra_yaml(cfg)

    print(OmegaConf.to_yaml(cfg))
    config_filename = 'hydra_config_inst'+str(instance)+'.txt'
    with open(config_filename, 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    timestamp = cfg.timestamp # timestamp from config
    target_path = timestamp+"/"+config_filename
    # move file into appropriate directory, avoids overwriting
    subprocess.run(["mv", config_filename, target_path], check=True)
    return cfg

def flatten_config(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def get_distribution_plot(data):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set_xlabel("-Energy")
    ax.set_ylabel("Frequency")
    ax.hist(data, bins=100)
    return fig

class GFN:
    def __init__(self, cfg, tokenizer, scaling_factors):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.scaling_factors = scaling_factors
        self.setup_vars()
        self.init_policy()

    def setup_vars(self):
        cfg = self.cfg
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # Task stuff
        self.max_len = cfg.max_len
        self.min_len = cfg.min_len
        # GFN stuff
        self.train_steps = cfg.train_steps
        self.random_action_prob = cfg.random_action_prob
        self.batch_size = cfg.batch_size
        self.reward_min = cfg.reward_min
        self.gen_clip = cfg.gen_clip
        self.sampling_temp = cfg.sampling_temp
        self.sample_beta = cfg.sample_beta
        self.val_batch_size = cfg.val_batch_size
        self.eval_batch_size = cfg.eval_batch_size
        self.eval_samples = cfg.eval_samples
        # Eval Stuff
        self.eval_freq = cfg.eval_freq
        self.offline_gamma = cfg.offline_gamma
        self.eos_char = "[SEP]"
        self.pad_tok = self.tokenizer.convert_token_to_id("[PAD]")
        self.use_boltzmann = cfg.use_boltzmann

    def init_policy(self):
        cfg = self.cfg
        self.model = hydra.utils.instantiate(cfg.model)

        self.model.to(self.device)
        self.opt = torch.optim.Adam(self.model.model_params(), cfg.pi_lr, weight_decay=cfg.wd,
                            betas=(0.9, 0.999))
        self.opt_Z = torch.optim.Adam(self.model.Z_param(), cfg.z_lr, weight_decay=cfg.wd,
                            betas=(0.9, 0.999))

    def optimize(self, task, mem_length, num_indices_per_pathway, instance, initial_paths, counts, partial_rewards=False, init_data=None, val_data=None):
        """
        optimize the task involving multiple objectives (all to be maximized) with 
        optional data to start with
        """
        # added to save model
        cfg = self.cfg
        losses, rewards, lens = [], [], []
        val_losses, rews = 0, 0
        pb = tqdm(range(self.train_steps))
        desc_str = "Evaluation := Reward: {:.3f} Val Loss: {:.3f} | Train := Loss: {:.3f} Rewards: {:.3f} Scaling Factor: {:.3f}"
        pb.set_description(desc_str.format(0, 0, sum(losses[-10:]) / 10, sum(rewards[-10:]) / 10, 0))
        # filename to store the validation awards
        val_rewards_filename = f"validation_rewards_extend_len{mem_length}_inds{num_indices_per_pathway}_inst{instance}.txt"

        # remove existing files for validation reward data (if needed)
        subprocess.run(["rm", "-f", val_rewards_filename], check=True)
        # use shell=True to allow for wildcard expansion (*) here in subprocess module
        # shell=True here allows argument to be passed as a string instead of list elements as above
        subprocess.run([f"rm -f Newpathways-extend-len{mem_length}-inds{num_indices_per_pathway}-inst{instance}-row*"], shell=True, check=True)

        for i in pb:
            loss, r, le, counts = self.train_step(task, mem_length, num_indices_per_pathway, instance, i, initial_paths, counts, self.batch_size, init_data)
            #experiment.log_metrics({"train_loss": loss, "train_reward": r}, step=i)
            losses.append(loss)
            rewards.append(r)
            lens.append(le)
            if i % self.eval_freq == 0:
                with torch.no_grad():
                    #rews, val_losses, eval_data = self.evaluation(task, val_data)
                    #experiment.log_metrics({"val_loss": val_losses, "eval_reward": rews}, step=i)
                    #figure = get_distribution_plot(eval_data["rewards"])
                    #experiment.log_figure("generated_samples", figure, step=i)
                    #plt.close(figure)
                    torch.save(self.model.state_dict(),
                    "model_len"+str(mem_length)+"_inds"+str(num_indices_per_pathway)+"_inst"+str(instance)+".pt") # save the model
                    if partial_rewards == True:
                        samples, scores, scores1, scores2, scores3, counts = self.generate(cfg.eval_batch_size, task, mem_length, num_indices_per_pathway, instance, i, initial_paths, counts, partial_rewards=True)
                    else:
                        samples, scores, counts = self.generate(cfg.eval_batch_size, task, mem_length, num_indices_per_pathway, instance, i, initial_paths, counts)
                    samples = np.array(samples)[:, 1:] # removed index subtraction here (-4)
                    np.save("samples_len"+str(mem_length)+"_inds"+str(num_indices_per_pathway)+"_inst"+str(instance)+".npy", samples)
                    np.save("scores_len"+str(mem_length)+"_inds"+str(num_indices_per_pathway)+"_inst"+str(instance)+".npy", scores)
                    np.save("counts_len"+str(mem_length)+"_inds"+str(num_indices_per_pathway)+"_inst"+str(instance)+".npy", counts)
                    
                    if i >= 0:
                        # get index for highest reward (score) and its value
                        highest_index = np.argmax(scores)
                        highest_reward = scores[highest_index]

                        # for the row (GFN sequence) corresponding to the highest reward, copy new pathways added to initial set
                        # to a new file and move it to the proper directory - will be useful for analysis later
                        pathsum_dir = "../../../PATHSUM/" # pathsum directory
                        new_paths_filename = f"Newpathways-len{mem_length}-inds{num_indices_per_pathway}-inst{instance}-row{highest_index}.txt"
                        new_paths_filename_copy = \
                            f"Newpathways-len{mem_length}-inds{num_indices_per_pathway}-inst{instance}-row{highest_index}-copy.txt"
                        subprocess.run(["cp", new_paths_filename, new_paths_filename_copy], cwd=pathsum_dir, check=True)
                        file_destination = \
                            f"../GFN_extend/data/potts/Newpathways-extend-len{mem_length}-inds{num_indices_per_pathway}-inst{instance}-L{self.max_len}-reward{highest_reward}-trainstep{i}-scalingfactor{self.scaling_factors[i]}.txt"
                        subprocess.run(["mv", new_paths_filename_copy, file_destination], cwd=pathsum_dir, check=True)

                        # add partial rewards to end of new paths file if necessary, useful for analysis later
                        if partial_rewards == True:
                            # can drop directory names here, since this file has now been moved to the current (potts) directory
                            #new_paths_file = f"Newpathways-extend-len{mem_length}-inds{num_indices_per_pathway}-L{self.max_len}-reward{highest_reward}-trainstep{i}-scalingfactor{self.scaling_factors[i]}.txt"
                            new_paths_file = file_destination[25:]
                            with open(new_paths_file, "a") as file: # open in append mode, avoids overwriting data
                                file.write("\n"+str(scores1[highest_index]))
                                file.write("\n"+str(scores2[highest_index]))
                                file.write("\n"+str(scores3[highest_index]))

                        # store validation rewards in a file - will be useful for analysis later
                        with open(val_rewards_filename, "a") as file: # open in append mode, avoids overwriting data
                            # repeat highest reward self.eval_freq times - highest for this validation
                            # will be updated at next validation
                            for j in range(self.eval_freq):
                                file.write(str(highest_reward)+"\n")

            pb.set_description(desc_str.format(np.max(scores), 0, sum(losses[-10:]) / 10, sum(rewards[-10:]) / 10, self.scaling_factors[i]))
        
        return {
            'losses': losses,
            'train_rs': rewards
        }
    
    def sample_offline_data(self, dataset, batch_size):
        w = np.array(dataset[1])
        return np.random.choice(dataset[0], size=batch_size, replace=True, p = np.exp(w - w.max()) / np.exp(w-w.max()).sum(0))

    def train_step(self, task, mem_length, num_indices_per_pathway, instance, num_train_step, initial_paths, counts, batch_size, init_data=None):
        #if init_data is not None and self.offline_gamma > 0 and int(self.offline_gamma * batch_size) > 0:
        #    offline_batch = self.sample_offline_data(init_data, int(self.offline_gamma * batch_size))
        #    offline_logprobs = self._get_log_prob(offline_batch)
        #    logprobs = torch.cat((logprobs, offline_logprobs), axis=0)
        #    states = np.concatenate((states, offline_batch), axis=0)
        
        # added for training on batch on known full solutions for first 1000 training steps
        # build known full solution, save to device to avoid pytorch device mismatch, unsqueeze first dimension for batch dimension
        #if num_train_step <= 1000:
        #    #base_seq = torch.ones(self.min_len, dtype=torch.long, device=self.device)*15
        #    base_seq = torch.tensor([7, 7, 7, 7, 16, 16, 16, 16], dtype=torch.long, device=self.device)
        #    base_seq = torch.cat((torch.tensor([1], dtype=torch.long, device=self.device), base_seq))
        #    states = base_seq.unsqueeze(0).repeat(batch_size, 1) # add batch dimension, make batch of size batch_size
        #    # note: this will not affect gradients, since gradients flow through logits and not token indices

        #    logprobs, lens = self._get_log_prob(states) # shape (1,)
        #    #logprobs = torch.cat((logprobs, offline_logprobs), dim=0)
        #    #states = torch.cat((states, extra_sequence), dim=0)

        #else:
        #    states, logprobs, lens = self.sample(batch_size)

        states, logprobs, lens = self.sample(batch_size)
    
        r, counts = self.process_reward(states.tolist(), task, mem_length, num_indices_per_pathway, instance, num_train_step, initial_paths, counts)
        r = r.to(self.device)
        self.opt.zero_grad()
        self.opt_Z.zero_grad()
        # TB Loss
        if self.use_boltzmann:
            loss = (logprobs - self.sample_beta * r).pow(2).mean()
        else:
            loss = (logprobs - self.sample_beta * r.clamp(self.reward_min).log()).pow(2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gen_clip)
        self.opt.step()
        self.opt_Z.step()
        return loss.item(), r.mean(), lens.float().mean(), counts

    def sample(self, episodes, train=True):
        states = [''] * episodes
        traj_logprob = torch.zeros(episodes).to(self.device)

        active_mask = torch.ones(episodes).bool().to(self.device)
        x = str_to_tokens(states, self.tokenizer, use_sep=False).to(self.device).t()[:1].long()
        lens = torch.zeros(episodes).long().to(self.device)
        uniform_pol = torch.empty(episodes).fill_(self.random_action_prob).to(self.device)
        
        ep_len = self.max_len
        for t in (range(ep_len) if episodes > 0 else []):
            logits = self.model(x, lens=lens, mask=None)
            
            if t <= self.min_len:
                logits[:, 0] = -1000 # Prevent model from stopping
                                     # without having output anything
                if t == 0:
                    traj_logprob += self.model.Z()
            sampling_dist = Categorical(logits=logits / self.sampling_temp)
            pf_dist = Categorical(logits=logits)
            actions = sampling_dist.sample()
            if train and self.random_action_prob > 0:
                uniform_mix = torch.bernoulli(uniform_pol).bool()
                actions = torch.where(uniform_mix, torch.randint(int(t <= self.min_len), logits.shape[1], (episodes, )).to(self.device), actions)

            log_prob = pf_dist.log_prob(actions) * active_mask
            lens += torch.where(active_mask, torch.ones_like(lens), torch.zeros_like(lens))
            traj_logprob += log_prob
            
            # removed index addition here (actions + 4 -> actions)
            actions_apply = torch.where(torch.logical_not(active_mask), torch.zeros(episodes).to(self.device).long(), actions)
            active_mask = torch.where(active_mask, actions != 0, active_mask)

            x = torch.cat((x, actions_apply.unsqueeze(0)), axis=0)
            if active_mask.sum() == 0:
                break
        #states = tokens_to_str(x.t(), self.tokenizer)
        states = x.t()

        # section added - actions are selected from the tokenizer vocab since indices will not always match
        vocab_ints = [0] + self.tokenizer.non_special_vocab_ints
        vocab = torch.tensor(vocab_ints).to(self.device)

        states = vocab[states] # select tokens from vocab according to the chosen action from completed sequences

        return states, traj_logprob, lens
    
    # added partial_rewards parameter here for the option of tracking reward components (validation only)
    def generate(self, num_samples, task, mem_length, num_indices_per_pathway, instance, num_train_step, initial_paths, counts, partial_rewards=False):
        generated_seqs = []
        rewards = []
        rewards1, rewards2, rewards3 = [], [], []
        while len(generated_seqs) < num_samples:
            with torch.no_grad():
                samples, _, l = self.sample(self.eval_batch_size, train=False)
                if partial_rewards == True:
                    #r, r1, r2, r3, counts = (x.cpu().numpy().tolist() for x in self.process_reward(samples.tolist(), task, mem_length, num_indices_per_pathway, num_train_step, initial_paths, counts, partial_rewards=True))
                    r, r1, r2, r3, counts = self.process_reward(samples.tolist(), task, mem_length, num_indices_per_pathway, instance, num_train_step, initial_paths, counts, partial_rewards=True)
                    r = r.cpu().numpy().tolist()
                    r1 = r1.cpu().numpy().tolist()
                    r2 = r2.cpu().numpy().tolist()
                    r3 = r3.cpu().numpy().tolist()
                    rewards.extend(r)
                    rewards1.extend(r1)
                    rewards2.extend(r2)
                    rewards3.extend(r3)
                else:
                    r, counts = self.process_reward(samples.tolist(), task, mem_length, num_indices_per_pathway, instance, num_train_step, initial_paths, counts)
                    r = r.cpu().numpy().tolist()
                    rewards.extend(r)
            generated_seqs.extend(samples.tolist())
            #rewards.extend(r)
        if partial_rewards == True:
            return np.array(generated_seqs), np.array(rewards), np.array(rewards1), np.array(rewards2), np.array(rewards3), counts
        else:
            return np.array(generated_seqs), np.array(rewards), counts

    def process_reward(self, seqs, task, mem_length, num_indices_per_pathway, instance, num_train_step, initial_paths, counts, partial_rewards=False):
        if partial_rewards == True:
            r, r1, r2, r3, counts = task(seqs, mem_length, num_indices_per_pathway, instance, num_train_step, initial_paths, counts, partial_rewards=True)
            #r = torch.tensor(r)
            #r1 = torch.tensor(r1)
            #r2 = torch.tensor(r2)
            #r3 = torch.tensor(r3)
            return (torch.as_tensor(r, dtype=torch.float32), torch.as_tensor(r1, dtype=torch.float32), torch.as_tensor(r2, dtype=torch.float32), torch.as_tensor(r3, dtype=torch.float32), counts)
        else:
            r, counts =  task(seqs, mem_length, num_indices_per_pathway, instance, num_train_step, initial_paths, counts)
            return torch.as_tensor(r, dtype=torch.float32), counts

    def val_step(self, val_data, batch_size, task, mem_length, num_indices_per_pathway, instance, num_train_step, initial_paths, counts):
        overall_loss = 0.
        num_batches = max(1, len(val_data[0]) // batch_size)
        losses = 0
        for i in range(num_batches):
            states = val_data[0][i*batch_size:(i+1)*batch_size]
            logprobs = self._get_log_prob(states)

            r, counts = self.process_reward(states, task, mem_length, num_indices_per_pathway, instance, num_train_step, initial_paths, counts)
            r = r.to(logprobs.device)
            loss = (logprobs - self.sample_beta * r.clamp(min=self.reward_min).log()).pow(2).mean()

            losses += loss.item()
        overall_loss += (losses)
        return overall_loss / num_batches

    def evaluation(self, task, mem_length, num_indices_per_pathway, instance, num_train_step, initial_paths, counts, val_data):
        val_loss = self.val_step(val_data, self.val_batch_size, task, mem_length, num_indices_per_pathway, instance, num_train_step, initial_paths, counts)
        samples, rewards, counts = self.generate(self.eval_samples, task, mem_length, num_indices_per_pathway, instance, num_train_step, initial_paths, counts)
        return rewards.mean(), val_loss, {'samples': samples, 'rewards': rewards.tolist()}

    def _get_log_prob(self, states):
        #lens = torch.tensor([len(z) + 2 for z in states]).long().to(self.device)
        #x = str_to_tokens(states, self.tokenizer).to(self.device).t()
        #mask = x.eq(self.tokenizer.padding_idx)
        #logits = self.model(x, mask=mask.transpose(1,0), return_all=True, lens=lens, logsoftmax=True)
        #seq_logits = (logits.reshape(-1, 4)[torch.arange(x.shape[0] * x.shape[1], device=self.device), (x.reshape(-1)-5).clamp(0)].reshape(x.shape) * mask.logical_not().float()).sum(0)
        #seq_logits += self.model.Z()
        #return seq_logits

        # function rewritten for insertion of known sequences
        # states in tensor of shape (batch size, seq_len) containing action indices included prepended [CLS] token at the start
        if states.dim() == 1:
            states = states.unsqueeze(0) # make batch_size = 1
        
        batch_size, seq_len = states.shape
        lens = torch.full((batch_size,), seq_len, dtype=torch.long, device=self.device)

        x = states.to(self.device) # ensure on device
        mask = x.eq(self.tokenizer.padding_idx)
        
        # forward pass through model
        # changed mask=mask.transpose(1, 0) to mask=mask, since model expects mask of size (batch_size, seq_len)
        logits = self.model(x.t(), mask=mask, return_all=True, lens=lens, logsoftmax=True)
        # logits: (seq_len, batch_size, num_actions)

        # flatten to index each timestep
        flat_logits = logits.view(-1, logits.size(-1)) # (seq_len*batch_size, num_actions)
        flat_indices = x.reshape(-1)

        # gather log probs of the actual actions
        selected = flat_logits[torch.arange(flat_logits.size(0), device=self.device), flat_indices]

        # reshape back to (seq_len, batch_size) and mask - ~ inverts the booleans
        selected = selected.view(seq_len, batch_size) * (~mask).float().T

        # sum over timesteps to get the total logprob per sequence
        seq_logprobs = selected.sum(0) + self.model.Z()

        return seq_logprobs, lens # shape (batch size,)

class PottsReward:
    def __init__(self, num_of_elements, tokenizer, pathsum_input_file, dynamics_executable, scaling_factors, paths_per_index, divisible, prefix="/home/mila/m/moksh.jain/BioSeq-GFN-AL/data/"):
        """
        load model and tokenizer
        """
        self.tokenizer = tokenizer
        self.pathsum_input_file = pathsum_input_file
        self.dynamics_executable = dynamics_executable
        self.scaling_factors = scaling_factors
        self.paths_per_index = paths_per_index
        self.divisible = divisible
        self.index_possibilities = [(1, ), (2, ), (3, ), (4, ), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4), (1, 2, 3), (1, 2, 4),
                       (1, 3, 4), (2, 3, 4), (1, 2, 3, 4), (5, )]
        self.index_dict = {ind+1: val for (ind, val) in enumerate(self.index_possibilities)}
        #self.J=np.zeros(shape=(4,4,num_of_elements))
        #for ii in range(num_of_elements):
            #nextJJ=prefix+'potts/JJ_'+str(ii+1)+'.mat'
            #Jdict=sco.loadmat(nextJJ)
            #self.J[:,:,ii]=Jdict['JJ_out']

        #hdict=sco.loadmat(prefix+'potts/h_out.mat')
        #self.h=hdict['h']
    
    def __call__(self, seqs, mem_length, num_indices_per_pathway, instance, num_train_step, initial_paths, counts, partial_rewards=False):
        # computes GFN reward for completed sequence

        # commented out extra lines - GFN now outputs tokens directly, see end of sample function (states = x.t())
        #seqs = str_to_tokens(seqs, self.tokenizer).numpy() - 4
        # removed index subtraction here (-4)
        seqs = np.array(seqs)[:, 1:] # [:, 1:] removes beginning of sequence token from all sequences
        #seqs = seqs[:, 1:-1] # removes CLS and SEP from seqs

        # introduce new reward function for spin dynamics problem
        # right now, seqs is a numpy array of dimension (batch_size,max_len)
        rewards = np.array([])
        rewards1 = np.array([])
        rewards2 = np.array([])
        rewards3 = np.array([])
        for row_index in range(len(seqs)):
            # change for extending GFN - reward is set up slightly differently
            # use indices outputted by GFN to append spins to initial paths, map larger paths back to indices,
            # and convert to a file used for computing dynamics
            row = seqs[row_index] # isolate a single sequence of indices
            num_vals = len(row) # number of values to consider is the sequence length
            extended_paths = np.array([])  # array to hold larger spin pathways

            for index1 in range(num_vals):  # loop over all entries in the sequence
                entry_val = row[index1]  # isolate entry for individual pathway
                #if num_train_step > 1000:
                #    counts[entry_val-1] = counts[entry_val-1]+1 # increment token count
                counts[entry_val-1] = counts[entry_val-1]+1 # increment token count
                index_vals = self.index_dict[entry_val]  # convert sequence entry into tuple of possibilities from index dictionary
                forward_spins = np.array([]) # arrays to hold forward and backward spins
                backward_spins = np.array([])
                for index2 in range(len(index_vals)):  # loop over entries in each tuple
                    if index_vals[index2] == 1:  # extend spin pathways accordingly
                        extra_forward_spin = '1'
                        extra_backward_spin = '1'
                        forward_spins = np.append(forward_spins, extra_forward_spin)
                        backward_spins = np.append(backward_spins, extra_backward_spin)
                    if index_vals[index2] == 2:
                        extra_forward_spin = '1'
                        extra_backward_spin = '2'
                        forward_spins = np.append(forward_spins, extra_forward_spin)
                        backward_spins = np.append(backward_spins, extra_backward_spin)
                    if index_vals[index2] == 3:
                        extra_forward_spin = '2'
                        extra_backward_spin = '1'
                        forward_spins = np.append(forward_spins, extra_forward_spin)
                        backward_spins = np.append(backward_spins, extra_backward_spin)
                    if index_vals[index2] == 4:
                        extra_forward_spin = '2'
                        extra_backward_spin = '2'
                        forward_spins = np.append(forward_spins, extra_forward_spin)
                        backward_spins = np.append(backward_spins, extra_backward_spin)
                    if index_vals[index2] == 5:
                        continue
                if len(forward_spins) == 0: # skip paths if there are no forward spins, move to next sequence entry
                    continue
                # if divisible is True, or divisible is False but not last entry in sequence, extend pathways as normal
                if self.divisible == True or index1 != num_vals-1:
                    for index3 in range(self.paths_per_index):
                        for index4 in range(len(index_vals)):
                            new_path = initial_paths[self.paths_per_index*index1+index3][0:mem_length]+forward_spins[index4]+initial_paths[self.paths_per_index*index1+index3][mem_length:2*mem_length]+backward_spins[index4]
                            extended_paths = np.append(extended_paths, new_path)
                else: # if neither of the above are True, only extend half the number of pathways to reach the end of file
                    for index3 in range(int(self.paths_per_index/2)): # this range is half its usual value
                        for index4 in range(len(index_vals)):
                            new_path = initial_paths[self.paths_per_index*index1+index3][0:mem_length]+forward_spins[index4]+initial_paths[self.paths_per_index*index1+index3][mem_length:2*mem_length]+backward_spins[index4]
                            extended_paths = np.append(extended_paths, new_path)

            paths_array = [list(extended_paths[i]) for i in range(len(extended_paths))]  # break extended paths into 2D array with spins as individual entries
            paths_array = [[int(paths_array[i][j])-1 for j in range(len(paths_array[i]))] for i in range(len(paths_array))]
            paths_array = np.array(paths_array)

            indices = np.zeros(len(paths_array), dtype=int)  # array to hold corresponding indices for pathways
            for num_it in range(len(extended_paths)):
                index = combo_to_index(paths_array[num_it], int(len(paths_array[num_it])/2))
                indices[num_it] = index

            # convert array of indices into an output file, used for computing fortran dynamics
            filename = f"GFNpathways_len{mem_length}_inds{num_indices_per_pathway}_inst{instance}_row{row_index}.txt"
            with open(filename, "w") as file:
                np.savetxt(file, indices, fmt="%s")
        
            # move file to correct directory
            destination = f"../../../PATHSUM/{filename}"
            subprocess.run(["mv", filename, destination], check=True)

            # next, use file with pathway indices to compute dynamics in fortran
            # do not need to compile fortran code here, already done in main block
            pathsum_dir = "../../../PATHSUM/" # pathsum directory
            # run dynamics with pathways from GFN
            subprocess.run(["./"+self.dynamics_executable, self.pathsum_input_file, filename], cwd=pathsum_dir, check=True)

            # now read in results from fortran dynamics back into python, needed to compute reward
            # GFN dynamics - skip 29 rows of text to load in proper values
            GFN_dynamics_filename = f"../../../PATHSUM/SMatPI-dynamics-pop-len{mem_length}-inds{num_indices_per_pathway}-inst{instance}-row{row_index}.out"
            GFN_dynamics = np.loadtxt(GFN_dynamics_filename, skiprows=29)
            # full dynamics - skip 25 rows of text to load in proper values
            full_dynamics_filename = f"../../../PATHSUM/SMatPI-dynamics-pop-len{mem_length}-inds{num_indices_per_pathway}-inst{instance}-full.out"
            full_dynamics = np.loadtxt(full_dynamics_filename, skiprows=25)
            # GFN imaginary parts of reduced density matrix entries
            GFN_ImRDM_filename = f"../../../PATHSUM/ImRDM-len{mem_length}-inds{num_indices_per_pathway}-inst{instance}-row{row_index}.txt"
            GFN_ImRDM = np.loadtxt(GFN_ImRDM_filename)

            #compute the final reward coming directly from the populations and trace of the dynamics
            scaling_factor1 = 50 # scaling factor serving as a prefactor for first exponential reward
            scaling_factor2 = self.scaling_factors[num_train_step] # scaling factor serving as temperature for the exponential rewards
            #if num_train_step < 1000: # added condition here: reward distribution is sharpened after 1000 training steps
            #    scaling_factor2 = 500 # scaling factor serving as temperature for the exponential rewards
            #else:
            #    scaling_factor2 = 5
            scaling_factor3 = 25 # scaling factor serving as a prefactor for second and third exponential reward
            
            # sum over all iterations to favour correct behaviour over all dynamics
            # first part of reward - conserved and consistent trace (close to 1, small changes over all iterations)
            trace = GFN_dynamics[:,3] # get trace data
            reward1 = scaling_factor1*np.exp(-np.sum(np.absolute(trace-1))/scaling_factor2)

            # second part of reward - real populations: favour small imaginary parts of populations
            im_pop1 = GFN_ImRDM[:,1] # get imaginary parts of populations (diagonal entries)
            im_pop2 = GFN_ImRDM[:,4]
            reward2 = scaling_factor3*(np.exp(-np.sum(np.absolute(im_pop1))/scaling_factor2)+
            np.exp(-np.sum(np.absolute(im_pop2))/scaling_factor2))
            
            # third part of reward - similar populations to reference: favour small changes in physics
            pop1 = GFN_dynamics[:,1] # get GFN dynamics population data
            pop2 = GFN_dynamics[:,2]
            full_pop1 = full_dynamics[:,1] # get full dynamics population data
            full_pop2 = full_dynamics[:,2]
            reward3 = scaling_factor3*(np.exp(-np.sum(np.absolute(
            pop1-full_pop1))/scaling_factor2)+
            np.exp(-np.sum(np.absolute(pop2-full_pop2))/scaling_factor2))

            reward = reward1+reward2+reward3
            rewards = np.append(rewards, reward)
            if partial_rewards == True:
                rewards1 = np.append(rewards1, reward1)
                rewards2 = np.append(rewards2, reward2)
                rewards3 = np.append(rewards3, reward3)

        if partial_rewards == True:
            return rewards, rewards1, rewards2, rewards3, counts
        else:
            return rewards, counts

@hydra.main(config_path='./config', config_name='potts.yaml')
def main(config):
    random.seed(None)

    # get pathsum input filename and dynamics executable filename, passed as an argument at runtime
    # this will allow multiple runs that read and write files without conflicting
    # change here for extending GFN - read in filename from previous run to use as a baseline
    pathsum_input_file = config.pathsum_input_file
    dynamics_executable = config.dynamics_executable
    starting_paths_input_file1 = config.starting_paths_input_file1
    #starting_paths_input_file2 = config.starting_paths_input_file2

    # current working directory is timestamp directory for storing hydra files, need to be in potts directory
    # timestamp directory is in the potts directory
    # change for extending GFN - this directory change happens sooner
    os.chdir("../")

    if not os.path.exists(starting_paths_input_file1):
        print(f"ERROR: File not found: {starting_paths_input_file1}", flush=True)
        sys.exit(1)

    # change here for extending GFN - sequence length will depend on the number of pathways in the input file
    # read in pathways from solution
    #with open(starting_paths_input_file1, 'r') as file1:
    #    lines = file1.readlines()
    #    paths1 = np.array([str(lines[i].strip()) for i in range(0, len(lines[:-4]), 2)])

    with open(starting_paths_input_file1, 'r') as file1:
        lines = file1.readlines()
        paths1 = np.array([str(lines[i].strip().replace(' ', '')) for i in range(1, len(lines))])

    #with open(starting_paths_input_file2, 'r') as file2:
    #    lines = file2.readlines()
    #    paths2 = np.array([str(lines[i].strip()) for i in range(0, len(lines[:-4]), 2)])

    #paths = np.concatenate((paths1, paths2))
    paths = paths1
    num_paths = len(paths)

    num_train_steps = config.gfn.train_steps
    #num_learning_steps = 1000
    #num_loops = int(num_train_steps/num_learning_steps)
    #scaling_factors = np.array([])
    #for num in range(num_loops):
    #    extra_factors = np.full(num_learning_steps, 500-num*50)
    #    scaling_factors = np.hstack((scaling_factors, extra_factors))
    scaling_factors = np.full(num_train_steps, 50)

    # read memory length for dynamics and number of indices per pathway from input file
    directory_path = str(os.getcwd())[:-21] # remove last characters from path, which are GFN_extend/data/potts
    file_path = directory_path+"PATHSUM/"+pathsum_input_file
    file = open(file_path, "r")
    for _ in range(24): # memory length is 24th line in input file
        line = file.readline()
    mem_length = int(line.replace(" ",""))
    mem_length_str = str(mem_length+1)
    for _ in range(25): # read another 25 lines, up to line 49, for number of indices per pathway
        line = file.readline()
    num_indices_per_pathway = int(line.replace(" ",""))
    for _ in range(2): # read another 2 lines, up to line 51, for number of paths per index (coarse-grained representation)
        line = file.readline()
    paths_per_index = int(line.replace(" ", ""))
    for _ in range(2): # read another 2 lines, up to line 53, for instance number (multiple GFN runs of same memory length)
        line = file.readline()
    instance = int(line.replace(" ", ""))
    file.close()

    # assign GFN length
    # 5 possible options for extension: (0, 0), (0, 1), (1, 0), (1, 1), skip
    # GFN will provide the options for each pathway - 16 ways to choose 1-4 forward-backward spin pairs or skip (see tokenizer)
    # note: paths_per_index can be double its value from a previous run here, this will affect GFN sequence length, treat carefully here
    if num_paths % paths_per_index == 0: # if paths_per_index divides num_paths, treat sequence length normally
        config.gfn.min_len = int(num_paths/paths_per_index) # change for extending GFN: each index maps to spins for multiple pathways
        divisible = True
    else: # if not, GFN sequence will be 1 entry longer, but last entry will only read in half the number of paths
        config.gfn.min_len = int(num_paths/paths_per_index)+1
        divisible = False
    config.gfn.max_len = config.gfn.min_len
    
    # assign model maximum length fron GFN maximum length - passed as argument at runtime
    config.gfn.model.max_len = config.gfn.max_len+2

    # pathsum directory
    pathsum_dir = file_path[:-15] # remove last 15 characters from file path, which correspond to PathSumSysi.inp
    #print(pathsum_dir)

    # remove existing input files if necessary
    #subprocess.run(["make", "clean-all-rows", "len="+str(mem_length), "inds="+str(num_indices_per_pathway)], cwd=pathsum_dir, check=True)
    #subprocess.run(["make", "clean-full", "len="+str(mem_length), "inds="+str(num_indices_per_pathway)], cwd=pathsum_dir, check=True)
    #subprocess.run(["make", "clean-pathways", "len="+str(mem_length), "inds="+str(num_indices_per_pathway)], cwd=pathsum_dir, check=True)
    #subprocess.run(["make", "clean-indices", "len="+str(mem_length), "inds="+str(num_indices_per_pathway)], cwd=pathsum_dir, check=True)

    # create input files with spin pathways if necessary
    # all pathways
    subprocess.run(["python", "build_pathways.py", "--L", mem_length_str, "--max_blips", mem_length_str, "--num_inds", str(num_indices_per_pathway), "--instance", str(instance), "--extra_paths", "False", "--empty_file", "False"], cwd=pathsum_dir, check=True)
    # initial set (filtered, max_blips=mem_length-1) for now
    #subprocess.run(["python", "build_pathways.py", "--L", mem_length_str, "--max_blips", str(mem_length), "--num_inds", str(num_indices_per_pathway), "--extra_paths", "True", "--empty_file", "False"], cwd=pathsum_dir, check=True)
    # empty initial set file
    subprocess.run(["python", "build_pathways.py", "--L", mem_length_str, "--max_blips", str(mem_length), "--num_inds", str(num_indices_per_pathway), "--instance", str(instance), "--extra_paths", "False", "--empty_file", "True"], cwd=pathsum_dir, check=True)
    # initial set not dependent on blips, but specific selected paths
    #subprocess.run(["python", "build_pathways.py", "--L", mem_length_str, "--max_blips", str(-1), "--num_inds", str(num_indices_per_pathway), "--extra_paths", "True", "--empty_file", "False"], cwd=pathsum_dir, check=True)

    # create files with spin indices if necessary
    # compile fortran code - only needs to be done once here
    subprocess.run(["make", "get-indices"], cwd=pathsum_dir, check=True)
    # all pathways
    subprocess.run(["./get_indices.out", pathsum_input_file, "full"], cwd=pathsum_dir, check=True)
    # initial set (filtered, max_blips=mem_length-1) for now
    subprocess.run(["./get_indices.out", pathsum_input_file], cwd=pathsum_dir, check=True)

    # run full dynamics on the given system using all spin pathways - needed for reward
    # compile fortran code - only needs to be done once here
    subprocess.run(["make", "run-dynamics", "executable="+dynamics_executable], cwd=pathsum_dir, check=True)
    # all pathways
    subprocess.run(["./"+dynamics_executable, pathsum_input_file, "full"], cwd=pathsum_dir, check=True)

    # next, read in values for num_actions and vocab_size parameters, based on values from dynamics input file
    # placeholder tokenizer - send pathsum file as input here
    tokenizer_placeholder = ToyTokenizer(pathsum_input_file)
    # get num_actions and vocab_size parameters from placeholder
    num_actions = len(tokenizer_placeholder.non_special_vocab)+1
    vocab_size = num_actions+4
    # update parameters dynamically
    config.gfn.model.num_actions = num_actions
    config.gfn.model.vocab_size = vocab_size
    counts = np.zeros(num_actions-1, dtype=int) # array for holding counts of each token (size of tokenizer non-special vocab)

    log_config = flatten_config(OmegaConf.to_container(config, resolve=True), sep='/')
    log_config = {'/'.join(('config', key)): val for key, val in log_config.items()}
    
    # config['job_name'] = wandb.run.name
    config = init_run(config, instance)
    
    #experiment.log_parameters(log_config)

    tokenizer = hydra.utils.instantiate(config.tokenizer)
    #dataset = hydra.utils.instantiate(config.dataset, tokenizer=tokenizer)

    task = PottsReward(config.num_of_elements, tokenizer, pathsum_input_file, dynamics_executable, scaling_factors, paths_per_index, divisible, prefix=config.prefix)

    generator = GFN(config.gfn, tokenizer, scaling_factors)
    #generator.model.load_state_dict(torch.load("model_len"+str(mem_length)+".pt")) # load the model
    #generator.optimize(task, init_data=dataset.training_data, val_data=None) #dataset.validation_set())
    # added mem_length as input parameter here, so that it does not need to be read in again
    # change for extending GFN - added paths as a parameter here so that starting paths file does not need to be read again
    generator.optimize(task, mem_length, num_indices_per_pathway, instance, paths, counts, partial_rewards=True, init_data=None, val_data=None)
    torch.save(generator.model.state_dict(), "model_len"+str(mem_length)+"_inds"+str(num_indices_per_pathway)+"_inst"+str(instance)+".pt") # save the model

    #samples, scores = generator.generate(config.num_samples, task, mem_length, num_indices_per_pathway, paths)
    #samples = np.array(samples)[:, 1:] # removed index subtraction here (-4)
    #np.save("samples_len"+str(mem_length)+"_inds"+str(num_indices_per_pathway)+".npy", samples)
    #np.save("scores_len"+str(mem_length)+"_inds"+str(num_indices_per_pathway)+".npy", scores)
    #figure = get_distribution_plot(scores)
    #experiment.log_figure(figure_name="final_distribution", figure=figure)
    #plt.close(figure)
    #experiment.log_others({'samples': samples, 'scores': scores.tolist()})

if __name__ == "__main__":
    #experiment = Experiment(project_name="GFN-Potts")
    main()
