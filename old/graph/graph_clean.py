import numpy as np
import torch as th
from torch import tensor
from torch.utils.data import TensorDataset
import torch.nn.functional as F
from utils import randexclude
import networkx as nx
from matplotlib import pyplot

# Toy dataset from Decision Transformer (Chen et. al 2021)
class RandomWalks:
    def __init__(self, n_nodes=20, max_length=10, n_walks=1000, p_edge=0.1, seed=1002):
        self.n_nodes = n_nodes
        self.n_walks = n_walks
        self.max_length = max_length
        self.walk_size = max_length
        rng = np.random.RandomState(seed)

        walks, rewards = [], []
        while True:
            self.adj = rng.rand(n_nodes, n_nodes) > (1 - p_edge)
            np.fill_diagonal(self.adj, 0)
            if np.all(self.adj.sum(1)): break

        # terminal state
        self.adj[0, :] = 0
        self.adj[0, 0] = 1

        self.goal = 0
        for _ in range(n_walks):
            node = randexclude(rng, n_nodes, self.goal)
            walk = [node]

            for istep in range(max_length-1):
                node = rng.choice(np.nonzero(self.adj[node])[0])
                walk.append(node)
                if node == self.goal:
                    break

            r = th.zeros(max_length-1)
            r[:len(walk)-1] = -1 if walk[-1] == self.goal else -100

            rewards.append(r)
            walks.append(walk)

        states = []
        attention_masks = []

        for r, walk in zip(rewards, map(th.tensor, walks)):
            attention_mask = th.zeros(max_length, dtype=int)
            attention_mask[:len(walk)] = 1

            attention_masks.append(attention_mask)
            states.append(F.pad(walk, (0, max_length-len(walk))))

        self.worstlen = self.max_length
        self.avglen = sum(map(len, walks)) / self.n_walks
        self.bestlen = 0
        g = nx.from_numpy_array(self.adj, create_using=nx.DiGraph)
        for start in set(range(self.n_nodes)) - {self.goal}:
            try:
                shortest_path = nx.shortest_path(g, start, self.goal)[:self.max_length]
                self.bestlen += len(shortest_path)
            except:
                self.bestlen += self.max_length

        self.bestlen /= self.n_nodes - 1

        print(f'{self.n_walks} walks of which {(np.array([r[0] for r in rewards])==-1).mean()*100:.0f}% arrived at destination')

        # disallows selecting unaccessible nodes in a graph
        self.logit_mask = tensor(~self.adj)

        self.dataset = TensorDataset(th.stack(states), th.stack(attention_masks), th.stack(rewards))
        self.eval_dataset = TensorDataset(th.arange(1, self.n_nodes).unsqueeze(1))

    def render(self):

        g = nx.from_numpy_array(self.adj, create_using=nx.DiGraph)
        pos = nx.spring_layout(g, seed=7357)

        pyplot.figure(figsize=(10, 8))
        nx.draw_networkx_edges(g, pos=pos, alpha=0.5, width=1, edge_color='#d3d3d3')
        nx.draw_networkx_nodes(g, nodelist=set(range(len(self.adj))) - {self.goal}, pos=pos, node_size=300, node_color='orange')
        nx.draw_networkx_nodes(g, nodelist=[self.goal], pos=pos, node_size=300, node_color='darkblue')
        pyplot.show()

    def eval(self, samples, beta):
        narrived = 0
        actlen = 0
        for node in range(self.n_nodes-1):
            for istep in range(self.max_length):
                if samples[node, istep] == self.goal:
                    narrived += 1
                    break

            actlen += (istep + 1) / (self.n_nodes - 1)

        current = (self.worstlen - actlen)/(self.worstlen - self.bestlen)
        average = (self.worstlen - self.avglen)/(self.worstlen - self.bestlen)

        stats = { 'actlen': actlen,
                  'avglen': self.avglen,
                  'bestlen': self.bestlen,
                  'worstlen': self.worstlen,
                  'arrived': f'{narrived / (self.n_nodes-1) * 100:.0f}%',
                  'optimal': f'{current*100:.0f}% > {average*100:.0f}%' }

        return -actlen, stats
    
import os
import torch as th
import numpy as np
from torch import tensor, nn
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, PretrainedConfig, AutoConfig

from typing import NamedTuple, Tuple, Union
from copy import deepcopy
from collections import defaultdict
# from accelerate.utils import compute_module_sizes
from itertools import chain

# import accelerate
# import deepspeed

def topk_mask(xs: th.FloatTensor, k: int):
    mintop = th.topk(xs, k)[0][:, -1].unsqueeze(-1)
    return th.where(xs < mintop, -np.inf * th.ones_like(xs, dtype=xs.dtype), xs)

class QVOutput(Tuple):
    logits: th.FloatTensor
    qs: th.FloatTensor
    target_qs: th.FloatTensor
    vs: th.FloatTensor
    past_key_values: Tuple[th.FloatTensor]

def make_head(n_embd: int, out: int):
    return nn.Sequential(
        nn.Linear(n_embd, n_embd * 2),
        nn.ReLU(),
        nn.Linear(n_embd * 2, out)
    )

class QVModel(nn.Module):
    def __init__(self, config: Union[PretrainedConfig, str], params):
        super().__init__()

        # enable zero3 init within from_pretrained
        if os.environ.get('DEEPSPEED_ZERO_STAGE', '0') == '3':
            config_path = os.environ.get('DEEPSPEED_CONFIG_FILE', '')
            if config_path:
                _hfconfig = transformers.deepspeed.HfDeepSpeedConfig(config_path)

        if isinstance(config, PretrainedConfig):
            self.gpt = AutoModelForCausalLM.from_config(config)
        else:
            self.gpt = AutoModelForCausalLM.from_pretrained(config)

        if hasattr(self.gpt.config, 'hidden_size'):
            self.n_embd = self.gpt.config.hidden_size
        else:
            self.n_embd = self.gpt.config.n_embd
        self.vocab_size = self.gpt.config.vocab_size

        self.v_head = make_head(self.n_embd, 1)
        self.q1_head = make_head(self.n_embd, self.vocab_size)
        self.target_q1_head = deepcopy(self.q1_head)
        self.target_q1_head.requires_grad_(False)

        self.tau = params['tau']
        self.alpha = params['alpha']
        self.gamma = params['gamma']
        self.awac_scale = params['awac_scale']
        self.cql_scale = params['cql_scale']
        self.two_qs = params['two_qs']

        if self.two_qs:
            self.q2_head = make_head(self.n_embd, self.vocab_size)
            self.target_q2_head = deepcopy(self.q2_head)
            self.target_q2_head.requires_grad_(False)

    def forward(self, **x):
        if hasattr(self.gpt, 'gpt_neox'):
            out = self.gpt.gpt_neox(**x)
        else:
            out = self.gpt.transformer(**x)

        hs = out.last_hidden_state

        if self.two_qs:
            qs = (self.q1_head(hs), self.q2_head(hs))
            target_qs = (self.target_q1_head(hs), self.target_q2_head(hs))
        else:
            qs = self.q1_head(hs)
            target_qs = self.target_q1_head(hs)

        if hasattr(self.gpt, 'gpt_neox'):
            logits = self.gpt.embed_out(hs)
        else:
            logits = self.gpt.lm_head(hs)

        return QVOutput((logits, qs, target_qs, self.v_head(hs), out.past_key_values))

    def loss(self, batch):
        tokens, attn, rewards = batch
        actions = tokens[:, 1:, None]
        isterminal = attn[:, :-1]

        logits, qs, target_qs, vs, _ = self(input_ids=tokens, attention_mask=attn)
        bsize, ntokens, dsize = logits.shape

        if self.two_qs:
            Q1 = qs[0][:, :-1].gather(-1, actions).squeeze(-1)
            Q2 = qs[1][:, :-1].gather(-1, actions).squeeze(-1)

            targetQ1 = target_qs[0][:, :-1].gather(-1, actions).squeeze(-1).detach()
            targetQ2 = target_qs[1][:, :-1].gather(-1, actions).squeeze(-1).detach()
            targetQ = th.minimum(targetQ1, targetQ2)
        else:
            Q = qs[:, :-1].gather(-1, actions).squeeze(-1)
            targetQ = target_qs[:, :-1].gather(-1, actions).squeeze(-1).detach()

        n_nonterminal = max(1, isterminal.sum())
        V = vs[:, 1:].squeeze() * isterminal
        Q_ = rewards + self.gamma * V

        if self.two_qs:
            loss_q1 = ((Q1 - Q_.detach()) * isterminal).pow(2).sum() / n_nonterminal
            loss_q2 = ((Q2 - Q_.detach()) * isterminal).pow(2).sum() / n_nonterminal
            loss_q = loss_q1 + loss_q2
        else:
            loss_q = ((Q - Q_.detach()) * isterminal).pow(2).sum() / n_nonterminal

        loss_v = (((targetQ >= V).int() * self.tau * (targetQ - V).pow(2) + (targetQ < V).int() * (1 - self.tau) * (targetQ - V).pow(2)) * isterminal).sum() / n_nonterminal

        if self.two_qs:
            loss_cql_q1 = (F.cross_entropy(qs[0][:, :-1].reshape(-1, dsize), actions.reshape(-1), reduction='none').reshape(bsize, ntokens-1) * isterminal).sum() / n_nonterminal
            loss_cql_q2 = (F.cross_entropy(qs[1][:, :-1].reshape(-1, dsize), actions.reshape(-1), reduction='none').reshape(bsize, ntokens-1) * isterminal).sum() / n_nonterminal
            loss_cql = loss_cql_q1 + loss_cql_q2
        else:
            loss_cql = (F.cross_entropy(qs[:, :-1].reshape(-1, dsize), actions.reshape(-1), reduction='none').reshape(bsize, ntokens-1) * isterminal).sum() / n_nonterminal

        loss_awac = (F.cross_entropy(logits[:, :-1].reshape(-1, dsize), actions.reshape(-1), reduction='none').reshape(bsize, ntokens-1) * isterminal).sum() / n_nonterminal

        loss = loss_q + loss_v + self.cql_scale * loss_cql + self.awac_scale * loss_awac
        stats = {
            k: v for k, v in locals().items() if k in
            ['loss', 'loss_v', 'loss_q', 'loss_cql', 'loss_awac']
        }

        return loss, stats

    def _sync_target_q_heads(self, alpha):
        for target_param, copy_param in zip(self.target_q1_head.parameters(), self.q1_head.parameters()):
            target_param.data.copy_((alpha * copy_param.data) + (1.0 - alpha) * target_param.data)

        if self.two_qs:
            for target_param, copy_param in zip(self.target_q2_head.parameters(), self.q2_head.parameters()):
                target_param.data.copy_((alpha * copy_param.data) + (1.0 - alpha) * target_param.data)

    @th.inference_mode()
    def sample(self, query, beta=1, max_length=32, temperature=1, top_k=20, logit_mask=None, logs=True, eos_token_id=50256):
        input = query.clone()
        past_key_values = None
        tensors = defaultdict(list)

        finished = th.zeros(input.shape[0], 1, dtype=th.long, device=query.device)

        for _ in range(max_length-1):
            logits, _, target_qs, vs, past_key_values = self.forward(input_ids=input, past_key_values=past_key_values)

            if self.two_qs:
                qs = th.minimum(target_qs[0][:, -1], target_qs[1][:, -1])
            else:
                qs = target_qs[:, -1]

            logits = logits[:, -1]

            if logit_mask is not None:
                logits[th.where(logit_mask[input[:, -1]])] = -np.inf

            adv = qs - vs[:, -1, :]
            pi = F.log_softmax(logits, -1)
            modpi = topk_mask(pi + beta * adv, top_k)
            ps = F.softmax(modpi / temperature, -1)

            tokens = th.multinomial(ps, 1)
            tokens = (1 - finished) * tokens + finished * eos_token_id

            query = th.hstack((query, tokens))

            input = tokens
            finished = (tokens == eos_token_id).long()

            if logs:
                tensors['qs'].append(qs)
                tensors['vs'].append(vs)
                tensors['adv'].append(adv)

        stats = {}
        for name, xs in tensors.items():
            xs = th.vstack(xs)
            stats.update({
                f'{name}-min': xs.min(),
                f'{name}-max': xs.max(),
                f'{name}-std': xs.std(),
                f'{name}-avg': xs.mean(),
            })

        return query, stats

    @property
    def dummy_inputs(self):
        return {'input_ids': th.ones(1, 1, device=self.gpt.device, dtype=th.long)}

    @property
    def device(self):
        return self.gpt.device
    
import os
import sys
import yaml
import torch as th
from torch.utils.data import DataLoader
from transformers import GPT2Config
from tqdm import tqdm, trange
import numpy as np

th.set_printoptions(sci_mode=False)

def main(**args):
    task = 'RandomWalks'
    config = yaml.safe_load(open('config.yaml'))[task]
    config.update(args)

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    th.manual_seed(config['seed'])

    data = RandomWalks(seed=config['seed'])
    gptconfig = GPT2Config(**config['gptconfig'], vocab_size=data.n_nodes)
    model = QVModel(gptconfig, config).to(device)

    for m in model.gpt.transformer.h[:-config['n_layers_unfrozen']]:
        m.requires_grad_(False)

    train_dataloader = DataLoader(data.dataset, batch_size=config['batch_size'])
    opt = th.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config['lr'], betas=config['opt_betas'])

    total_steps = int(config['n_epochs'] * (len(data.dataset) // config['batch_size']))
    model.train()

    for _ in trange(total_steps):
        for batch in train_dataloader:
            batch = [item.to(device) for item in batch]
            loss, stats = model.loss(batch)

            opt.zero_grad()
            loss.backward()
            opt.step()

            # if stats:
            #     print(stats)  # Optionally, print stats for each batch

    return model, data

import os
import sys
from torch import tensor
import torch as th
import networkx as nx
import numpy as np
from matplotlib import pyplot
import matplotlib

optimal_lengths = []
sampled_lengths = []
iql_lengths = []

for seed in range(2):
    model, data = main(seed=seed, debug=True)
    model.eval()

    g = nx.from_numpy_array(data.adj, create_using=nx.DiGraph)

    # optimal
    for start in set(range(data.n_nodes)) - {data.goal}:
        try:
            shortest_path = nx.shortest_path(g, start, data.goal)[:data.walk_size]
            optimal_lengths.append(len(shortest_path)-1)
        except:
            optimal_lengths.append(data.walk_size)

    # ilql
    starts = th.arange(1, data.n_nodes).unsqueeze(1).to(model.device)
    paths, _ = model.sample(starts, max_length=data.walk_size, logit_mask=tensor(~data.adj), beta=10) # argmax
    for path in paths:
        length = data.walk_size
        for ind, node in enumerate(path):
            if node == data.goal:
                length = ind
                break

        iql_lengths.append(length)

    # all samples
    for path in data.tensors[0]:
        length = data.walk_size
        for ind, node in enumerate(path):
            if node == data.goal:
                length = ind
                break

        sampled_lengths.append(length)

fontcolor = '#444'
matplotlib.rcParams['text.color'] = fontcolor
matplotlib.rcParams['axes.labelcolor'] = fontcolor
matplotlib.rcParams['xtick.color'] = fontcolor
matplotlib.rcParams['ytick.color'] = fontcolor
matplotlib.rcParams['xtick.labelcolor'] = fontcolor
matplotlib.rcParams['ytick.labelcolor'] = fontcolor
matplotlib.rcParams['xtick.labelcolor'] = fontcolor

matplotlib.rcParams["font.family"] = "Futura"
matplotlib.rcParams["font.size"] = 15
matplotlib.rcParams["xtick.labelsize"] = 20
matplotlib.rcParams["ytick.labelsize"] = 20
matplotlib.rcParams["figure.titlesize"] = 12
matplotlib.rcParams["figure.figsize"] = 15, 8

matplotlib.style.use('ggplot')
matplotlib.rcParams['figure.dpi'] = 70

ax = pyplot.gca()
ax.set_facecolor('#fff')
ax.grid(color='lightgray', alpha=0.4, axis='y')
ax.tick_params(top=False, labeltop=False, bottom=False, labelbottom=True, left=False, labelleft=True)

optimal_hist = np.histogram(optimal_lengths, bins=np.arange(1, data.walk_size+2), density=True)[0]
sampled_hist = np.histogram(sampled_lengths, bins=np.arange(1, data.walk_size+2), density=True)[0]
iql_hist = np.histogram(iql_lengths, bins=np.arange(1, data.walk_size+2), density=True)[0]

barsize = 0.36
iql_color = '#99a3fd'
opt_color = '#f2ad48'
random_color='lightgray'

pyplot.bar(np.arange(1, data.walk_size+1)-barsize/1.5, optimal_hist, width=barsize, label='shortest path', color=opt_color, zorder=2)
pyplot.bar(np.arange(1, data.walk_size+1), iql_hist, width=barsize, label='ILQL', color=iql_color, zorder=3)
pyplot.bar(np.arange(1, data.walk_size+1)+barsize/1.5, sampled_hist, width=barsize, label='random walk', color=random_color, zorder=1)

pyplot.legend(fontsize=16)
pyplot.xticks(np.arange(1, data.walk_size+1), list(np.arange(1, data.walk_size)) + ['∞'])

pyplot.xlabel('# of steps to goal', fontsize=22, color=fontcolor, labelpad=20)
pyplot.ylabel('proportion of paths', fontsize=22, color=fontcolor, labelpad=20)

pyplot.savefig('scripts/graph_plot.svg')