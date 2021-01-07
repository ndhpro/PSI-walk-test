import sys
import time
from copy import copy, deepcopy
import networkx as nx
import pandas as pd
from env import Environment
from agent import QLearningTable
from pathlib import Path
from tqdm import tqdm


def load_graph(path):
    G = nx.MultiDiGraph()
    with open(path, 'r') as f:
        lines = f.readlines()

    for line in lines[2:]:
        line = line.strip()
        if ' ' in line and line.find(' ') == line.rfind(' '):
            (u, v) = line.split(' ')
            G.add_edge(u, v)

    # Add super_root
    child = set()
    for u, v in G.edges():
        if u != v:
            child.add(v)
    start_edge = list()
    for n in G.nodes():
        if len(child) == len(G.nodes) or n not in child:
            start_edge.append(('ndhpro', n))

    G.add_edges_from(start_edge)

    return G


def update(num_episode, env, RL, steps, all_costs):
    for e in tqdm(range(num_episode), desc='Training agent'):
        i = 0
        cost = 0
        done = False
        state = env.reset()

        while not done:
            avail_action = env.get_avail_action(state)

            action = RL.choose_action(str(state), avail_action)
            state_, reward, done = env.step(action, state)

            cost += RL.learn(str(state), action,
                             reward, str(state_))

            state = deepcopy(state_)
            # print(state['node'], end=' ')

            i += 1
            if i == 50:
                break
        all_costs.append(cost)
        steps.append(i)
        # print()

        if (e+1) % (num_episode//9) == 0:
            RL.reduce_epsilon()


def get_final_path(path, env, RL, steps, all_costs):
    f_path = Path('results/') / (Path(path).stem + '.txt')
    Q = RL.get_q_table()
    # print('Length of full Q-table =', len(Q.index))
    # print('Full Q-table:')
    # print(Q)

    state = env.reset()
    i = 0
    done = 0
    walk = ''
    while not done and i < 50:
        avail_actions = env.get_avail_action(state)
        try:
            state_action = Q.loc[str(state), avail_actions]
        except Exception as e:
            print(e)
            break
        action = state_action.idxmax()
        state, _, done = env.step(action, state)
        walk += state['node'] + ' '
        if (state['node'], state['node']) in env.graph.edges():
            walk += state['node'] + ' '
        i += 1

    # RL.plot_results(steps, all_costs)
    return walk


def generate_walk(path):
    G = load_graph(path)
    if len(G.edges()) == 0:
        print('Graph has no edge')
        return
    # print(len(G.nodes()), len(G.edges()), end=' ')
    env = Environment(graph=G, root='ndhpro')
    RL = QLearningTable(actions=G.nodes())
    steps = []
    all_costs = []

    # t = time.time()
    n_epoch = 1000
    update(n_epoch, env, RL, steps, all_costs)
    # print(round((time.time()-t), 2))

    walk = get_final_path(path, env, RL, steps, all_costs)
    return walk
