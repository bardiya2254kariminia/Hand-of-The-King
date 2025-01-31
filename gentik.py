import torch
import torch.nn
from training_ai import Networks
from utils.classes import Card, Player
from training_ai.Networks import Qnetwork
import random
import os
import utils.training_utils
import random


m = Qnetwork()
m = utils.training_utils.load_weigths(m)
m.eval()
li = []
li.append(m)
for i in range(9):
    li.append(utils.training_utils.mutation(m))
    li[-1].eval()

for gen in range(4):
    next_gen = utils.training_utils.selection(
        population_models=li, primal_model=m)
    for i in range(5):
        parents = random.sample(li, 2)
        next_gen.append(utils.training_utils.crossover(parents[0], parents[1]))
    li.clear()
    for i in next_gen:
        li.append(utils.training_utils.mutation(i))

mx = -1
best = Qnetwork()
for i in li:
    sc = utils.training_utils.fittness(i)
    if sc > mx:
        mx = sc
        best = i
utils.training_utils.save_weights(best)
