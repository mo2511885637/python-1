#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Find the global maximum for function: f(x) = x + 10sin(5x) + 7cos(4x)
'''

import math as m
import numpy as np

import matplotlib.pyplot as plt
from gaft import GAEngine
from gaft.components import BinaryIndividual
from gaft.components import Population
from gaft.operators import TournamentSelection
from gaft.operators import UniformCrossover
from gaft.operators import FlipBitMutation

# Analysis plugin base class.
from gaft.plugin_interfaces.analysis import OnTheFlyAnalysis

# Built-in best fitness analysis.
from gaft.analysis.fitness_store import FitnessStore

# Define population.
indv_template = BinaryIndividual(ranges=[(0, 15)], eps=0.001)
population = Population(indv_template=indv_template, size=30).init()

# Create genetic operators.
selection = TournamentSelection()
crossover = UniformCrossover(pc=0.8, pe=0.5)
mutation = FlipBitMutation(pm=0.1)

# Create genetic algorithm engine.
engine = GAEngine(population=population, selection=selection,
                  crossover=crossover, mutation=mutation,
                  analysis=[FitnessStore])

# Define fitness function.
@engine.fitness_register
def fitness(indv):
    x, = indv.solution
    return  -3*(x-30)**2*m.sin(x) 

# Define on-the-fly analysis.
@engine.analysis_register
class ConsoleOutputAnalysis(OnTheFlyAnalysis):
    interval = 1
    master_only = True

    def register_step(self, g, population, engine):
        best_indv = population.best_indv(engine.fitness)
        msg = 'Generation: {}, best fitness: {:.3f}'.format(g, engine.ori_fmax)
        self.logger.info(msg)

    def finalize(self, population, engine):
        best_indv = population.best_indv(engine.fitness)
        x = best_indv.solution
        y = engine.ori_fmax
        msg = 'Optimal solution(max): ({}, {})'.format(x, y)
        self.logger.info(msg)

if '__main__' == __name__:
    # Run the GA engine.
    engine.run(ng=100)
    
    
    x = np.linspace(0,15)
y =  -3*(x-30)**2*np.sin(x)

plt.figure(figsize=(8,4))

plt.plot(x,y,color="red",linewidth=2) 

plt.ylim(-3000,3000)

plt.legend() 

plt.show() 

plt.savefig("sinx.jpg")


"""
遗传算法：求最有解的算法
  以生物进化的方式来慢慢锁定最优解（及适应度）。
包括：
1、初始化：人口，每个人的染色体（二进制编码，浮点数编码等，解码过程将染色体编码与适应度函数中的变量联系起来）

2、选择：选择当前种群中最优的两个个体（拥有优质基因最多，利于后面遗传算子时产生更优的解）作为父母

3、遗传算子：交叉，变异产生新的总量一致的种群，

4、启发式

其中选择与遗传算子部分使适应度向最优解方向进发，
"""
