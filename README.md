# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 10:22:34 2019

@author: 25118
"""

import numpy as np

import matplotlib.pyplot as plt


from gaft import GAEngine
from gaft.components import BinaryIndividual, Population
from gaft.operators import RouletteWheelSelection, UniformCrossover, FlipBitMutation
import gaft.analysis.fitness_store as gaf
import math as m
# Analysis plugin base class.
from gaft.plugin_interfaces.analysis import OnTheFlyAnalysis
indv_template = BinaryIndividual(ranges=[(0, 15)], eps=0.001)
population = Population(indv_template=indv_template, size=50)
population.init()  # Initialize population with individuals.
# Use built-in operators here.
selection = RouletteWheelSelection()
crossover = UniformCrossover(pc=0.8, pe=0.5)
mutation = FlipBitMutation(pm=0.1)
engine = GAEngine(population=population, selection=selection,
                  crossover=crossover, mutation=mutation,
                  analysis=[gaf.FitnessStore])
@engine.fitness_register
def fitness(indv):
    x, = indv.solution
    return -3*(x-30)**2*m.sin(x)   #-3*(x-30)**2*sin(x)

@engine.analysis_register
class ConsoleOutput(OnTheFlyAnalysis):
    master_only = True
    interval = 1
    def register_step(self, g, population, engine):
        best_indv = population.best_indv(engine.fitness)
        msg = 'Generation: {}, best fitness: {:.3f}'.format(g, engine.fmax)
        engine.logger.info(msg)
        
if '__main__' == __name__:
    engine.run(ng=100)

x = np.linspace(0,15)
y =  -3*(x-30)**2*np.sin(x)

plt.figure(figsize=(8,4))

plt.plot(x,y,color="red",linewidth=2) 

plt.ylim(-3000,3000)

plt.legend() 

plt.show() 

plt.savefig("sinx.jpg")
