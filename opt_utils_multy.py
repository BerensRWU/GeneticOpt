import numpy as np
from abc import ABC, abstractmethod
from graph_tool.topology import shortest_path
import copy

class Individual(ABC):
    def __init__(self, value=None, init_params=None, graph = None):
        self.graph = graph
        if value is not None:
            self.value = value
            self.n_sensors = value.shape[1]
        else:
            self.n_sensors = init_params["n_sensors"]
            self.value = self._random_init(init_params)
        if self.value.shape != (4, self.n_sensors):
            raise ValueError

    @abstractmethod
    def pair(self, other, pair_params):
        pass

    @abstractmethod
    def mutate(self, mutate_params):
        pass

    @abstractmethod
    def _random_init(self, init_params):
        pass

class Optimization(Individual):
    def pair(self, other, pair_params):
        new_value = np.zeros((4,self.n_sensors))
        for i in range(self.value.shape[1]):
            path_v, _ = shortest_path(self.graph,self.graph.vertex(self.value[0,i]),self.graph.vertex(other.value[0,i]))
                
            if len(path_v):
                new_value[0,i] = int(path_v[int(len(path_v)/2)])
            else:
                new_value[0,i] = copy.deepcopy(self.value[0,i])
        new_value[1:4] = pair_params['alpha'] * self.value[1:4] + (1 - pair_params['alpha']) * other.value[1:4]
        return Optimization(new_value, graph = self.graph)

    def mutate(self, mutate_params, epoch):
        self.value[1:4] += np.array([np.random.normal(0, mutate_params['rate_rot'](epoch), self.n_sensors),
                                     np.zeros(self.n_sensors),
                                     np.random.normal(0, mutate_params['rate_rot'](epoch), self.n_sensors)])
        
        for i in range(self.value.shape[1]):
        
            v = int(self.graph.vertex(self.value[0,i]))
            for j in range(np.random.randint(0,mutate_params['rate_loc'])):
                neighbors = list(self.graph.vertex(v).all_neighbors())
                v = int(self.graph.vertex(neighbors[np.random.randint(0,len(neighbors))]))
                
            self.value[0,i] = int(v)
            if not mutate_params['constrains_pitch'](self.value[1,i]):
                self.value[1,i] = min(max([mutate_params['lower_bound_pitch'], self.value[1,i]]), mutate_params['upper_bound_pitch'])
            if self.value[2,i] != 0:
                self.value[2,i] = 0
            if not mutate_params['constrains_roll'](self.value[3,i]):
                self.value[3,i] = min(max([mutate_params['lower_bound_roll'], self.value[3,i]]), mutate_params['upper_bound_roll'])
        
    def _random_init(self, init_params):
        return np.array([np.random.randint(0, self.graph.num_vertices(), init_params['n_sensors']),
                np.random.uniform(init_params['lower_bound_pitch'], init_params['upper_bound_pitch'], init_params['n_sensors']),
                np.zeros(init_params['n_sensors']),
                np.random.uniform(init_params['lower_bound_roll'], init_params['upper_bound_roll'], init_params['n_sensors'])])

class Population:
    def __init__(self, size, fitness, individual_class, init_params, sensor, graph):
        self.fitness = fitness
        self.graph = graph
        self.individuals = [individual_class(init_params=init_params, graph = self.graph) for _ in range(size)]
        self.individuals.sort(key=lambda x: self.fitness(x, sensor, self.graph, 0))
        
    def replace(self, new_individuals, sensor, epoch):
        
        size = len(self.individuals)
        self.individuals.extend(new_individuals)
        fitness = [self.fitness(x, sensor, self.graph, epoch) for x in self.individuals]
        self.individuals = [x for _, x in sorted(zip(fitness, self.individuals))]
        
        a = [x for _, x in sorted(zip(fitness, np.arange(len(self.individuals))))]
        a = np.array(a)
        if (a[-size:] >= size).any():
            print("Added something new", a[-size:][a[-size:] >= size])
        else:
            print("Nothing changed")
        
        self.individuals = self.individuals[-size:]
        
        return self.fitness(self.individuals[-1], sensor, self.graph, epoch)
        
    def get_parents(self, n_offsprings):
        mothers = self.individuals[-2 * n_offsprings::2]
        fathers = self.individuals[-2 * n_offsprings + 1::2]
        return mothers, fathers
        
        
class Evolution:
    def __init__(self, pool_size, fitness, individual_class, n_offsprings, pair_params, mutate_params, init_params, sensor, graph):
        self.pair_params = pair_params
        self.mutate_params = mutate_params
        self.graph = graph
        self.pool = Population(pool_size, fitness, individual_class, init_params, sensor, self.graph)
        self.n_offsprings = n_offsprings

    def step(self, sensor, epoch):
        mothers, fathers = self.pool.get_parents(self.n_offsprings)
        offsprings = []
        for mother, father in zip(mothers, fathers):
            offspring = mother.pair(father, self.pair_params)
            offspring.mutate(self.mutate_params, epoch)
            offsprings.append(offspring)
            
        for mother1, mother2 in zip(mothers, mothers):
            offspring = mother1.pair(mother2, self.pair_params)
            offspring.mutate(self.mutate_params, epoch)
            offsprings.append(offspring)
            
        for father1, father2 in zip(fathers, fathers):
            offspring = father1.pair(father2, self.pair_params)
            offspring.mutate(self.mutate_params, epoch)
            offsprings.append(offspring)
            
        return self.pool.replace(offsprings, sensor, epoch)





































