import pygad
import numpy
import time

S = [{'nazwa':'zegar','wartosc':100,'waga':7,},
      {'nazwa':'obraz-pejzaz',
     'wartosc':300,
     'waga':7,},
      {'nazwa':'zegar',
     'wartosc':300,
     'waga':7,},
     {'nazwa':'obraz-portret',
      'wartosc':200,
        'waga':6},
      {'nazwa':'radio',
        'wartosc':40,
        'waga':2},
        {'nazwa':'laptop',
         'wartosc':500,
         'waga':5},
         {'nazwa':'lampka nocna',
            'wartosc':70,
            'waga':6},
          {"nazwa":"srebrne sztućce",
            "wartosc":100,
            "waga":1},
            {'nazwa':'porcelana',
             'wartosc':250,
             'waga':3},
             {'nazwa':'figura z brązu',
              'wartosc':300,
              'waga':10},
              {"nazwa":"skórzana torebka",
               "wartosc":280,
               "waga":3},
               {"nazwa":"odkurzacz",
                "wartosc":300,
                "waga":15}]
wartosci = []
for i in S:
    wartosci.append(i['wartosc'])
print(wartosci)
wagi = []
for i in S:
    wagi.append(i['waga'])
nazwy = []
for i in S:
    nazwy.append(i['nazwa'])

gene_space = [0, 1]


def fitness_func(solution, solution_idx):
    sum1 = numpy.sum(solution * wartosci)
    sum2 = numpy.sum(solution * wagi)

    fitness = sum1 if sum2 <= 25.0 else 0

    return fitness

fitness_function = fitness_func

sol_per_pop = 10
num_genes = len(wartosci)


num_parents_mating = 5
num_generations = 30
keep_parents = 2

parent_selection_type = "sss"

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 10

start=time.time()
ga_instance = pygad.GA(gene_space=gene_space,
                       num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       stop_criteria=["reach_1600"]
                       )

ga_instance.run()
end=time.time()
print("time of algorithm",end-start)

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("generations_completed", ga_instance.generations_completed)
mapped_solution = [nazwy[i] for i, s in enumerate(solution) if s == 1]
print("Parameters of the best solution: {solution}".format(solution=mapped_solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

prediction = numpy.sum(wartosci * solution)
print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))
times = [0.002228975296020508,0.004930973052978516,0.0019788742065429688,0.0034699440002441406,0.0039522647857666016,0.003299236297607422,0.003390789031982422,0.005889892578125,0.00711512565612793,0.0018260478973388672]
print("times of finding solution by algorithm :", times)
print("mean of times :", numpy.average(times))


ga_instance.plot_fitness()

