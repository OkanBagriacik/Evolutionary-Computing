import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import trange
import time
import random
from deap import base
from deap import creator
from deap import tools
import math

# We want the same result every time that is why we added seed. It doesn't affect the codes functions.
random.seed(1)

# Nmae of the Parameters
PARAM_NAMES = ["enlem", "boylam"]

NGEN = 20 # Number of Genereations
NPOP = 100 # Number of Populations
CXPB = 0.5 # Crossover Rate
MUTPB = 0.3 # Mutation Rate

# Our formula to calculate fitness values of latitude and longitude
def evaluate(individual):

    # We are taking the parameters
    params = {k: v for k, v in zip(PARAM_NAMES, individual)}
    enlem = params["enlem"]
    boylam = params["boylam"]
    # Calculating Fitness Value
    enlem = enlem * math.pi /180
    boylam = boylam * math.pi / 180
    fitness = math.cos(enlem)*math.sin(boylam)-math.cos(enlem+boylam)-math.sin(enlem-boylam)
    return [fitness]
# During Mutation values can go out of our range
# We are checking our parameters are they in the range or not
# If they go out of our range we are putting them back
def checkBounds(min, max):
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                for i in range(len(child)):
                    if child[i] > max[i]: # Parametre max değerden büyük ise max değere eşitle
                        child[i] =  max[i]
                    elif child[i] <  min[i]: # Parametre min değerden küçük ise min değere eşitle
                        child[i] =  min[i]
            return offspring
        return wrapper
    return decorator

# We want the max fitness value
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# We are determining our population size and we want i to start with random variables
toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(NPOP), NPOP)
# Crossover Part
toolbox.register("mate", tools.cxUniform, indpb=CXPB)
# Mutation Part

toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
# We want to take 3 off-spring from each generation
toolbox.register("select", tools.selTournament, tournsize=3)
# Fitness function
toolbox.register("evaluate", evaluate)

# We chose parameters max and min values as Turkey's borders.
MIN = [36, 26]
MAX = [42, 45]
# Each crossover and mutation is in the range or not. We are checking that.
toolbox.decorate("mate", checkBounds(MIN, MAX))
toolbox.decorate("mutate", checkBounds(MIN, MAX))

# Introducing our parameteres
for i, p in enumerate(PARAM_NAMES):
    toolbox.register(p, random.uniform,MIN[i] , MAX[i])

toolbox.register(
    "individual",
    tools.initCycle,
    creator.Individual,
    (
        toolbox.enlem,
        toolbox.boylam,
    ),
)
# Determining our population
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

mean = np.ndarray(NGEN)
best = np.ndarray(NGEN)
hall_of_fame = tools.HallOfFame(maxsize=3)
t = time.perf_counter()
pop = toolbox.population(n=NPOP)
# We are running the algortihm as long as the number of our genereations
for g in trange(NGEN):
    # Creating off-spring for crossover
    offspring = toolbox.select(pop, len(pop))
    # Adding them to population
    offspring = list(map(toolbox.clone, offspring))

    # We are doing crossover on new generations
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    # We are doing mutation on new generations
    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Calculating our new fitness values
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # We are printing 3 most high value off-springs
    pop[:] = offspring
    hall_of_fame.update(pop)
    print(
        "HALL OF FAME:\n"
        + "\n".join(
            [
                f"    {_}: {ind}, Fitness: {ind.fitness.values[0]}"
                for _, ind in enumerate(hall_of_fame)
            ]
        )
    )

    fitnesses = [
        ind.fitness.values[0] for ind in pop if not np.isinf(ind.fitness.values[0])
    ]
    # We are calculating average fitness value of the genereation
    mean[g] = np.mean(fitnesses)
    # determining the most high fitness valeu in this generation
    best[g] = np.max(fitnesses)

# Printing the time
end_t = time.perf_counter()
print(f"Time Elapsed: {end_t - t:,.2f}")
# Plot of best and average values in generations
fig, ax = plt.subplots(sharex=True, figsize=(16, 9))

sns.lineplot(x=range(NGEN), y=mean, ax=ax, label="Average Fitness Score")
sns.lineplot(x=range(NGEN), y=best, ax=ax, label="Best Fitness Score")
ax.set_title("Fitness Score")
ax.set_xticks(range(NGEN))
ax.set_xlabel("Iteration")

plt.tight_layout()
plt.show()