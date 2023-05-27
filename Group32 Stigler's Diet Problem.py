from typing import List
import random

class Food:
    """
    Represents a food item witha name, cost, and nutritional values.
    """
    def __init__(self, name, cost, nutritional_values):
        self.name = name
        self.cost = cost
        self.nutritional_values = nutritional_values

class Diet:
    """
    Represents a diet plan with a list of food items, total cost, and nutritional values.
    """
    def __init__(self):
        self.items = []
        self.total_cost = 0
        self.total_nutrition = {}
    
    def add_food(self, food: Food):
        """
        Adds a food item to the current diet plan. Updates the total cost and nutritional values.

        Args:
            food (Food): The food item to be added.
        """
        self.items.append(food)
        self.total_cost += food.cost

        for nutrient, value in food.nutritional_values.items():
            if nutrient not in self.total_nutrition:
                self.total_nutrition[nutrient] = value
            else:
                self.total_nutrition[nutrient] += value

class DietIndividual:
    """
    Represents an individual in the Stigler Diet optimization problem.
    Each individual consists of a list of food item selections as their genome.
    """
    def __init__(self, food_selections: List[int]):
        self.food_selections = food_selections

    def __str__(self):
        return repr(self.food_selections)

    def __hash__(self):
        return hash(str(self.food_selections))

    def fitness(self, nutrient_requirements) -> float:
        """
        Calculates the fitness of the individual based on the total cost of food items
        and how well the diet plan meets the minimum nutrient requirements.

        Args:
            nutrient_requirements (dict): A dictionary with nutrient names as keys and their minimum requirements as values.

        Returns:
            float: The fitness value of the individual.
        """
        diet = Diet()

        for index, food in zip(self.food_selections, available_foods):
            if index:
                diet.add_food(food)

        # Calculate fitness based on total cost and how well the diet meets the nutrient requirements
        fitness = diet.total_cost
        for nutrient, requirement in nutrient_requirements.items():
            consumed_nutrient = diet.total_nutrition.get(nutrient, 0)
            if consumed_nutrient < requirement:
                # Penalty for not meeting nutrient requirements
                fitness = fitness * 10 + 99
        return fitness

data = [
    [
        'Wheat Flour (Enriched)', '10 lb.', 36, 44.7, 1411, 2, 365, 0, 55.4,
        33.3, 441, 0
    ],
    ['Macaroni', '1 lb.', 14.1, 11.6, 418, 0.7, 54, 0, 3.2, 1.9, 68, 0],
    [
        'Wheat Cereal (Enriched)', '28 oz.', 24.2, 11.8, 377, 14.4, 175, 0,
        14.4, 8.8, 114, 0
    ],
    ['Corn Flakes', '8 oz.', 7.1, 11.4, 252, 0.1, 56, 0, 13.5, 2.3, 68, 0],
    [
        'Corn Meal', '1 lb.', 4.6, 36.0, 897, 1.7, 99, 30.9, 17.4, 7.9, 106,
        0
    ],
    [
        'Hominy Grits', '24 oz.', 8.5, 28.6, 680, 0.8, 80, 0, 10.6, 1.6,
        110, 0
    ],
    ['Rice', '1 lb.', 7.5, 21.2, 460, 0.6, 41, 0, 2, 4.8, 60, 0],
    ['Rolled Oats', '1 lb.', 7.1, 25.3, 907, 5.1, 341, 0, 37.1, 8.9, 64, 0],
    [
        'White Bread (Enriched)', '1 lb.', 7.9, 15.0, 488, 2.5, 115, 0,
        13.8, 8.5, 126, 0
    ],
    [
        'Whole Wheat Bread', '1 lb.', 9.1, 12.2, 484, 2.7, 125, 0, 13.9,
        6.4, 160, 0
    ],
    ['Rye Bread', '1 lb.', 9.1, 12.4, 439, 1.1, 82, 0, 9.9, 3, 66, 0],
    ['Pound Cake', '1 lb.', 24.8, 8.0, 130, 0.4, 31, 18.9, 2.8, 3, 17, 0],
    ['Soda Crackers', '1 lb.', 15.1, 12.5, 288, 0.5, 50, 0, 0, 0, 0, 0],
    ['Milk', '1 qt.', 11, 6.1, 310, 10.5, 18, 16.8, 4, 16, 7, 177],
    [
        'Evaporated Milk (can)', '14.5 oz.', 6.7, 8.4, 422, 15.1, 9, 26, 3,
        23.5, 11, 60
    ],
    ['Butter', '1 lb.', 30.8, 10.8, 9, 0.2, 3, 44.2, 0, 0.2, 2, 0],
    ['Oleomargarine', '1 lb.', 16.1, 20.6, 17, 0.6, 6, 55.8, 0.2, 0, 0, 0],
    ['Eggs', '1 doz.', 32.6, 2.9, 238, 1.0, 52, 18.6, 2.8, 6.5, 1, 0],
    [
        'Cheese (Cheddar)', '1 lb.', 24.2, 7.4, 448, 16.4, 19, 28.1, 0.8,
        10.3, 4, 0
    ],
    ['Cream', '1/2 pt.', 14.1, 3.5, 49, 1.7, 3, 16.9, 0.6, 2.5, 0, 17],
    [
        'Peanut Butter', '1 lb.', 17.9, 15.7, 661, 1.0, 48, 0, 9.6, 8.1,
        471, 0
    ],
    ['Mayonnaise', '1/2 pt.', 16.7, 8.6, 18, 0.2, 8, 2.7, 0.4, 0.5, 0, 0],
    ['Crisco', '1 lb.', 20.3, 20.1, 0, 0, 0, 0, 0, 0, 0, 0],
    ['Lard', '1 lb.', 9.8, 41.7, 0, 0, 0, 0.2, 0, 0.5, 5, 0],
    [
        'Sirloin Steak', '1 lb.', 39.6, 2.9, 166, 0.1, 34, 0.2, 2.1, 2.9,
        69, 0
    ],
    ['Round Steak', '1 lb.', 36.4, 2.2, 214, 0.1, 32, 0.4, 2.5, 2.4, 87, 0],
    ['Rib Roast', '1 lb.', 29.2, 3.4, 213, 0.1, 33, 0, 0, 2, 0, 0],
    ['Chuck Roast', '1 lb.', 22.6, 3.6, 309, 0.2, 46, 0.4, 1, 4, 120, 0],
    ['Plate', '1 lb.', 14.6, 8.5, 404, 0.2, 62, 0, 0.9, 0, 0, 0],
    [
        'Liver (Beef)', '1 lb.', 26.8, 2.2, 333, 0.2, 139, 169.2, 6.4, 50.8,
        316, 525
    ],
    ['Leg of Lamb', '1 lb.', 27.6, 3.1, 245, 0.1, 20, 0, 2.8, 3.9, 86, 0],
    [
        'Lamb Chops (Rib)', '1 lb.', 36.6, 3.3, 140, 0.1, 15, 0, 1.7, 2.7,
        54, 0
    ],
    ['Pork Chops', '1 lb.', 30.7, 3.5, 196, 0.2, 30, 0, 17.4, 2.7, 60, 0],
    [
        'Pork Loin Roast', '1 lb.', 24.2, 4.4, 249, 0.3, 37, 0, 18.2, 3.6,
        79, 0
    ],
    ['Bacon', '1 lb.', 25.6, 10.4, 152, 0.2, 23, 0, 1.8, 1.8, 71, 0],
    ['Ham, smoked', '1 lb.', 27.4, 6.7, 212, 0.2, 31, 0, 9.9, 3.3, 50, 0],
    ['Salt Pork', '1 lb.', 16, 18.8, 164, 0.1, 26, 0, 1.4, 1.8, 0, 0],
    [
        'Roasting Chicken', '1 lb.', 30.3, 1.8, 184, 0.1, 30, 0.1, 0.9, 1.8,
        68, 46
    ],
    ['Veal Cutlets', '1 lb.', 42.3, 1.7, 156, 0.1, 24, 0, 1.4, 2.4, 57, 0],
    [
        'Salmon, Pink (can)', '16 oz.', 13, 5.8, 705, 6.8, 45, 3.5, 1, 4.9,
        209, 0
    ],
    ['Apples', '1 lb.', 4.4, 5.8, 27, 0.5, 36, 7.3, 3.6, 2.7, 5, 544],
    ['Bananas', '1 lb.', 6.1, 4.9, 60, 0.4, 30, 17.4, 2.5, 3.5, 28, 498],
    ['Lemons', '1 doz.', 26, 1.0, 21, 0.5, 14, 0, 0.5, 0, 4, 952],
    ['Oranges', '1 doz.', 30.9, 2.2, 40, 1.1, 18, 11.1, 3.6, 1.3, 10, 1998],
    ['Green Beans', '1 lb.', 7.1, 2.4, 138, 3.7, 80, 69, 4.3, 5.8, 37, 862],
    ['Cabbage', '1 lb.', 3.7, 2.6, 125, 4.0, 36, 7.2, 9, 4.5, 26, 5369],
    ['Carrots', '1 bunch', 4.7, 2.7, 73, 2.8, 43, 188.5, 6.1, 4.3, 89, 608],
    ['Celery', '1 stalk', 7.3, 0.9, 51, 3.0, 23, 0.9, 1.4, 1.4, 9, 313],
    ['Lettuce', '1 head', 8.2, 0.4, 27, 1.1, 22, 112.4, 1.8, 3.4, 11, 449],
    ['Onions', '1 lb.', 3.6, 5.8, 166, 3.8, 59, 16.6, 4.7, 5.9, 21, 1184],
    [
        'Potatoes', '15 lb.', 34, 14.3, 336, 1.8, 118, 6.7, 29.4, 7.1, 198,
        2522
    ],
    ['Spinach', '1 lb.', 8.1, 1.1, 106, 0, 138, 918.4, 5.7, 13.8, 33, 2755],
    [
        'Sweet Potatoes', '1 lb.', 5.1, 9.6, 138, 2.7, 54, 290.7, 8.4, 5.4,
        83, 1912
    ],
    [
        'Peaches (can)', 'No. 2 1/2', 16.8, 3.7, 20, 0.4, 10, 21.5, 0.5, 1,
        31, 196
    ],
    [
        'Pears (can)', 'No. 2 1/2', 20.4, 3.0, 8, 0.3, 8, 0.8, 0.8, 0.8, 5,
        81
    ],
    [
        'Pineapple (can)', 'No. 2 1/2', 21.3, 2.4, 16, 0.4, 8, 2, 2.8, 0.8,
        7, 399
    ],
    [
        'Asparagus (can)', 'No. 2', 27.7, 0.4, 33, 0.3, 12, 16.3, 1.4, 2.1,
        17, 272
    ],
    [
        'Green Beans (can)', 'No. 2', 10, 1.0, 54, 2, 65, 53.9, 1.6, 4.3,
        32, 431
    ],
    [
        'Pork and Beans (can)', '16 oz.', 7.1, 7.5, 364, 4, 134, 3.5, 8.3,
        7.7, 56, 0
    ],
    ['Corn (can)', 'No. 2', 10.4, 5.2, 136, 0.2, 16, 12, 1.6, 2.7, 42, 218],
    [
        'Peas (can)', 'No. 2', 13.8, 2.3, 136, 0.6, 45, 34.9, 4.9, 2.5, 37,
        370
    ],
    [
        'Tomatoes (can)', 'No. 2', 8.6, 1.3, 63, 0.7, 38, 53.2, 3.4, 2.5,
        36, 1253
    ],
    [
        'Tomato Soup (can)', '10 1/2 oz.', 7.6, 1.6, 71, 0.6, 43, 57.9, 3.5,
        2.4, 67, 862
    ],
    [
        'Peaches, Dried', '1 lb.', 15.7, 8.5, 87, 1.7, 173, 86.8, 1.2, 4.3,
        55, 57
    ],
    [
        'Prunes, Dried', '1 lb.', 9, 12.8, 99, 2.5, 154, 85.7, 3.9, 4.3, 65,
        257
    ],
    [
        'Raisins, Dried', '15 oz.', 9.4, 13.5, 104, 2.5, 136, 4.5, 6.3, 1.4,
        24, 136
    ],
    [
        'Peas, Dried', '1 lb.', 7.9, 20.0, 1367, 4.2, 345, 2.9, 28.7, 18.4,
        162, 0
    ],
    [
        'Lima Beans, Dried', '1 lb.', 8.9, 17.4, 1055, 3.7, 459, 5.1, 26.9,
        38.2, 93, 0
    ],
    [
        'Navy Beans, Dried', '1 lb.', 5.9, 26.9, 1691, 11.4, 792, 0, 38.4,
        24.6, 217, 0
    ],
    ['Coffee', '1 lb.', 22.4, 0, 0, 0, 0, 0, 4, 5.1, 50, 0],
    ['Tea', '1/4 lb.', 17.4, 0, 0, 0, 0, 0, 0, 2.3, 42, 0],
    ['Cocoa', '8 oz.', 8.6, 8.7, 237, 3, 72, 0, 2, 11.9, 40, 0],
    ['Chocolate', '8 oz.', 16.2, 8.0, 77, 1.3, 39, 0, 0.9, 3.4, 14, 0],
    ['Sugar', '10 lb.', 51.7, 34.9, 0, 0, 0, 0, 0, 0, 0, 0],
    ['Corn Syrup', '24 oz.', 13.7, 14.7, 0, 0.5, 74, 0, 0, 0, 5, 0],
    ['Molasses', '18 oz.', 13.6, 9.0, 0, 10.3, 244, 0, 1.9, 7.5, 146, 0],
    [
        'Strawberry Preserves', '1 lb.', 20.5, 6.4, 11, 0.4, 7, 0.2, 0.2,
        0.4, 3, 0
    ],
]

# Nutrient minimums.
nutrients = [
    ['Calories (kcal)', 3],
    ['Protein (g)', 70],
    ['Calcium (g)', 0.8],
    ['Iron (mg)', 12],
    ['Vitamin A (KIU)', 5],
    ['Vitamin B1 (mg)', 1.8],
    ['Vitamin B2 (mg)', 2.7],
    ['Niacin (mg)', 18],
    ['Vitamin C (mg)', 75],
]

nutrient_names = [n[0] for n in nutrients]

nutrient_requirements = {
    nutrient_names[i]: nutrients[i][1] for i in range(len(nutrients))
}

available_foods = [
    Food(
        data[i][0],
        data[i][2],
        {
            nutrient_names[j]: data[i][j + 3] for j in range(len(nutrient_names))
        },
    )
    for i in range(len(data))
]

initial_population_size = 50

GENERATIONS = 500
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.1
ELITISM_RATE = 0.1

def generate_initial_population(count: int) -> List[DietIndividual]:
    """
    Generates an initial population of distinct diet individuals for the
    genetic algorithm optimization process.

    Each diet individual is represented by a binary list where each element
    corresponds to the presence (1) or absence (0) of a specific food
    item in the diet.

    Args:
        count (int): The desired number of diet individuals in the initial
            population.

    Returns:
        List[DietIndividual]: A list of unique diet individuals to use as the
            initial population in the genetic algorithm.
    """
    population = set()

    while len(population) != count:
        food_selections = [
            random.randint(0, 1)
            for _ in available_foods
        ]
        population.add(DietIndividual(food_selections))

    return list(population)

def selection(population: List[DietIndividual]) -> List[DietIndividual]:
    """
    Selects two parent individuals for the genetic algorithm based on their fitness values.
    It chooses the fittest individuals from the first four shuffled members of the population.

    Args:
        population (List[DietIndividual]): The current generation's population.

    Returns:
        List[DietIndividual]: A list containing two parent individuals.
    """
    parents = []
    random.shuffle(population)

    if population[0].fitness(nutrient_requirements) < population[1].fitness(nutrient_requirements):
        parents.append(population[0])
    else:
        parents.append(population[1])

    if population[2].fitness(nutrient_requirements) < population[3].fitness(nutrient_requirements):
        parents.append(population[2])
    else:
        parents.append(population[3])

    return parents

def tournament_selection(population: List[DietIndividual], k: int = 4) -> List[DietIndividual]:
    """
    Selects two parent individuals from the population by running a k-way tournament selection.
    The function picks k random individuals and chooses the two individuals with the lowest fitness values.

    Args:
        population (List[DietIndividual]): The current generation's population.
        k (int): The number of individuals to be selected for the tournament.

    Returns:
         List[DietIndividual]: A list containing two parent individuals.
    """
    
    selected_individuals = random.sample(population,k)
    return sorted(selected_individuals, key=lambda i: i.fitness(nutrient_requirements), reverse=False)[:2]


def crossover(parents: List[DietIndividual]) -> List[DietIndividual]:
    """
    Performs a single-point crossover on the given parents, producing two offspring.
    The crossover point is selected randomly between the first and last indices.

    Args:
        parents (List[DietIndividual]): A list containing two parent individuals.

    Returns:
        List[DietIndividual]: A list of the two offspring individuals.
    """
    N = len(available_foods)

    crossover_point = random.randint(1, N-1)

    child1 = parents[0].food_selections[:crossover_point] + parents[1].food_selections[crossover_point:]
    child2 = parents[1].food_selections[:crossover_point] + parents[0].food_selections[crossover_point:]

    return [DietIndividual(child1), DietIndividual(child2)]

def two_point_crossover(parents: List[DietIndividual]) -> List[DietIndividual]:
    """
    Performs a two-point crossover on the given parents, producing two offspring.
    The crossover points are selected randomly between the first and last indices.

    Args:
        parents (List[DietIndividual]): A list containing two parent individuals.

    Returns:
        List[DietIndividual]: A list of the two offspring individuals.
    """
    N = len(available_foods)

    crossover_point1 = random.randint(1, N - 2)
    crossover_point2 = random.randint(crossover_point1 + 1, N - 1)

    child1 = parents[0].food_selections[:crossover_point1] + parents[1].food_selections[crossover_point1:crossover_point2] + parents[0].food_selections[crossover_point2:]
    child2 = parents[1].food_selections[:crossover_point1] + parents[0].food_selections[crossover_point1:crossover_point2] + parents[1].food_selections[crossover_point2:]

    return [DietIndividual(child1), DietIndividual(child2)]

def mutate(individuals: List[DietIndividual]) -> List[DietIndividual]:
   """
    Performs mutation for a list of individuals.
    The mutation probability is determined by MUTATION_RATE.
    The mutation changes the value of a randomly picked index in the food_selections list.

    Args:
        individuals (List[DietIndividual]): A list of individuals to mutate.

    Returns: List[DietIndividual]: A list of mutated individuals.
    """

   for individual in individuals:
        for i in range(len(individual.food_selections)):
            if random.random() < MUTATION_RATE:
                individual.food_selections[i] = 1 - individual.food_selections[i]

        return individuals
   
def swap_mutation(individual: DietIndividual) -> DietIndividual:
    """
    Performs a swap mutation on a single individual.
    The function randomly chooses two different indices of the food_selections list and swaps their values.

    Args:
        individual (DietIndividual): An individual to mutate.

    Returns:
        DietIndividual: The mutated individual.
    """
    N = len(individual.food_selections)

    mutation_point1 = random.randint(0, N - 1)
    mutation_point2 = random.randint(0, N - 1)

    while mutation_point2 == mutation_point1:
        mutation_point2 = random.randint(0, N - 1)

    individual.food_selections[mutation_point1], individual.food_selections[mutation_point2] = individual.food_selections[mutation_point2], individual.food_selections[mutation_point1]

    return individual

def mutate_v2(individuals: List[DietIndividual]) -> List[DietIndividual]:
    """
    Performs swap mutation for a list of individuals.
    The mutation probability is determined by MUTATION_RATE.

    Args:
        individuals (List[DietIndividual]): A list of individuals to mutate.

    Returns:
        List[DietIndividual]: A list of mutated individuals.
    """
     
    for individual in individuals:
        if random.random() < MUTATION_RATE:
            swap_mutation(individual)
    return individuals


def next_generation(population: List[DietIndividual]) -> List[DietIndividual]:
    """
    Generates the next generation of the population using the tournament selection, crossover, and mutation functions.
    It also applies elitism, retaining the fittest individuals from the current population.

    Args:
        population (List[DietIndividual]): The current generation's population.

    Returns:
        List[DietIndividual]: The next generation's population.
    """
    next_gen = []
    population_size = len(population)
    num_elites = int(population_size * ELITISM_RATE)

    population = sorted(population, key=lambda i: i.fitness(nutrient_requirements), reverse=False) # Lower cost is better

    next_gen.extend(population[:num_elites])

    while len(next_gen) < population_size:
        children = []

        parents = tournament_selection(population)

        if random.random() < CROSSOVER_RATE:
            children = crossover(parents)

        if random.random() < MUTATION_RATE:
            mutate_v2(children)

        next_gen.extend(children)

    return next_gen[:population_size]

def solve_diet_problem(selection_fn, mutation_fn, crossover_fn) -> float:
    """
    Solves the diet optimization problem using a genetic algorithm.
    The solution is based on the defined GENERATIONS, initial_population_size, and other specified parameters.

    Args:
        selection_fn (function): The selection function to use in the genetic algorithm.
        mutation_fn (function): The mutation function to use in the genetic algorithm.
        crossover_fn (function): The crossover function to use in the genetic algorithm.

    Returns:
        float: The fitness value of the best individual (solution) found after the genetic algorithm process.
    """
    population = generate_initial_population(initial_population_size)

    for _ in range(GENERATIONS):
        population = next_generation(population)
        population.sort(key=lambda x: x.fitness(nutrient_requirements))
        
        # Replace selection, crossover, and mutation functions used in next_generation with passed arguments
        parents = selection_fn(population)

        children = []
        if random.random() < CROSSOVER_RATE:
            children = crossover_fn(parents)

        if random.random() < MUTATION_RATE:
            mutation_fn(children)
 
        population.extend(children)

    population.sort(key=lambda i: i.fitness(nutrient_requirements), reverse=False)
    return population[0].fitness(nutrient_requirements)


def experiment(num_runs: int = 20) -> dict:
    """
    Runs the genetic algorithm with all possible combinations of selection, mutation, and crossover techniques.
    Each combination is run for a specified number of times (num_runs) and the mean cost is computed.

    Args:
        num_runs (int, optional): The number of runs for each combination. Defaults to 500.

    Returns:
        dict: A dictionary summarizing the mean cost for each combination of techniques.
    """
    selection_functions = [selection, tournament_selection]
    mutation_functions = [mutate, mutate_v2]
    crossover_functions = [crossover, two_point_crossover]

    results = {}

    config_index = 1  # Add a counter for configurations
    for s_id, selection_fn in enumerate(selection_functions):
        for m_id, mutation_fn in enumerate(mutation_functions):
            for c_id, crossover_fn in enumerate(crossover_functions):

                # Create a unique key for the combination of selection, mutation, and crossover functions
                config_key = f"Configuration {config_index} - {selection_fn.__name__}, {mutation_fn.__name__}, {crossover_fn.__name__}"
                config_index += 1

                total_cost = 0

                # Run the genetic algorithm for each configuration 'num_runs' times
                for i in range(num_runs):
                    # Call the run_genetic_algorithm function with the current combination of functions
                    solution_cost = solve_diet_problem(selection_fn, mutation_fn, crossover_fn)
                    total_cost += solution_cost
                    print(f'[{config_key}] Run: {i + 1}, Cost: {solution_cost}') # Print the progress

                # Compute the mean cost and store it in the results dictionary
                mean_cost = total_cost / num_runs
                results[config_key] = mean_cost

    return results

if __name__ == '__main__':
    # Run the experiment
    experiment_results = experiment()

    # Print the mean cost for each combination of selection, mutation, and crossover techniques
    for config, mean_cost in experiment_results.items():
        print(f"{config}: Mean Cost: {mean_cost}")
   