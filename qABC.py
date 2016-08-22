# For running optimization
import numpy as np

#      qABC Variables     
# ========================
MIN_PROBABILITY = 0.1 # Minimum probability for selecting solutions
RADIUS = 1.0 # The multiplier on the distance between two solutions to
             # be considered in the same neighborhood (1.0 = less than
             # mean distance to all points)

# Pre:  "objective_values" is a numpy array of numbers
# Post: Corresponding fitness numbers that translate to "lower is
#       better" are returned in an array.
def calculate_fitness(objective_values):
    return np.where(objective_values >= 0, 
                    1.0 / (objective_values+1), 1 - objective_values)

# Pre:  "objective" a numpy array and returns an object that has a
#         comparison operator 
#       "solution" is a valid initial solution for this algorithm (in
#         numpy array format) and can be modified with expected side
#         effects outside the scope of this algorithm 
#       "bounds" is a list of tuples (lower,upper) for each parameter
#       "halt" is a function that takes no parameters and returns
#         "True" when the algorithm should halt and "False" otherwise
# Post: Returns an (objective value, solution) tuple, best achieved
def qABC(objective, solutions, bounds, halt, 
         min_probability=MIN_PROBABILITY, radius=RADIUS):
    # Calculate exhaustion limit (as suggested by authors)
    exhaustion_limit = np.prod(solutions.shape)
    # Initialize internal arrays for tracking population
    objective_values = np.array([objective(s) 
                                 for s in solutions])
    solution_trials = np.zeros((solutions.shape[0],1))
    # Track the best solution obtained (for return value)
    best_solution_index = min(range(solutions.shape[0]), 
                              key=lambda i:objective_values[i])
    best_solution = solutions[best_solution_index].copy()
    best_solution_obj = objective_values[best_solution_index]

    # ================================================================
    # Pre:  0 < "index" < solutions.shape[0] (rows in "solutions")
    # Post: The solution at "index" is randomly mutated with a random
    #       neighbor, the better of the mutated and original is kept
    def mutate_solution(index):
        nonlocal solutions, bounds, objective_values, solution_trials
        nonlocal best_solution, best_solution_obj
        # Pick a random neighbor
        nb_index = np.random.randint(solutions.shape[0]-1)
        # Add one to shift the random solution away from "index"
        nb_index += (1 if nb_index >= index else 0)
        # Pick a random parameter to modify (and record the old value)
        param_index = np.random.randint(solutions.shape[1])
        old_value = solutions[index][param_index]
        # Generate a new value for that parameter based off of neighbor
        new_value = ( old_value + np.random.uniform(-1,1) *
                      (solutions[nb_index][param_index] - old_value) )
        # Make sure the param value is within the appropriate bounds
        new_value = max(bounds[param_index][0], 
                          min(bounds[param_index][1], new_value))
        # Calculate new objective value of shifted solution
        solutions[index][param_index] = new_value
        obj_value = objective(solutions[index])
        # Check to see if the solution is better (or best)
        if obj_value < objective_values[index]:
            objective_values[index] = obj_value
            solution_trials[index] = 0
        if obj_value < best_solution_obj:
            best_solution_obj = obj_value
            best_solution = solutions[index].copy()
        else:
            # If not better, reset parameter value and increment trials
            solutions[index][param_index] = old_value
            solution_trials[index] += 1
    # ================================================================

    # Begin the optimization process
    while not halt():        
        # =============================
        #      Employeed Bee Phase     
        # =============================

        # Modify all solutions
        for i in range(solutions.shape[0]):
            if halt(): break
            mutate_solution(i)

        # ============================
        #      Onlooker Bee Phase     
        # ============================

        # Modify solutions proportional to fitness
        fitness = calculate_fitness(objective_values)
        selection_probabilities = ( min_probability + 
                                    (1.0 - min_probability) *
                                    (fitness / np.max(fitness)) )
        solutions_modified = i = 0
        while solutions_modified < solutions.shape[0]:
            if halt(): break
            if np.random.random() < selection_probabilities[i]:
                solutions_modified += 1
                # Get the distances between the current solution and all others
                distances = np.sum((solutions - solutions[i])**2,
                                   axis=1)**0.5
                mean_distance = np.sum(distances) / distances.shape[0]
                # Find the index of the best solution closer than the
                # mean distance away from the currently selected solution
                to_modify = min(range(solutions.shape[0]), 
                                key=lambda s: objective_values[s] 
                                if distances[s] <= radius*mean_distance 
                                else float("inf"))
                mutate_solution(to_modify)
            i += 1
            # Reset "i" if it has gotten too large
            if i == solutions.shape[0]: i = 0            

        # =========================
        #      Scout Bee Phase     
        # =========================

        # Recycle solutions that have been exhausted
        # Default exhaution from author = num solutions * num dimensions
        for i in range(solutions.shape[0]):
            if halt(): break
            # Check to see if that solution is exhausted
            if solution_trials[i] >= exhaustion_limit:
                # Generate a new solution and calculate objective value
                solutions[i] = np.array([np.random.uniform(*b)
                                         for b in bounds])
                objective_values[i] = objective(solutions[i])
                # If this new random solution is best, set it
                if objective_values[i] < best_solution_obj:
                    best_solution_obj = objective_values[i]
                    best_solution = solutions[i].copy()
                # Reset exhaustion back to 0
                solution_trials[i] = 0

    # Final return value of the algorithm
    return (best_solution_obj, best_solution)
            
