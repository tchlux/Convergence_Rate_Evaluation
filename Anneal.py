# Standard Libraries
import random
from numpy import asarray

# ===================================
#      Low Temperature Annealing     
# ===================================

MIN_EXHAUSTION = 0
EXHAUSTION_LIMIT = 1000

# Pre:  "objective" a numpy array and returns an object that has a
#         comparison operator 
#       "solution" is a valid initial solution for this algorithm (in
#         numpy array format) and can be modified with expected side
#         effects outside the scope of this algorithm 
#       "bounds" is a list of tuples (lower,upper) for each parameter
#       "halt" is a function that takes no parameters and returns
#         "False" when the algorithm should continue and "True" when
#         it should halt
# Post: Returns an (objective value, solution) tuple, best achieved
def Anneal(objective, solution, bounds, halt, 
           min_exhaustion=MIN_EXHAUSTION,
           exhaustion_limit=EXHAUSTION_LIMIT):
    exhaustion = min_exhaustion
    obj_value = objective(solution)
    while not halt():
        # Select a random parameter to modify
        param = random.randint(0,len(solution)-1)
        # Store the old value and generate a new one
        old_value = solution[param]
        stdev = (bounds[param][1] - bounds[param][0]) \
                / max(1, exhaustion)
        new_value = random.gauss(old_value, stdev)
        # Put the new value back into bounds
        new_value = max(new_value, bounds[param][0])
        new_value = min(new_value, bounds[param][1])
        # Check the new solution
        solution[param] = new_value
        new_obj_value = objective(solution)
        # If it is better, keep it and reset exhaustion
        if new_obj_value < obj_value:
            obj_value = new_obj_value
            exhaustion = min_exhaustion
        else: # Otherwise reset the value and increment exhaustion
            exhaustion += 1 
            solution[param] = old_value
        # Check to see if exhaustion has gotten too large
        if exhaustion > exhaustion_limit:
            exhaustion = min_exhaustion
    return (obj_value, solution)
