# ====================================================
#      Backtracking Search optimization Algorithm     
# ====================================================

# Cite this algorithm as: [1] P. Civicioglu, "Backtracking Search
# Optimization Algorithm for numerical optimization problems", Applied
# Mathematics and Computation.

import numpy as np
from numpy import ceil
from numpy.random import rand, randint, shuffle, permutation, randn

DEFAULT_MIXRATE = 1.0
DEFAULT_NUM_SOLUTIONS = 30

# Pre:  "objective" takes a numpy array and returns an object that has
#         a comparison operator (optional arguments as well)
#       "solution" is a valid initial solution for this algorithm (in
#         numpy array format) and can be modified with expected side
#         effects outside the scope of this algorithm 
#       "bounds" is a list of tuples (lower,upper) for each parameter
#       "halt" is a function that takes no parameters and returns
#         "False" when the algorithm should continue and "True" when
#         it should halt
#       "objargs" is a set of arguments to be passed to the objective
#         function after the first parameter (a solution)
# Post: Returns an (objective value, solution) tuple, best achieved
def BSA(objective, solutions, bounds, halt,
        num_solutions=DEFAULT_NUM_SOLUTIONS,
        mixrate=DEFAULT_MIXRATE, *objargs):
    obj_vals = np.array([objective(s) for s in solutions])
    old_solutions = solutions.copy()

    # Main optimization loop
    while not halt():
        if rand() < rand():
            old_solutions = solutions.copy()
            
        # Shuffle the old solutions
        shuffle(old_solutions)
        # Get scale factor
        F = 3 * randn()

        #      Selection 1     
        # =====================
        # Defined in the BSA research paper, it selects a group of
        # columns and rows that will be modified to create the next
        # batch of solutions
        shape = solutions.shape
        mask = np.zeros(shape)
        if rand() < rand():
            for row in range(shape[0]):
                p = permutation(shape[1])
                cols = p[:ceil(mixrate * rand() * shape[1])]
                mask[row,cols] = 1
        else:
            for row in range(shape[0]):
                col = randint(0,shape[1]-1) if (shape[1] > 1) else 0
                mask[row,col] = 1

        # Recombination (Mutation and Crossover)
        new_solutions = solutions + (mask * F) * \
                        (old_solutions - solutions)
        # Put the new solutions into bounds
        bound(new_solutions, bounds) 

        #      Selection 2     
        # =====================
        # Greedily keep the better of current solutions and new
        new_obj_vals = np.array([objective(s) for s in new_solutions])
        better_ind = new_obj_vals < obj_vals
        obj_vals[better_ind] = new_obj_vals[better_ind]
        solutions[better_ind] = new_solutions[better_ind]

    # Return the best solution obtained
    best_obj = min(obj_vals)
    best_sol = solutions[np.argmin(obj_vals)]
    return (best_obj, best_sol)

# Pre:  "solutions" is of shape (num_solutions, num_params)
#       "bounds" is a list of tuples of [(lower bound, upper bound)]
# Post: The parameter values in each solution are put into bounds
def bound(solutions,bounds):
    for row in range(solutions.shape[0]):
        for col in range(solutions.shape[1]):
            lower = bounds[col][0]
            upper = bounds[col][1]
            if rand() < rand():
                if solutions[row,col] < lower:
                    solutions[row,col] = lower
                elif solutions[row,col] > upper:
                    solutions[row,col] = upper
            elif (solutions[row,col] < lower or
                  solutions[row,col] > upper):
                solutions[row,col] = lower + rand() * (upper - lower)
    return solutions
