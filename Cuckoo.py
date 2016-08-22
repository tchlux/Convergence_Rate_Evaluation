import numpy as np
from numpy.random import rand, randn, permutation
import math

#      Cuckoo Variables     
# ==========================
NUM_SOLUTIONS = 20 # Number of solutions to maintain in solution pool
DEFAULT_PA = 0.0 # Max recycle rate of solutions = (1.0 - PA)
DEFAULT_BETA = 2 / 2 # Levy flight control parameter
DEFAULT_L = 0.01 # Levy flight control parameter

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
def Cuckoo(objective, solutions, bounds, halt, pa=DEFAULT_PA,
           beta=DEFAULT_BETA, l=DEFAULT_L):
    # Levy Flight Calculation
    sigma = ( math.gamma(1+beta) * math.sin(math.pi * beta / 2) /
              (  math.gamma( (1+beta) / 2 ) * 
                 beta * 2 ** ((beta-1) / 2)  )) ** ( 1 / beta )
    lower = np.array([l for l,u in bounds]) * np.ones(solutions.shape)
    upper = np.array([u for l,u in bounds]) * np.ones(solutions.shape)
    obj_values = np.array([objective(s) for s in solutions])

    # ================================================================
    # Pre:  "new_solutions" is a list of lists (list of solutions)
    # Post: The best solutions of current and new are greedily chosen
    def keep_best_solutions(new_solutions):
        nonlocal objective, obj_values, solutions
        for i in range(solutions.shape[0]):
            # Calculate new objective value
            new_obj = objective(new_solutions[i])
            # Greedily select the best of current value or new value
            if new_obj <= obj_values[i]:
                obj_values[i] = new_obj
                solutions[i] = new_solutions[i]
    # ================================================================

    # Main computation loop for Cuckoo search
    while not halt():
        #      Levy flights     
        # ======================
        # Get the best solution (for random levy flight global search)
        best = solutions[np.argmin(obj_values)]
        new_sols = solutions.copy()
        # Here we use Levy flights, for standard random walks use step=1
        for sol in range(solutions.shape[0]):
            solution = new_sols[sol]
            # Levy flight calculation
            u = randn(solutions.shape[1]) * sigma
            v = randn(solutions.shape[1])
            step = u / abs(v)**(1/beta)
            stepsize = 0.01 * step * (solution - best)
            # Final new location is old value plus the step
            solution += stepsize * randn(solutions.shape[1])
        # Make sure the new solutions are within bounds
        new_sols[lower > new_sols] = lower[lower > new_sols]
        new_sols[upper < new_sols] = upper[upper < new_sols]

        # Keep the best of the new solutions and the old
        keep_best_solutions(new_sols)
        # Check halting condition before doing another search
        if halt(): break

        #      Local Random Walks     
        # ============================
        # Generate new solutions with local random walks
        # Randomly decide which nests are going to be recycled
        to_recycle = np.ones(solutions.shape)
        to_recycle = (to_recycle.T * (rand(solutions.shape[0]) > pa)).T
        # Use randomly chosen solutions to generate random walks
        stepsize = rand() * (
            solutions[permutation(solutions.shape[0])] -
            solutions[permutation(solutions.shape[0])])
        # Calculate new solutions, a random walk from current
        new_sols = solutions + stepsize * to_recycle
        # Box bound all solutions
        new_sols[lower > new_sols] = lower[lower > new_sols]
        new_sols[upper < new_sols] = upper[upper < new_sols]

        # Keep the best of the new solutions and the old
        keep_best_solutions(new_sols)

    # Get the best solution for return
    solution = solutions[ np.argmin(obj_values) ]
    obj_value = min(obj_values)
    return (obj_value, solution)
