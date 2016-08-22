import os, pickle, time
from itertools import combinations


# Pre:  "data" is a python object, "string" is a string without any
#       illegeal characters for filenames in it
# Post: A pickle file with this local machine's hostname is saved to
#       the current working directory
def save_local(data, string="out", folder="./"):
    filename = folder + os.uname().nodename + "_" + string + ".pkl"
    with open(filename, "wb") as f:
        pickle.dump(data, f)

# Pre:  "iterable" is a list type, "value" is an iterable whose
#       elements have comparison operators, "key" is a function to use
#       for sorting
# Post: A binary search over the iterable is performed and "value" is
#       inserted in a sorted order left of anything with equal key,
#       "key" defaults to sorting by the last element of "value"
def insert_sorted(iterable, value, key=lambda i:i[-1]):
    low = 0
    high = len(iterable)
    index = (low + high) // 2
    while high != low:
        if key(iterable[index]) >= key(value):   high = index
        elif key(iterable[index]) < key(value):  low = index + 1
        index = (low + high) // 2
    iterable.insert(index, value)

# Pre:  "numbers" is a python list of numbers
# Post: All of the values in "numbers" multiplied together
def mult_sum(numbers):
    value = 1
    for n in numbers:
        value *= n
    return value

# Pre:  "indexable" is an object that can be index accessed
#       "indexable" must be in sorted order
#       "value" is a value that can be compared with values in "indexable"
# Post: The number of elements in "indexable" that are greater than "value"
def count_greater(indexable, value):
    low = 0
    high = len(indexable)
    index = (low + high) // 2    
    while high != low:
        if   indexable[index] >  value:
            high = index
        elif indexable[index] <= value:  
            low  = index + 1
        else:
            string = "High: %i\tLow: %i\tIndex: %i\n"%(high,low,index)
            string += "Indexable:\n%s"%str(indexable)
            raise(Exception("Encountered invalid number in "+
                            "comparison.\n\n"+string))
        index = (low + high) // 2
    return len(indexable) - index

# Pre:  "interest" is an iterable of length > 0
#       "other_data" is a list of iterables each of length > 0
#       "rank" is a 0-indexed position in an ordering, 0'th being best
# Post: The probability that if random single elements are selected
#       from each of the sets in "other_data" and "interest", that the
#       value selected from "interest" will be in position "rank" from
#       the smallest value
def rank_probability(interest, other_data, rank=0):
    # Prepare all the data in tuple format (less memory) make sure
    # that all lists in other data are sorted for improved runtime
    interest = tuple(interest)
    other_data = tuple(list(d) for d in other_data)
    for l in other_data: l.sort()
    other_data = tuple(tuple(d) for d in other_data)
    # Initialize list for holding all probabilities    
    probabilities = []
    # Calculate the total different selections possible
    total = mult_sum(tuple(len(d) for d in other_data))
    # Cycle trough and calculate a probability that each value in
    # "interest" has exactly "rank" greater than it

    for value in interest:
        # The probability that value from one list is better than current value
        greater = [count_greater(d, value) for d in other_data]
        if rank == 0:
            num_greater = mult_sum(greater)
        else:
            # Initialize the number greater to 0
            num_greater = 0
            # Get (in terms of indices) all combinations of the
            # other data sets that could be less than value
            for s in combinations(range(len(other_data)), rank):
                # Get the substitued "lesser" lengths of the selected indices
                lesser = [len(other_data[index]) - greater[index] 
                          for index in s]
                # Temporarily remove those elements from the list of lengths
                temp = [greater.pop(index) for index in s[::-1]][::-1]
                # Count up the number of possible unique sets
                num_greater += mult_sum(lesser + greater)
                # Reinsert the temporarily popped values back
                for i,index in enumerate(s):
                    greater.insert(index,temp[i])
        # Append the probability for this value to the list
        probabilities.append( num_greater / total )
    # Return the average of all probabilities (assuming each value is
    # equally likely to be selected from interest set)
    return sum(probabilities) / len(probabilities)

# Pre:  "interest" is an iterable of length > 0
#       "other_data" is a list of iterables each of length > 0
#       ** The minimum values at each index in "interest" and the
#          sub-sequences of "other_data" MUST NOT BE 0
# Post: The ratio defined in More and Wild in "Benchmarking
#       Derivative-Free Optimization Algorithms" (2009) as a
#       subsequent to performance profiles and data profiles.
def performance_ratio(interest, other_data):
    absolute_mins = [min(o) for o in zip(*(other_data+[interest]))]
    return [ i / m for i,m in zip(interest, absolute_mins) ]

# Pre:  "performance" is a list of performance ratios (numbers) where
#       smaller values denote better performances
#       "to_beat" is the performance ratio that we want to measure
#       against and guage percentage passed
# Post: The percentage of performances that were better than "to_beat"
def performance_profile(performance, to_beat):
    return sum(p <= to_beat for p in performance) / len(performance)

# Pre:  "performance" is a 2D python list of floats of performances
#       [iterations [trials]], where lower is better.
#       "best" is a 1D python list of floats of the absolute minimum
#       value that could be seen in trials for a given iteration
#       "to_beat" is a list of percentages that represent the maximum
#       distance from convergence allowed after n iterations for an
#       algorithm to be considered "passing"
# Post: The data profile for the performances given is returned in
#       terms of a dictionary as defined below
#       {max tolerance: [percentage passed per iteration]}
def data_profile(performances, best, to_beat=[10**(-i) for i in (1,3,5,7)]):
    profile = {t:[] for t in to_beat}
    for trials,abs_min in zip(performances,best):
        for t in to_beat:
            passed = sum( performances[0][i] - trials[i] >= 
                          (1 - t)*(performances[0][i] - abs_min) 
                          for i in range(len(trials)) ) / len(trials)
            profile[t].append( passed )
    return profile

# ====================================================================
#      Class for telling when a random sequence converges in mean     
# ====================================================================
try:
    import numpy as np

    class MeanTracker:
        def __init__(self, min_length=500, window=100, min_confidence=0.001):
            # Check for user
            self.converged = False
            # Internal parameter storage
            self.window = window
            self.min_length = min_length
            self.min_confidence = min_confidence
            # Internal records for monitoring mean shift
            self.values = np.array([])
            self.mean = np.array([])
            self.shift = None
            self.divisor = None
            # Functions for adding values and updating updating the mean
            self.pushv = lambda v: \
                         np.concatenate( (self.values, np.array([v])) )
            self.pushm = lambda: np.concatenate( (self.mean, 
                                        np.array([self.values.mean()])) )

        # Pre:  "value" is a number, "len(self.values)" < 2
        # Post: The new value is added to internal records, for length = 2
        #       a shift and divisor are calculated to normalize the data
        def base_case(self, value):
            # Base cases, when the list of values is still small
            if len(self.values) == 1:
                self.values = self.pushv(value)
                self.mean = self.pushm()
                # Declare new shift
                self.shift = min(self.values)
                # Shift all records accordingly
                self.values -= self.shift
                self.mean -= self.shift
                # Declare new divisor
                self.divisor = max(self.values)
                # Special case for receiving multiple of the same
                # value from the objective function
                if self.divisor == 0:
                    self.divisor = 1.0
                # Divide all values accordingly
                self.values /= self.divisor
                self.mean /= self.divisor
            # Only one value, just store
            if len(self.values) == 0:
                self.values = self.pushv(value)
                self.mean = self.pushm()                

        # Pre:  "value" is a number
        # Post: "value" is stored in an internal array, the mean of all
        #       values appended to this point is updated, if the sequences
        #       of values has converged given initial criteria then
        #       "self.converged" is set to True
        def append(self, value, display=False):
            # Base Case
            if len(self.values) < 2:
                self.base_case(value)
                return
            # New min, need to update shift amount for normalization
            if value < self.shift:
                # Update the shift on all records
                update = (self.shift - value) / self.divisor
                self.values += update
                self.mean += update
                self.shift = value
                # Update the divisor which was shifted to be wrong
                new_divisor = self.divisor + update
                self.values *= (self.divisor / new_divisor)
                self.mean *= (self.divisor / new_divisor)            
                self.divisor = new_divisor
            # Shift the new value into normal range
            value -= self.shift
            # New max, need to update divisor for normalization
            if value > self.divisor:
                self.values *= (self.divisor / value)
                self.mean *= (self.divisor / value)
                self.divisor = value
            # Divide the new value to normal range
            value /= self.divisor
            # Push the normalized value into records
            self.values = self.pushv(value)
            # Push the mean of the of the normalized data set into records
            self.mean = self.pushm()
            # Check convergence criteria
            max_value = max(self.mean[-self.window:])
            min_value = min(self.mean[-self.window:])
            if (len(self.values) > self.min_length and
                max_value - min_value <= self.min_confidence):
                self.converged = True
            if display:
                print("%i: %0.3f"%( len(self.values), 
                                    max_value - min_value ), end="\r")
            return self.converged        

except:
    class MeanTracker:
        def __init__(self,a,b):
            print("Cannot use mean tracker without numpy installed.")

# ==========================================================================
#      Class for tracking the objective use of an optimization alorithm     
# ==========================================================================
try:
    from tl_multi import simple_multi
except:
    print("WARNING: Could not import multi processing library.")

class TrialTracker:
    def __init__(self, opt_alg, display=True):
        self.opt_alg = opt_alg
        self.display = display

    def track_trials(self, num_trials, obj_func, init_solutions,
                     bounds, steps, *obj_args): 
        if len(init_solutions) != num_trials:
            print("Must use same number of initial solutions as trials.")
            return
        trackers = [Tracker(self.opt_alg, self.display) 
                    for i in range(num_trials)]
        func_calls = [ lambda: trackers[i].track(obj_func, init_solutions[i],
                                                 bounds, steps, *obj_args)  
                       for i in range(num_trials) ]
        try:
            returns = simple_multi(func_calls)
        except:
            raise(Exception("ERROR: No multi processing module."))
        self.results = zip(*returns)
        return self.results

class Tracker:
    # Pre:  "opt_alg" is a optimization algorithm that takes
    #       parameters in the order: objective function, initial
    #       solution, bounds, halt callback, (extra arguments)
    # Post: Class attributes initialized
    def __init__(self, opt_alg, display=True):
        self.timer = None
        self.display = display
        self.optimizer = opt_alg
        self.objective = lambda s: None
        self.obj_args = ()
        self.steps = self.curr_step = 0
        self.objective_use = []

    # Pre:  "solution" is a valid solution to give to "obj_func",
    #       "obj_args" has been initialized
    # Post: The objective value of the given solution is returned
    #       after being stored in a class attribute list
    def _obj_tracker(self, solution):
        # If the algorithm has gone past halt, return "infinity"
        if len(self.objective_use) > self.steps: return float("inf")
        # Noral execution
        self.timer.step(display=self.display)
        obj_value = self.objective(solution, *self.obj_args)
        self.objective_use.append(obj_value)
        return obj_value

    # Pre:  Called
    # Post: True if this function has been called "self.steps" times,
    #       False otherwise 
    def _halt(self):
        return len(self.objective_use) >= self.steps

    # Pre:  "obj_func" is a function that takes a solution and
    #       "obj_args" and returns an objective value float,
    #       "init_solution" is a valid solution to "obj_func",
    #       "bounds" is a list of tuples of [(lower,upper)] bounds of
    #       solution space for "obj_func",
    #       "steps" is an integer number of steps the algorithm should
    #       be allowed to run
    #       "obj_args" are optional arguments required to be sent to
    #       the objective function after a solution
    # Post: The solution after "steps" have been taken, and the list
    #       of objective values obtained as well
    def track(self, obj_func, init_solution, bounds, steps, *obj_args):
        self.steps = steps
        self.timer = Timer(steps)
        self.objective = obj_func
        self.obj_args = obj_args
        # Run the optimization algorithm using internal halt
        # mechanism, track its performance along the way
        obj, sol = self.optimizer(self._obj_tracker, init_solution,
                                  bounds, self._halt) 
        return obj, sol, self.objective_use


# =====================================================
#      Timer class for presenting approximated ETA     
# =====================================================

# This class can be used for estimating and printing the approximated
# ramining time of a task
class Timer:
    # Pre:  "total_steps" is a natural number if it is provided
    # Post: The start time is saved, total steps and current step set
    def __init__(self, total_steps=float("inf")):
        self.reset(total_steps)

    # Pre:  "total_steps" is a natural number if it is provided
    # Post: The start time is saved, total steps and current step set
    def reset(self, total_steps=float("inf")):
        self.start = time.time()
        self.current_step = 0
        self.total_steps = total_steps

    # Pre:  This timer has been initialized or reset at start time,
    #       during initialization the number of total steps was
    #       declared
    #       "percentage" is a number in the range [0.0,1.0]
    # Post: The number of current steps is incremented and the
    #       estimated time remaining given percent progress is printed
    # Post: The estimated time remaining given the Timer's start
    #       (either from initialization or "reset") and the percentage
    #       complete is returned. Just uses linear approximation.
    def step(self, percentage=None, display=True):
        if (percentage == None):
            self.current_step += 1
            percentage = self.current_step / self.total_steps

        if percentage == 0: percentage = 0.0001
        percentage = 100.0 * percentage
        etr =  ( (time.time() - self.start) / percentage) \
               * (100.0 - percentage)
        if display:
            print("(%0.1f%%) with [%0.fs] left"%
                  (percentage, etr), end="\r")
        return self.current_step >= self.total_steps


# ======================================================
#      Plotting code that uses matplotlib and numpy     
# ======================================================

try:
    # Plotting libraries
    import matplotlib
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import colors
    import matplotlib.pyplot as plt
    # Numerical libraries
    import numpy
    from numpy import asarray, atleast_1d, linspace, meshgrid, zeros

    # Pre:  "benchmark_fxn" is a Benchmark class instance
    # Post: A plot image is saved to a "plots" folder in the CWD
    def plot_benchmark(bnchmrk_fxn, set_title=False):
        # Get the name of this class
        name = bnchmrk_fxn.__class__.__name__

        # If this function cannot be plotted, exit
        # TODO:  Add the ability to reduce problems that can be reduced
        if bnchmrk_fxn.dimensions > 2:
            print("Error: Cannot plot %s, too many dimensions."%(name))
            return

        fig = plt.figure()
        if bnchmrk_fxn.custom_bounds:
            bounds = bnchmrk_fxn.custom_bounds
        else:
            bounds = bnchmrk_fxn.bounds
        xmin, xmax = bounds[0]
        # Prevent plots from being larger that (-100,100)
        if xmin < -1e4: xmin, xmax = -100, 100
        # Set plot spacing
        spacing = bnchmrk_fxn.spacing
        # Generate a sample x-point space
        X = linspace(xmin, xmax, spacing)
        # For plotting three dimensional functions
        if bnchmrk_fxn.dimensions == 2:
            # 3D functions, create a plot inside of figure
            ax = Axes3D(fig)
            # Get the y-bounds of this function
            ymin, ymax = bounds[1]
            if ymin < -1e4: ymin, ymax = -100, 100
            # Generate a sample y-point space
            Y = linspace(ymin, ymax, spacing)
            # Create a meshgrid for plotting
            X, Y = meshgrid(X, Y)
            # Generate a holder for the Z values
            Z = zeros(X.shape)
            # Calculate the objective function values
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Z[i,j] = bnchmrk_fxn.evaluator(asarray([X[i,j], Y[i,j]]))
            # Get the index of the minimum point
            min_index = Z.argmin()
            # Create a color map
            cmap = matplotlib.cm.jet

            # If there are any Nan values we have a special case
            if numpy.any(numpy.isnan(Z)):
                # Create a masked array without any "nan" values
                Z = numpy.ma.array(Z, mask=numpy.isnan(Z))
                # Create the scale of different color value numbers
                lev = linspace(Z.min(), Z.max(), bnchmrk_fxn.spacing)
                # Normalize that linear space of colors to integers
                norml = colors.BoundaryNorm(lev, 256)
                ax.plot_surface(X, Y, Z, rstride=1, cstride=1, 
                                cmap=cmap, linewidth=0.0, shade=True, 
                                norm=norml)
            else: # No nan, this plot is ready!
                ax.plot_surface(X, Y, Z, rstride=1, cstride=1, 
                                cmap=cmap, linewidth=0.0, shade=True)

            # Set the labels for each axis individually
            ax.set_xlabel(r'$x_1$', fontsize=16)
            ax.set_ylabel(r'$x_2$', fontsize=16)
            ax.set_zlabel(r'$f(x_1, x_2)$', fontsize=16)

        else:
            # 2D functions
            ax = fig.add_subplot(111)
            # Initialize an array for holding the Y values
            Y = zeros(X.shape)
            # Cycle through X values and get associated Y values
            for i in range(X.shape[0]):
                Y[i] = bnchmrk_fxn.evaluator(atleast_1d(X[i]))
            # Plot the objective function line
            ax.plot(X, Y, 'b-', lw=1.5, zorder=30)
            # Get the location of the global min and put a red dot on it
            xf, yf = bnchmrk_fxn.global_optimum, bnchmrk_fxn.fglob
            ax.plot(xf, yf, 'r.', ms=11, zorder=40)
            # Add a grid to the back and set axis labels
            ax.grid()
            ax.set_xlabel(r'$x$', fontsize=16)
            ax.set_ylabel(r'$f(x)$', fontsize=16)
        # If the user wanted a title, put it in
        if set_title:
            ax.set_title(name + ' Test Function', fontweight='bold')
        # Find / create folders to hold pictures
        out_folder = os.path.join(os.getcwd(), 'plots')
        if not os.path.isdir(out_folder):
            os.makedirs(out_folder)
        # Save file and close - delete figure
        filename = os.path.join(out_folder, '%s.png'%name)
        fig.savefig(filename)
        plt.close(fig)
        del fig

    # Pre:  "graph" is a pylab subplot, "x" is a one dimensional list
    #       of x_values, "y" is a list of lists of y values where
    #       each inner list belongs to one x value, "color" is a string
    # Post: The mean of the the sets of y values is plotted across the
    #       given x values
    def plot_mean(graph=plt.subplot(), x=[], y=[], label="",
                  extrema=False, marker=None, markerfacecolor=None,
                  dashes=(None,None)):
        if (len(y) == 0) or (len(x) != len(y)):
            raise(Exception("Error: x values and y values"+
                            " lists must be same length."))
        mean = [sum(ys_per_x)/len(ys_per_x) for ys_per_x in y]
        line = graph.plot(x, mean, dashes=dashes,label=label)[0]
        if extrema: # If extrema should be plotted
            color = plt.getp(line, 'color')
            min_vals = [min(ys_per_x) for ys_per_x in y]
            max_vals = [max(ys_per_x) for ys_per_x in y]
            graph.plot(x, min_vals, color+"--",label=label+" Extrema",
                       marker=marker, markerfacecolor=markerfacecolor)
            graph.plot(x, max_vals, color+"--", marker=marker,
                       markerfacecolor=markerfacecolor)
        return graph

except:
    # Pre:  "benchmark_fxn" is a Benchmark class instance
    # Post: A plot image is saved to a "plots" folder in the CWD
    def plot_benchmark(bnchmrk_fxn, set_title=False):
        print("No plotting libraries available.")

    # Pre:  "graph" is a pylab subplot, "x" is a one dimensional list
    #       of x_values, "y" is a list of lists of y values where
    #       each inner list belongs to one x value, "color" is a string
    # Post: The mean of the the sets of y values is plotted across the
    #       given x values
    def plot_mean(graph=None, x=[], y=[], label="",
                  extrema=False, marker=None, markerfacecolor=None):
        print("No plotting libraries available.")    

if __name__ == "__main__":
    print("Testing step counter")
    timer = Timer(100)
    for i in range(100):
        timer.step()
        time.sleep(0.05)
    print("Done!")

    print("Testing percentage counter")
    timer.reset()
    for i in range(100):
        timer.step(i / 100)
        time.sleep(0.05)
    print("Done!")

