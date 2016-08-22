# Creator: Thomas C.H. Lux
# Contact: thomas.ch.lux@gmail.com

# Optimization functions
from AMPGO import AMPGO
from Anneal import Anneal
from BSA import BSA
from Cuckoo import Cuckoo
from qABC import qABC

# Test functions for the optimization
import benchmark
# Utilities for tracking progress
from util import Tracker, MeanTracker, rank_probability, data_profile

# Libraries
import pickle, sys
import numpy as np

FOLDERNAME = "/home/thomas/Dropbox/Research/TL ML/Optimization/Optimization_Stats/"

if __name__ == "__main__":
    # to_run = sys.argv[1]
    
    # Parameters for Trials at runtime 
    min_cycles = 25
    num_solutions = 20
    hist_size = 9
    algorithms = [AMPGO, Anneal, BSA, Cuckoo, qABC]
    # List of function names to be used in testing
    names = [("Adjiman", 2),
             ("Beale", 2),
             ("EggHolder",2),
             ("GoldsteinPrice",2),
             ("Langermann",2),
             ("Shubert01",2), 
             ("SixHumpCamel",2),
             ("UrsemWaves",2),             
             ("DropWave",4),
             ("Whitley",4),
             ("MieleCantrell",4),
             ("Weierstrass",4),
             ("Rastrigin",4),
             ("Katsuura",4),
             ("Salomon",4),
             ("Deceptive",8),
             ("Giunta",8),
             ("Griewank",8),
             ("Trigonometric02",8),
             ("Paviani",8),
             ("Sargan",8),
             ("ZeroSum",8),
             ("Plateau",16),
             ("Michalewicz",16),
             ("Mishra11",16),
             ("OddSquare",16),
             ("Qing",16),
             ("Rosenbrock",16), 
             ("Alpine01",16),
             ("Bohachevsky", 24),
             ("Easom",24),
             ("Levy03",24),
             ("MultiModal",24),
             ("Penalty02",24),
             ("Quintic",24),
             ("Vincent",24),
             ("Ackley",48), 
             ("CosineMixture",48),
             ("Wavy",48),
             ("NeedleEye",48),
             ("Pathological",48),
             ("Rana",48),
             ("Schwefel22",48),
             ("DeflectedCorrugatedSpring",96),
             ("Mishra02",96),
             ("Penalty01",96),
             ("Exponential",96),
             ("Ripple01",96),
             ("Schwefel26",96),
             ("SineEnvelope",96)
         ]

    # Do the tests!
    for name, dimensions in names:
        # if name != to_run:
        #     continue
        
        # Initialize the test function class
        try:
            test_class = getattr(benchmark, name)(dimensions=dimensions)
        except TypeError:
            test_class = getattr(benchmark, name)()
            print("Having trouble setting dimensions for %s."%name)
            exit()
        # Parameter creation
        objective = test_class.evaluator
        bounds = test_class.bounds

        # Header for test function
        print("============Testing Function===========")
        spaces = " " * ((39 - len(name)) // 2)
        print(spaces,name,spaces,sep="")
        print("Dimensions:",len(bounds))

        # Try each algorithm
        raw_data = {}
        for algorithm in algorithms:
            # print("Loading algorithm...",end="\r")
            # Header for algorithm
            alg_name = str(algorithm).split()[1]
            spaces = "=" * ((33 - len(alg_name)) // 2)
            print(spaces," ",alg_name," ",spaces,sep="")

            # Initialize trackers for each iteration's objective
            # function value so that we can tell when the objective
            # function values being obtained at a particular iteration
            # have converged after repeated trials
            raw_data[alg_name] = [MeanTracker(100) for i 
                                  in range(min_cycles)]
            # Run repeated trials until final objective value converges
            while not raw_data[alg_name][min_cycles-1].converged:
                # Generate random initial solution(s)
                if alg_name in ["BSA", "Cuckoo", "qABC"]:
                    solution = np.array([np.array(test_class.generator())
                                         for i in range(num_solutions)])
                else:
                    solution = np.array(test_class.generator())
                # Optimize function and organize results
                t = Tracker(algorithm, display=False)
                obj, sol, record = t.track(objective, solution,
                                           bounds, min_cycles)
                # Make the record only the "best" values obtained
                for i in range(1,min_cycles):
                    record[i] = min(record[i-1],record[i])
                # Place raw data into convergence trackers
                for i in range(min_cycles):
                    raw_data[alg_name][i].append(record[i])
                print("Itaration %i, convergence: %0.1f%%"%
                      (raw_data[alg_name][0].values.shape[0], 
                       100.0 / min_cycles *
                       sum(i.converged for i in raw_data[alg_name])),
                      end="\r", flush=True)
                # End while not converged

            # Record all of the statistics specific to this algorithm
            alg_stats = {"mean":[], "min":[], "max":[], "stdev":[],
                         "hist":[], "conv":[]}
            for i in range(min_cycles):
                data = raw_data[alg_name][i]
                # Undo the normalization that occurred for convergence analysis
                data.values *= data.divisor
                data.values += data.shift
                values = data.values
                # Generate histogram data
                min_value = min(values)
                max_value = max(values)
                hist = [0] * hist_size
                width = (max_value - min_value) / hist_size
                for i in range(hist_size):
                    lower = min_value + i*width
                    upper = min_value + (i+1)*width
                    # Special end case
                    if i == 0:
                        hist[i] = values[values <= upper].shape[0]
                    else: # Normal case
                        temp = values[lower < values]
                        hist[i] = temp[temp <= upper].shape[0]
                # Record all stats
                alg_stats["mean"].append(values.mean())
                alg_stats["min"].append(min_value)
                alg_stats["max"].append(max_value)
                alg_stats["stdev"].append(values.std())
                alg_stats["hist"].append(hist)
                alg_stats["conv"].append(values.converged)
                # End for i in range(min_cycles)

            # Record the number of trials that this algorithm needed to converge
            alg_stats["trials"] = raw_data[alg_name][0].values.shape[0]

            print("Trials:",alg_stats["trials"])
            # Dump the records into an appropriately named file
            filename = name + "_" + str(dimensions) + "_" + alg_name + ".pkl"
            with open(FOLDERNAME+filename, "wb") as f:
                pickle.dump(alg_stats, f)
            # End for algorithm in algorithms

        # Record all stats specific to this objective function, about
        # all of the algorithms tested in comparison to each other
        obj_func_stats = {"rank0prob":{}, "dataprofile":{}}
        alg_names = [str(a).split()[1] for a in algorithms]
        best_objs = [min(min(raw_data[alg_name][i].values) 
                         for alg_name in alg_names)
                     for i in range(min_cycles)]
        # Generate rank 0 probability statistics
        for alg_name in alg_names:
            # Initialize holders for data profiling and ranking
            performances = []
            rank_0_probability = []
            others = [a for a in alg_names if a != alg_name]
            # Cycle through iterations of objective function executions
            print("Computing rank probability...",end="\r")
            for i in range(min_cycles):
                # Calculate the rank information for this algorithm
                interest = raw_data[alg_name][i].values
                other_data = [raw_data[a][i].values for a in others]

                rank_0_probability.append( 
                    rank_probability(interest, other_data, rank=0) )
                # Append performances (for later data profiling)
                performances.append(interest)
            print("Computing data profile...", end="\r")
            # Compute and store data profile, store rank 0 probability
            obj_func_stats["dataprofile"][alg_name] = \
                            data_profile(performances, best_objs)
            obj_func_stats["rank0prob"][alg_name] = rank_0_probability

        print("Saving data to file...",end="\r")
        # Dump the obj func records into an appropriately named file
        filename = name + "_" + str(dimensions) + "_Compare_5.pkl"
        with open(FOLDERNAME+filename, "wb") as f:
            pickle.dump(obj_func_stats, f)
        print("Save Completed!",end="\n")
        print("--------------------------------------------------")
        # End for name, dimension

    # End main if




