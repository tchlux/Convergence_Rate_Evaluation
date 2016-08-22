import pickle, os
import pylab as pl
import numpy as np
import math

PATH_TO_PLOTS = "./Plots/"
PATH_TO_DATA = "./Optimization_Stats/"
ALGORITHMS = ["AMPGO", "Anneal", "BSA", "Cuckoo", "qABC"]
FUNCTIONS = ["Adjiman", "Beale", "EggHolder", "GoldsteinPrice",
             "Langermann", "Shubert01", "SixHumpCamel", "UrsemWaves",
             "DropWave", "Whitley", "MieleCantrell", "Weierstrass",
             "Rastrigin", "Katsuura", "Salomon", "Deceptive",
             "Giunta", "Griewank", "Trigonometric02", "Paviani",
             "Sargan", "ZeroSum", "Plateau", "Michalewicz",
             "Mishra11", "OddSquare", "Qing", "Rosenbrock",
             "Alpine01", "Bohachevsky", "Easom", "Levy03",
             "MultiModal", "Penalty02", "Quintic", "Vincent",
             "Ackley", "CosineMixture", "Wavy", "NeedleEye",
             "Pathological", "Rana", "Schwefel22",
             "DeflectedCorrugatedSpring", "Mishra02", "Penalty01",
             "Exponential", "Ripple01", "Schwefel26", "SineEnvelope"]

X = list(range(5000))
# Set the line styles for each algorithm with pyplot (on,off) syntax
LINE_STYLES = {"qABC":(None,None),
               "AMPGO":(6,2),
               "Anneal":(1,1),
               "BSA":(3,6),
               "Cuckoo":(5,1,2,1,5,1)}

NAME_FIXER = {"mean":"Mean",
              "min":"Absolute Min",
              "max":"Absolute Max",
              "stdev":"Standard Deviation"}

#      Load all algorithm specific files     
# ===========================================
algorithm_files = {}
for alg_name in ALGORITHMS:
    algorithm_files[alg_name] = [line.strip().split("/")[-1] for line
                                 in os.popen("ls %s*%s*"%
                                             (PATH_TO_DATA, alg_name))
                                 .readlines()]


#      Create Plots of Min,Max,Stdev,Avg,Rank,Data Per Function     
# ==================================================================
for fun_name in FUNCTIONS:
    print("[%0.1f%%] Plotting results..."%(100.0 * FUNCTIONS.index(fun_name)
                                           /len(FUNCTIONS)),end="\r")
    stats = {alg_name:{} for alg_name in ALGORITHMS}
    for alg_name in ALGORITHMS:
        file_name = [fn for fn in algorithm_files[alg_name]
                     if fun_name in fn][0]
        with open(PATH_TO_DATA+file_name, "rb") as f:
            raw_data = pickle.load(f)
        stats[alg_name].update(raw_data)
    for stat in stats[ALGORITHMS[0]]:
        if stat not in NAME_FIXER: continue
        plot_name = fun_name + " " + NAME_FIXER[stat]
        for alg_name in ALGORITHMS:
            pl.plot(X, stats[alg_name][stat],
                    dashes=LINE_STYLES[alg_name], label=alg_name,
                    linewidth=1)
        pl.title(plot_name)
        pl.xlabel("Objective function exectuions")
        pl.ylabel(NAME_FIXER[stat])
        pl.legend(loc="best")
        pl.savefig(PATH_TO_PLOTS+plot_name.replace(" ","_")+".svg")
        pl.clf()


#      Calculate average number of trials     
# ============================================
lens = []
for alg_name in ALGORITHMS:
    for file_name in algorithm_files[alg_name]:
        print("[%0.1f%%] Loading %s..."%(100.0 * ALGORITHMS.index(alg_name)
                                         /len(ALGORITHMS), file_name
                                     ),end="\r")
        with open(PATH_TO_DATA+file_name, "rb") as f:
            raw_data = pickle.load(f)
            lens.append(sum(raw_data["hist"][0]))

print("Average length: %i"%(sum(lens) / len(lens)))
print("Max length:     %i"%(max(lens)))

#      Calculate 1% best algorithms     
# ======================================
in_the_best = {alg_name:[] for alg_name in ALGORITHMS}
for fun_name in FUNCTIONS:
    print("[%0.1f%%] Recording best.."%(FUNCTIONS.index(fun_name) / 
                                        len(FUNCTIONS)*100.0), end="\r")
    abs_best = []
    abs_min = float("inf")
    abs_max = -float("inf")
    # First calculate the "best" and the absolute min and max
    for alg_name in ALGORITHMS:
        # Get the file specific to this function for this algorithm
        file_name = [fn for fn in algorithm_files[alg_name]
                     if fun_name in fn][0]
        with open(PATH_TO_DATA+file_name, "rb") as f:
            raw_data = pickle.load(f)
            abs_min = min(abs_min, min(raw_data["min"]))
            abs_max = max(abs_max, max(raw_data["max"]))
            if len(abs_best) == 0:
                abs_best = raw_data["min"]
            else:
                abs_best = [min(a,b) for a,b in 
                            zip(abs_best, raw_data["min"])]

    # Second calculate the actual percentage of the time the algorithm was best
    one_percent = 0.01 * (abs_max - abs_min)
    for alg_name in ALGORITHMS:
        # Initialize the list of best counters for this algorithm
        if len(in_the_best[alg_name]) == 0:
            in_the_best[alg_name] = [0] * len(abs_best)
        # Get the file specific to this function for this algorithm
        file_name = [fn for fn in algorithm_files[alg_name]
                     if fun_name in fn][0]
        # Calculate the iterations for which this algorithm was in the best
        with open(PATH_TO_DATA+file_name, "rb") as f:
            raw_data = pickle.load(f)
            for i in range(len(abs_best)):                
                if abs(raw_data["min"][i] - abs_best[i]) < one_percent:
                    in_the_best[alg_name][i] += 1

#      Load all comparison metric files     
# ==========================================
loaded = 0
rank_average = {}
profile_tols = []
profile_average = {}
function_files = [line.strip().split("/")[-1] for line in 
                  os.popen("ls %s*Compare*"%PATH_TO_DATA)]

# Tau for selecting data profile plot to create
T = [10**(-i) for i in (1,3,5,7)][0]

for file_name in function_files:
    print("[%0.1f%%] Loading %s..."%(100.0 * function_files.index(file_name)
                                     /len(function_files), file_name
                                 ),end="\r")
    with open(PATH_TO_DATA+file_name, "rb") as f:
        raw_data = pickle.load(f)
        loaded += 1
        for alg_name in ALGORITHMS:
            # Uncomment rank 0 probabilities and comment 
            # "Plot Data Profiles"  in order to switch between two,
            # also flip the comments for the "stat" variable

            # #      Plot Rank 0 Probabilities     
            # # ===================================
            # pl.plot(X, raw_data["rank0prob"][alg_name],
            #         dashes=LINE_STYLES[alg_name], label=alg_name,
            #         linewidth=1)

            # stat = "Rank 0 Probability"
            stat = "Data Profile T-%0.7f"%T

            #      Plot Data Profiles     
            # ============================
            pl.plot(X, raw_data["dataprofile"][alg_name][T],
                    dashes=LINE_STYLES[alg_name], label=alg_name,
                    linewidth=1)

            #      Rank 0 Probability     
            # ============================
            if rank_average.get(alg_name,None) == None:
                rank_average[alg_name] = np.array(raw_data["rank0prob"][alg_name])
            else:
                # Dynamically average the rank 0 probabilities
                rank_average[alg_name] += (
                    np.array(raw_data["rank0prob"][alg_name]) - 
                    rank_average[alg_name]) / loaded


            #      Data Profiling     
            # ========================
            if len(profile_tols) == 0:
                profile_tols = list(raw_data["dataprofile"][alg_name].keys())
                profile_tols.sort()

            for tol in profile_tols:
                if tol not in profile_average:
                    profile_average[tol] = {}
                if alg_name not in profile_average[tol]:
                    profile_average[tol][alg_name] = np.array(
                        raw_data["dataprofile"][alg_name][tol])
                else:
                    profile_average[tol][alg_name] += (
                        np.array(raw_data["dataprofile"][alg_name][tol]) -
                        profile_average[tol][alg_name]) / loaded


        plot_name = file_name.split("_")[0] + " " + stat
        pl.title(plot_name)
        pl.xlabel("Objective function exectuions")
        pl.ylabel(NAME_FIXER.get(stat,stat))
        pl.legend(loc="best")
        pl.savefig(PATH_TO_PLOTS+plot_name.replace(" ","_")+".svg")
        pl.clf()


#      Plot results     
# ======================
pl.rcParams.update({'font.size': 11, 'font.family':'serif'})

#      Plot the percentage best results     
# ==========================================
for alg_name in ALGORITHMS:
    in_the_best[alg_name] = np.array(in_the_best[alg_name])
    in_the_best[alg_name] = in_the_best[alg_name] / float(len(FUNCTIONS))
    pl.plot(X, in_the_best[alg_name] * 100.0,
            dashes=LINE_STYLES[alg_name], label=alg_name,
            linewidth=1)
pl.xlabel("Executions of objective function")
pl.ylabel("Probability of being able to achieve the best 1%")
pl.legend(loc="best")
pl.savefig(PATH_TO_PLOTS+"Average_Prob_Best.png")
# pl.show()

#      Plot and save Rank 0 Probability results     
# ==================================================
RP_BOX_LOC = (0.995,0.478)
DP_BOX_LOC = (0.995,0.46)
# Pick an arbitrary function and count how many iterations it goes out to

for alg_name in ALGORITHMS:
    pl.plot(X, rank_average[alg_name] * 100,
            dashes=LINE_STYLES[alg_name], label=alg_name,
            linewidth=1)

# pl.title("Average Rank 0 Probability")
pl.xlabel("Executions of objective function")
pl.ylabel("Probability of being rank 0")
pl.ylim( (0.0, 60.0) )
pl.legend(bbox_to_anchor=RP_BOX_LOC)
pl.savefig(PATH_TO_PLOTS+"Average_Rank_0_Probability.png")
# pl.show()

#      Plot and save Data Profile results     
# ============================================
pl.rcParams.update({'font.size': 14, 'font.family':'serif'})
for tol in profile_tols:
    pl.clf()
    for alg_name in ALGORITHMS:
        pl.plot(X, profile_average[tol][alg_name] * 100,
                dashes=LINE_STYLES[alg_name], label=alg_name,
                linewidth=2)
    tol = round(math.log10(tol))
    # pl.title("Average Data Profile T=10e%i"%tol)
    pl.xlabel("Executions of objective function")
    pl.ylabel("Percent successfully converged")
    pl.legend(bbox_to_anchor=DP_BOX_LOC)
    pl.savefig(PATH_TO_PLOTS+"Average_Data_Profile_T%i.png"%tol)
    # pl.show()

