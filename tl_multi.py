from multiprocessing import Process, Queue, cpu_count, set_start_method
import sys, time

# Make sure new processes started are minimal (not complete copies)
set_start_method("spawn", force=True)
MAX_PROCS = cpu_count() # From multiprocessing library
PROC_TIMEOUT = 420 # 7 Minutes, should be adjusted per use

# Pre:  "funcs" is a list of function calls,
#       "timeout" is a number > 0
#       "procs" is a number > 0
# Post: Each function in "funcs" is called with no parameters in its
#       own process (all of the results are compiled into a list and
#       returned
def simple_multi(funcs, timeout=PROC_TIMEOUT, procs=MAX_PROCS):
    processes = []
    queue = Queue();
    for f in funcs:
        processes.append( Process(target=f) )
        # Tell the process to start running
        processes[-1].start()

    # Retrieve all process results from the queue in one big list
    returns = []
    for p in processes:
        try:    returns.append( queue.get(timeout=timeout) )
        except: print("Waring: %s didn't finish."%str(p))
    for p in processes: p.join(timeout=timeout)
    for p in processes: p.terminate()
    # Return the list of all function call results
    return returns
    
# ================================================
#      Distributed Arguments multi-processing     
# ================================================

def _check_multi_args(multi_args, procs):
    if len(multi_args) == 0: 
        raise(ValueError("Need at least one tuple of arguments to" +
                         " pass to processors.")) 
    if len(multi_args) < procs:
        raise(ValueError(("%s is not enough arguments to divide among"
                          + " %s processors.")%(len(multi_args),procs)))

# Pre:  "func" is a function that should be called with all the
#       arguments in "whole_args" and the ith value from each list in
#       "multi_args" for i in range(len(multi_args[0]))
# Post: 
def multi_proc(func, whole_args, multi_args, queue):
    returns = []
    for instance_args in multi_args:
        returns.append(func( *(whole_args+instance_args) ))
    queue.put(returns)

# Pre:  "func" is a function that: takes each item in "whole_args" as
#         an argument (in order) for every execution; and takes one
#         tuple from "multi_args" per execution of "func". 
#       "whole_args" is a tuple
#       "multi_args" is a list of tuples
#       "timeout" > 0 and "procs" > 0.
# Post: "func" is called by "procs" processes with each object in
#       "whole_args" being passed for every execution, the lists in
#       "multi_args" are divided among "procs" and each proc will
#       execute "func" len(multi_args[0]) // procs times, once
#       for each set of items provided by the ith index in each list
#       of "multi_args". A timeout is used on each process to ensure
#       progress with warnings sent when the timeout is reached.
def conquer(func, whole_args, multi_args,
            timeout=PROC_TIMEOUT, procs=MAX_PROCS):
    # Check to make sure the argument lists are of the same length
    _check_multi_args(multi_args, procs)
    # Begin preparing constants / variables for multiprocessing
    batch_sizes = [len(multi_args) // procs for i in range(procs)]
    large_batches = len(multi_args) % procs
    for i in range(large_batches): batch_sizes[i] += 1
    processes = []; queue = Queue();
    for i in range(procs):
        start = sum(bs for bs in batch_sizes[:i])
        end = start + batch_sizes[i]
        # Convert multi_args sub-list to tuple for process creation
        multi_args_for_proc = tuple(multi_args[start:end])
        processes.append( Process(target=multi_proc, 
                                  args=(func, whole_args, 
                                        multi_args_for_proc,queue)) )
        # Tell the process to start running
        processes[-1].start()

    # Retrieve all process results from the queue in one big list
    returns = []
    for p in processes:
        try:    returns += queue.get(timeout=timeout)
        except: print("Waring: %s didn't finish."%str(p))
    for p in processes: p.join(timeout=timeout)
    for p in processes: p.terminate()
    # Return the list of all function call results
    return returns

# Pre:  "group" is a list, "procs" is integer > 0
# Post: "group" is divided into "procs" lists of approximately equal
#       length. This function is for dividing a list into "procs" sub
#       lists as evenly as possible presumably so that each individual
#       list can be sent to a process
def divide(group, procs=MAX_PROCS):
    sub_groups = []
    batch_sizes = [len(group) // procs for i in range(procs)]
    large_batches = len(group) % procs
    for i in range(large_batches): batch_sizes[i] += 1
    for i in range(procs):
        start = sum(bs for bs in batch_sizes[:i])
        end = start + batch_sizes[i]
        sub_groups.append( group[start:end] )
    return sub_groups

# ======================
#      Testing Code     
# ======================

# For testing purposes only
def _(intro, list_of_numbers):
    strings = []
    for i in list_of_numbers:
        strings.append( intro + str(i) + "." )
    return strings

if __name__ == "__main__":    
    send_every_time = ("Hello my number is ",)
    send_per_proc = divide([str(n) for n in range(17)])        
    send_per_proc = list(zip(send_per_proc))
    answers = conquer(_, send_every_time, send_per_proc)
    for a in answers:
        print(a)


