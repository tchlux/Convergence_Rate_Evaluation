import os

machines = 25
user = "thlux"
path = "/home/students/"+user
status = path + "/Status/%s"
host = user+"@cs.roanoke.edu"
base_name = "lab-mcsp-"
connect = "ssh %s \"ssh %s -o ConnectTimeout=%i %s 2> /dev/null\""
timeout = 5
remote_run = lambda num, command: connect%(host, base_name + num, 
                                           timeout, command)
test_code = lambda fxn_name, status_name: ("python3 " +path+ 
            "/Opt/test_algorithms.py %s > " +status+ " 2> " 
            +status+ " &")%(fxn_name, status_name, status_name)

names = ["Adjiman", "Beale", "EggHolder", "GoldsteinPrice",
         "Langermann", "Shubert01", "SixHumpCamel", "UrsemWaves",
         "DropWave", "Whitley", "MieleCantrell", "Weierstrass",
         "Rastrigin", "Katsuura", "Salomon", "Deceptive", "Giunta",
         "Griewank", "Trigonometric02", "Paviani", "Sargan",
         "ZeroSum", "Plateau", "Michalewicz", "Mishra11", "OddSquare",
         "Qing", "Rosenbrock", "Alpine01", "Bohachevsky", "Easom",
         "Levy03", "MultiModal", "Penalty02", "Quintic", "Vincent",
         "Ackley", "CosineMixture", "Wavy", "NeedleEye",
         "Pathological", "Rana", "Schwefel22",
         "DeflectedCorrugatedSpring", "Mishra02", "Penalty01",
         "Exponential", "Ripple01", "Schwefel26", "SineEnvelope"]

names = ["Alpine01", "UrsemWaves"]

curr = 0
while curr < len(names):
    for i in range(10, machines):
        print()
        num = str(i//10) + str(i%10)
        print( "Connecting to %s..."%(base_name+num), end="\r" )
        online = os.popen(remote_run(num, "ls")).readline() != ""
        if online:
            # cancel = "pkill python3"
            # print( "Cancelling process on %s..."%(base_name+num), end="\r" )            
            # os.system(remote_run(num,cancel))
            status_file = base_name+num+"_"+names[curr]
            code = test_code(names[curr], status_file)
            print( "Commanding %s..."%(base_name+num), end="\r" )
            os.system(remote_run(num, code))
            print("Running %s on %s."%(names[curr], 
                                       base_name+num))
            curr += 1
        if curr == len(names): break
