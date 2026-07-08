import numpy as np
import argparse

# convert string to boolean
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# set up argument parser
parser = argparse.ArgumentParser(description="Generate pathways based on specified parameters.")
parser.add_argument("--L", type=int, default=3, help="Memory length (Default: 3)")
parser.add_argument("--max_blips", type=int, default=3, help="Maximum number of blips (Default: 3)")
parser.add_argument("--num_inds", type=int, default=3, help="Number of GFN indices per complete spin pathway (Default: 3)")
parser.add_argument("--instance", type=int, default=2, help="Instance number for multiple GFN runs (Default: 2)")
parser.add_argument("--extra_paths", type=str2bool, default=False, help="Whether to add full antisymmetric pathways at end of file (Default: False)")
parser.add_argument("--empty_file", type=str2bool, default=False, help="Whether file is empty with only 0 for NumIter (Default: False)")

# parse arguments
args = parser.parse_args()

# assign variables
nsys = 2
L = args.L
max_blips = args.max_blips
num_inds = args.num_inds
instance = args.instance
extra_paths = args.extra_paths
empty_file = args.empty_file

if L == max_blips:
    filename = f"specific-paths-len{L-1}-inds{num_inds}-inst{instance}-full.inp"
else:
    filename = f"specific-paths-len{L-1}-inds{num_inds}-inst{instance}-gen.inp"

if empty_file == True:
    with open(filename, "w") as file:
        file.write("0 \n")

else:
    levels = np.arange(nsys, dtype="int")+1
    spin_grid = np.zeros((2*L, len(levels)), dtype="int")

    for index in range(len(spin_grid)):
        spin_grid[index] = levels

    #print(spin_grid)
    # generate all possible spin combinations
    spin_combos = np.array(np.meshgrid(*spin_grid)).T.reshape(-1, 2*L)  # unpack entries from grid using *
    #print(spin_combos)
    #print(len(spin_combos))
    
    # Define a function to count matches between first and last L elements
    def match_count(row):
        return sum(row[i] == row[i + L] for i in range(L))
    
    # Sort spin_combos by the match count
    # sorts spin_combos by their value of match_count
    # reverse=True sorts in descending order (higher value of match_count first)
    # this sorts pathways by increasing number of blips
    sorted_combos = np.array(sorted(spin_combos, key=match_count, reverse=True))
    #print(sorted_combos)
    
    # count number of blips for each spin pathway
    num_blips = [match_count(combo) for combo in spin_combos]
    num_blips = np.array(num_blips)

    # create extra paths if necessary
    if extra_paths == True:
        #extra_halfpath1 = np.full(L, 1) # fully antisymmetric
        #extra_halfpath2 = np.full(L, 2)
        #extra_path1 = np.hstack((extra_halfpath1, extra_halfpath2))
        #extra_path2 = np.hstack((extra_halfpath2, extra_halfpath1))
        #extra_combos1 = np.vstack((extra_path1, extra_path2))

        #extra_combos2 = np.full((2*L, 2*L), 1, dtype=int)
        #count = 0
        #for ind in range(len(extra_combos2)):
        #    extra_combos2[ind, count] = 2
        #    count += 1
        #extra_combos = np.vstack((extra_combos1, extra_combos2)) # single blips
        
        extra_combos = np.zeros((9, 2*L), dtype=int)
        extra_combos[0] = np.array([1,1,2,1,1,1])
        extra_combos[1] = np.array([1,2,2,1,2,1])
        extra_combos[2] = np.array([2,1,1,2,1,1])
        extra_combos[3] = np.array([1,1,1,2,1,2])
        extra_combos[4] = np.array([2,1,1,2,2,2])
        extra_combos[5] = np.array([2,2,1,2,1,1])
        extra_combos[6] = np.array([1,2,1,2,2,1])
        extra_combos[7] = np.array([1,1,1,2,2,2])
        extra_combos[8] = np.array([2,2,2,1,1,1])

    with open(filename, "w") as file:
        if extra_paths == True:
            file.write(f"{sum(num_blips<=max_blips)+len(extra_combos)} \n")
        else:
            file.write(f"{sum(num_blips<=max_blips)} \n")
    
        for combo in sorted_combos:
            # only write pathways with appropriate number of blips
            if (L-match_count(combo)<=max_blips):
                forward_path = " ".join(map(str, combo[:L]))
                file.write(forward_path + "    ")
                backward_path = " ".join(map(str, combo[L:2*L])) + "\n"
                file.write(backward_path)
    
        if extra_paths == True:
            for combo in extra_combos:
                forward_path = " ".join(map(str, combo[:L]))
                file.write(forward_path + "    ")
                backward_path = " ".join(map(str, combo[L:2*L])) + "\n"
                file.write(backward_path)
