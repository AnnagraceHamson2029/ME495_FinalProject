import subprocess
import sys
import random
import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# path = "Documents/ME495/Lab3/evolve_please.py"


def configure_smudges(mygen, min_loss, max_loss_change, childlist):
    # Combine the best children IDs from the two lists
    best_children_id = list(set(min_loss + max_loss_change))
    print("Best children IDs:", best_children_id)
    if mygen ==1:
        with open("Documents/ME495/final_project/best_children.pkl", "wb") as file:
            pickle.dump([best_children_id], file)
    else:
        with open("Documents/ME495/final_project/best_children.pkl", "rb") as file:
            bestlist = pickle.load(file)
            bestlist.append(best_children_id)
        with open("Documents/ME495/final_project/best_children.pkl", "wb") as file:
            pickle.dump(bestlist, file)

    # make list of best features
    best_x = []
    best_rad = []
    best_dir = []
    new_childlist = []
    best_child_list = []
    for i in range(len(best_children_id)):
        thischild = childlist[best_children_id[i]]
        thischild[4] = mygen
        thischild[0] = i
        new_childlist.append(thischild)
        best_child_list.append(thischild)
        best_x.append(thischild[1])
        best_rad.append(thischild[2])
        best_dir.append(thischild[3])

    # make new features based on best features
    for i in range(len(childlist) - len(new_childlist)):
        xlist = []
        radlist = []
        directionlist = []
        rand_list = [random.randint(0, len(best_child_list) - 1), random.randint(0, len(best_child_list) - 1), random.randint(0, len(best_child_list) - 1)]
        for i in range(5):
            # randomly incorporate from best, or point mutation
            if random.randint(0,1) == 0:
                myx = best_x[rand_list[0]][random.randint(0,4)]
            else:
                myx = random.uniform(0, 0.26)
            child_id = random.randint(0, len(best_child_list) - 1)
            if random.randint(0,1) == 0:            
                myrad = best_rad[rand_list[1]][random.randint(0,4)]
            else:
                myrad = random.uniform(0,0.09)
            if random.randint(0,1) == 0:
                mydir = best_dir[rand_list[2]][random.randint(0,4)]
            else:
                if random.randint(0,1) == 0:
                    mydir = 'left'
                else:
                    mydir = "right"

                
            # add drift and check for out of bounds
            if random.randint(0,1) == 0:
                myx += random.uniform(0, 0.02)
            if myx >= 0.26:
                myx = 0.26
            if random.randint(0,1) == 0:
                myrad += random.uniform(0, 0.02)
            if 2*myrad >= 0.1:
                myrad = 0.05
            
            xlist.append(myx)
            radlist.append(myrad)
            directionlist.append(mydir)
        
        new_childlist.append([i+len(new_childlist)-1, xlist, radlist, directionlist, mygen])

    return new_childlist


def assess_child(N, losses, loss_change):
    # Ensure the total number of losses is a multiple of N (or handle the remainder separately)
    num_generations = len(losses) // N
    losses_array = np.array(losses[:num_generations * N])
    generations_matrix = losses_array.reshape(num_generations, N)

        # Calculate per-generation statistics
    means = generations_matrix.mean(axis=1)
    medians = np.median(generations_matrix, axis=1)
    stds = generations_matrix.std(axis=1)

    generations = np.arange(num_generations)
    plt.figure(figsize=(8, 5))
    plt.plot(generations, means, label='Mean Loss', marker='o')
    plt.plot(generations, medians, label='Median Loss', marker='x')
    plt.fill_between(generations, means - stds, means + stds, color='gray', alpha=0.3, label='Std Dev')
    plt.xlabel('Generation')
    plt.ylabel('Loss')
    plt.legend()
    plt.title("Loss Statistics Over Generations")
    plt.show()

    first_gen = generations_matrix[0]
    last_gen = generations_matrix[-1]

    # t-test (assuming independent samples and approximate normality)
    t_stat, p_val = stats.ttest_ind(first_gen, last_gen)
    print("T-test: t-statistic = {:.3f}, p-value = {:.3f}".format(t_stat, p_val))

    # Alternatively, for non-parametric comparison:
    u_stat, p_val_u = stats.mannwhitneyu(first_gen, last_gen)
    print("Mann-Whitney U test: U statistic = {:.3f}, p-value = {:.3f}".format(u_stat, p_val_u))
    


def generation_generator():
    all_losses = []
    all_loss_change = []
    N_gen = 10
    N_children = 10
    N_smudges = 5
    direction_options = ['left', 'right']

    # randomly initialize children
    all_child_list = []
    child_list=[]
    last_loss = []
    for child in range(N_children):
        xlist = []
        dirlist = []
        radlist = []
        for i in range(N_smudges):
            # pick x
            px = random.uniform(0.0, 0.26)
            rad = random.uniform(0.01, 0.1)
            dir_int = random.randint(0,1)
            dir = direction_options[dir_int]
            xlist.append(px)
            radlist.append(rad)
            dirlist.append(dir)
        child_list.append([child, xlist, radlist, dirlist, 0])
    all_child_list.append(child_list)
    # with open("Documents/ME495/final_project/init_child.pkl", "wb") as file:
    #     pickle.dump(child_list, file)
    
    for gen in range(N_gen):
        print(f"assessing generation {gen}")
        gen_losses = []
        gen_lc = []
        for i in range(len(child_list)):
            print(f"Training child {i}")
            # 1. write this child's info to the loading pickle file
            # 2. run for that child
            with open("Documents/ME495/final_project/evolve_input.pkl", "wb") as file:
                pickle.dump(child_list[i], file)
            print(child_list[i])
            run_script()
            # 3. load childs losses from pickle file
            with open("Documents/ME495/final_project/losses.pkl", "rb") as file:
                child_loss = pickle.load(file)
            last_loss.append(child_loss[-1])
            loss_change = abs(child_loss[-1]-child_loss[0])
            all_loss_change.append(loss_change)
            all_losses.append(child_loss)
            gen_losses.append(child_loss)
            gen_lc.append(loss_change)


        #### Find the two with best losses and two with most change
        min_two_losses = sorted(range(len(gen_losses)), key=lambda i: min(gen_losses[i]))[0:1]
        max_two_lc = sorted(gen_lc, reverse=True)[:2]  # Sort in descending order and take the first two
        top_indices = [idx for idx, _ in sorted(enumerate(gen_lc), key=lambda x: x[1], reverse=True)[0:1]]


        print(gen+1)
        child_list = configure_smudges(gen+1, min_two_losses, top_indices, child_list)
        print(child_list)
        all_child_list.append(child_list)
    assess_child(N_children, last_loss, all_loss_change)


def run_script():
    try:
        result = subprocess.run(
            [sys.executable, "Documents/ME495/final_project/diffmpm.py"],  # Replace with your actual script
            check=True
        )
        return 0  # Normal execution
    except subprocess.CalledProcessError:
        with open("Documents/ME495/final_project/losses.pkl", "rb") as file:
            my_loss = pickle.load(file)
        if len(my_loss) !=20:
            print("Segmentation fault, checking length of loss file")
            losses = [0]*100 # 100 is number of iters in diffmpm file
            with open("Documents/ME495/final_project/losses.pkl", "wb") as file:
                pickle.dump(losses, file)
          # If a segmentation fault or bus error occurs, exit with 0

if __name__ == "__main__":
    sys.exit(generation_generator())