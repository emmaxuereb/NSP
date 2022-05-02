# genetic algorithm search of the one max optimization problem

#importing required packages
import numpy as np, numpy.random
import pandas as pd
import copy
import time
import random
import math
import operator # for dictionary 

#data
#(1) 10 nurses, 1 week 
#(2)  20 nurses, 1 week  
#(3)  20 nurses, 2 weeks 
#(4)  40 nurses, 2 weeks 
#(5)  80 nurses, 2 weeks 
#(6)  20 nurses, 4 weeks  
#(7)  60 nurses, 4 weeks 
#(8)  120 nurses, 4 weeks 
#(9)  200 nurses, 4 weeks 
#(10)  200 nurses, 6 weeks 
#(11)  120 nurses, 8 weeks 


#importing data

# data
data1 = pd.read_csv('C:/Users/Emma/Desktop/Fourth Year/Dissertation/data\instance1_1.csv')
data2 = pd.read_csv('C:/Users/Emma/Desktop/Fourth Year/Dissertation/data\instance1_2.csv')
data3 = pd.read_csv('C:/Users/Emma/Desktop/Fourth Year/Dissertation/data\instance1_3.csv')

#defining parameters
n = data1.to_numpy()[:, (0)]
I = len(n)  # number of nurses
n2=data2.to_numpy()[:, (0)]
J = len(n2)  # number of shifts

F = data1.to_numpy()[:, (1)]  # max morning shifts
A = data1.to_numpy()[:, (2)]  # max afternoon shifts
N = data1.to_numpy()[:, (3)]  # max night shifts
E = data1.to_numpy()[:, (9)]  # min night shifts
L = data1.to_numpy()[:, (5)]  # min total shifts
smax = data1.to_numpy()[:, (6)]  # max consecutive working days
omin = data1.to_numpy()[:, (7)]  # min consecutive days off
wmax = data1.to_numpy()[:, (8)]  # max weekends
M = data2.to_numpy()[:, (2)]  # quota requirement parameter

h=60 #nurse preference weight
h2=60 #consecutive days worked weight
h3=30 #weekend constraint weight
h4=30#consecutive days off weight

cn = 10*8  # normal nurse cost per shift
co = 5*8  # normal nurse overtime cost per shift

#defining sets
nurses = [i for i in range(1, I+1)] #set of nurses
Shifts = [j for j in range(1, J+1)] #set of shifts
MShifts = [j for j in range(1, J+1, 3)]  # morning shifts
AShifts = [j for j in range(2, J+1, 3)]  # afternoon shifts
NShifts = [j for j in range(3, J+1, 3)]  # night shifts
Weekends = [j for j in range(1, (J // 21)+1)]  # set of weekends
indices2 = [i for i, x in enumerate(omin) if x == 3] #used for omin
indices3 = [i for i, x in enumerate(smax) if x == 4] #used for smax
indices4 = [i for i, x in enumerate(smax) if x == 5] #used for smax


# shift off requests parameter
pp = data3.to_numpy()[:, (0)]
p = np.zeros((I, J))
nurseid = data3.to_numpy()[:, (0)]
shiftindex = data3.to_numpy()[:, (1)]
oo = [i for i in range(1, len(pp)+1)]
for i in oo:
    p[nurseid[i-1]-1, shiftindex[i-1]-1] = 1

q=J//3 #number of days in scheduling horizon
days=[j for j in range(1, J//3+1)] #list of days
days_without_last=[j for j in range(1, J//3)] #list of days from day 1 to day q-1

dict = {} #empty dictionary

tens = [j for j in range(10, 3000,10)] #used to print every 10th iteration

#######################################################################################################

#parameters of ga - given values from parameter tuning
mutation_rate = 0.1
n_gen = 1600 #number of generations
init_pop_size = 40
number_of_elements = math.floor(0.05*I*q) #number of elements to be considered in mutation = 10% of total no. of elements
sub_pop_size = math.floor(1/2*init_pop_size)
number_of_chroms = math.floor(mutation_rate*sub_pop_size) #number of chroms for mutation
remaining_pop_size = init_pop_size - sub_pop_size
crossover_rate=0.5

########################################################################################################

#function to give fitness function value
def fitnessvalue(arr):
    
    #total costs - sum is the total cost of all nurses
    sum=0
    for i in range(I):
        numofshifts=np.count_nonzero(arr[i,:])
        costs=cn*numofshifts + co*max(0,numofshifts-L[i-1]) 
        sum += costs
    
    #nurse preferences
    count=0 #count is the number of times there is a violation of nurse preferences
    for i in range(I):
        for j in range (0,J-1,3):
            if (p[i,j] == arr[i,j//3] or p[i,j+1]*2 == arr[i,j//3] or p[i,j+2]*3 == arr[i,j//3]) and (arr[i,j//3] != 0):
                count += 1
    
    #weekends worked
    count2 = 0
    for i in range(I):
        count3 = 0
        for j in Weekends:
            if ((arr[i,7*j-2] != 0) or (arr[i,7*j-1] != 0)):
                count3 += 1
        if count3 > wmax[i]:
            difference = wmax[i] - count3
            count2 += difference
            
    #consecutive days off - for omin=2 and omin=3
    count4 = 0
    for i in range(I):
        for j in range(0,q-2):
            if ((arr[i,j] != 0) and (arr[i,j+1] == 0) and (arr[i,j+2] != 0)):
                count4 += 1
    
    #consecutive days off - for omin=3
    count5 = 0
    for i in indices2:
        for j in range(0,q-3):
            if ((arr[i,j] != 0) and (arr[i,j+1] == 0) and (arr[i,j+2] == 0) and (arr[i,j+3] != 0)):
                count5 += 1
                
    #consecutive days worked (smax=4)
    count6 = 0
    for i in indices3:
        for j in range(0,q-4):
            if ((arr[i,j] != 0) and (arr[i,j+1] != 0) and (arr[i,j+2] != 0) and (arr[i,j+3] != 0) and (arr[i,j+4] != 0)):
                count6 += 1
                
    #consecutive days worked (smax=5)
    count7 = 0
    for i in indices4:
        for j in range(0,q-5):
            if ((arr[i,j] != 0) and (arr[i,j+1] != 0) and (arr[i,j+2] != 0) and (arr[i,j+3] != 0) and (arr[i,j+4] != 0) and (arr[i,j+5] != 0)):
                count7 += 1
   

    #fitness function value is represented by ans
    ans = sum + h*count + h3*count2 + h4*(count4+count5) + h2*(count6+count7)

    return ans

##########################################################################################################

#function to check feasibility of matrix
def feasibility(arr):
    currentpop=np.concatenate(arr)
    ans=True
    
    #hard constraints for each nurse
    for i in range(I):
        if not ((np.count_nonzero(arr[i,0:q]) >= L[i]) and (np.count_nonzero(arr[i,0:q] == 1) <= F[i]) and (np.count_nonzero(arr[i,0:q] == 2) <= A[i]) and (np.count_nonzero(arr[i,0:q] == 3) <= N[i]) and (np.count_nonzero(arr[i,0:q] == 3) >= E[i])):
            ans=False
            break
     
    #quota requirements constraint
    for j in days:
        m1=currentpop[j-1::q]
        if not (np.count_nonzero(m1 == 1) >= M[3*j-2-1] and  np.count_nonzero(m1 == 2) >= M[3*j-1-1] and np.count_nonzero(m1 == 3) >= M[3*j-1] ):
            ans=False
            break
    
    #shift pattern (night-morning) prohibited constraint
    for i in range(I):
        for j in range(q-1):
            if (arr[i,j] == 3 and arr[i,j+1] == 1):
                ans=False
                break
    
    return(ans)

############################################################################################################

#function to check feasibility of vector
def feasibility_vector(currentpop):
    arr=currentpop.reshape((I,-1))
    ans=True
    
    #hard constraints for each nurse
    for i in range(I):
        if not ((np.count_nonzero(arr[i,0:q]) >= L[i]) and (np.count_nonzero(arr[i,0:q] == 1) <= F[i]) and (np.count_nonzero(arr[i,0:q] == 2) <= A[i]) and (np.count_nonzero(arr[i,0:q] == 3) <= N[i]) and (np.count_nonzero(arr[i,0:q] == 3) >= E[i])):
            ans=False
            break
     
    #quota requirements constraint
    for j in days:
        m1=currentpop[j-1::q]
        if not (np.count_nonzero(m1 == 1) >= M[3*j-2-1] and  np.count_nonzero(m1 == 2) >= M[3*j-1-1] and np.count_nonzero(m1 == 3) >= M[3*j-1] ):
            ans=False
            break
    
    #shift pattern (night-morning) prohibited constraint
    for i in range(I):
        for j in range(q-1):
            if (arr[i,j] == 3 and arr[i,j+1] == 1):
                ans=False
                break
    
    return(ans)

############################################################################################################

#using best two selection method for crossover - defining function to select the best two chromosomes
#from the 2 parents and 2 children
def crossoverbest2(par1,par2,child1,child2):
    par1fitness=fitnessvalue(par1)
    par2fitness =fitnessvalue(par2)
    child1fitness= fitnessvalue(child1)
    child2fitness= fitnessvalue(child2)
    #create list of fitness values
    listoffitness = [par1fitness,par2fitness,child1fitness,child2fitness]
    bestchrom = min(listoffitness)
    listoffitness.remove(bestchrom)
    secondbestchrom = min(listoffitness)
    if bestchrom == par1fitness:
        if secondbestchrom == par2fitness:
                return par1, par2
        elif secondbestchrom == child1fitness:
            return par1, child1
        else:
            return par1, child2
    elif bestchrom == par2fitness:
        if secondbestchrom == par1fitness:
            return par1, par2
        elif secondbestchrom == child1fitness:
            return par2, child1
        else:
            return par2, child2
    elif bestchrom == child1fitness:
        if secondbestchrom == par1fitness:
            return par1, child1
        elif secondbestchrom == par2fitness:
            return par2, child1
        else:
            return child1, child2
    else:
        if secondbestchrom == par1fitness:
            return par1, child2
        elif secondbestchrom == par2fitness:
            return par2, child2
        else:
            return child1, child2

#########################################################################################################

#using best two selection method for crossover - when one of the children is infeasible
def crossoverbest2for3(par1,par2,child1):
    par1fitness=fitnessvalue(par1)
    par2fitness =fitnessvalue(par2)
    child1fitness= fitnessvalue(child1)
    #create list of fitness values
    listoffitness = [par1fitness,par2fitness,child1fitness]
    bestchrom = min(listoffitness)
    listoffitness.remove(bestchrom)
    secondbestchrom = min(listoffitness)
    if bestchrom == par1fitness:
        if secondbestchrom == par2fitness:
            return par1, par2
        else: 
            return par1, child1
    elif bestchrom == par2fitness:
        if secondbestchrom == par1fitness:
            return par1, par2
        else: 
            return par2, child1
    else:
        if secondbestchrom == par1fitness:
            return par1, child1
        else: 
            return par2, child1

###########################################################################################################

#function to make chromosome satisfy quota requirements and shift pattern
def make_chrom_feasible(chrom):

    endloop=0
    
    while True:
        number_1s = np.zeros(q)
        number_2s = np.zeros(q)
        number_3s = np.zeros(q)
        
        for k in days:
            number_1s[k-1] = np.count_nonzero(chrom[:,k-1] == 1) #number of ones on each day
            number_2s[k-1] = np.count_nonzero(chrom[:,k-1] == 2)
            number_3s[k-1] = np.count_nonzero(chrom[:,k-1] == 3) 
        
        diff_1s = number_1s - M.reshape((q,-1))[:,0] #M.reshape((q,-1)) gives us array of 7 (days) rows, each with 3 columns (shifts)
 
    
        if diff_1s.min() >= 0:
            break
        if max(diff_1s) == 0:
            break
        
        # which day lacks 1s    
        day_lacks_1s = np.argmin(diff_1s)
        day_exceeds_1s = np.argmax(diff_1s)
        
        for i in range(I):
            #if day_lacks_1s != 0 and  
            if (chrom[i, day_exceeds_1s] == 1 and chrom[i, day_lacks_1s] != 1 and not (day_lacks_1s != 0 and chrom[i, day_lacks_1s-1] == 3) and not (day_exceeds_1s != q-1 and chrom[i, day_lacks_1s] == 3 and chrom[i, day_exceeds_1s+1]==1) and not (chrom[i, day_lacks_1s] == 3 and day_lacks_1s - day_exceeds_1s == 1)):
                chrom[i, day_exceeds_1s], chrom[i, day_lacks_1s] = chrom[i, day_lacks_1s], chrom[i, day_exceeds_1s]
                break
            if i == I-1:
                endloop=1
                
                
        if endloop == 1:
            break
                
    # do the same for 2s
    while True:
        if endloop == 1: #if ones could not be arranged to satisfy constraints - skip this part
            break
        
        number_1s = np.zeros(q)
        number_2s = np.zeros(q)
        number_3s = np.zeros(q)
        
        for k in days:
            number_1s[k-1] = np.count_nonzero(chrom[:,k-1] == 1)
            number_2s[k-1] = np.count_nonzero(chrom[:,k-1] == 2)
            number_3s[k-1] = np.count_nonzero(chrom[:,k-1] == 3) 
        
        diff_1s = number_1s - M.reshape((q,-1))[:,0] 
        diff_2s = number_2s - M.reshape((q,-1))[:,1] 
    
        not_move_1s = list(np.where(diff_1s == 0)[0])

        if diff_2s.min() >= 0:
            break
        if max(diff_2s) == 0:
            break
         
        # which day lacks 2s    
        day_lacks_2s = np.argmin(diff_2s)
        day_exceeds_2s = np.argmax(diff_2s)
        

        for i in range(I):
            if day_lacks_2s in not_move_1s:
                if (chrom[i, day_exceeds_2s] == 2 and chrom[i, day_lacks_2s] != 2 and chrom[i, day_lacks_2s] != 1 and not (day_exceeds_2s != q-1 and chrom[i, day_lacks_2s] == 3 and chrom[i, day_exceeds_2s+1] == 1)):
                    chrom[i, day_exceeds_2s], chrom[i, day_lacks_2s] = chrom[i, day_lacks_2s], chrom[i, day_exceeds_2s]
                    break
            else:
                if (chrom[i, day_exceeds_2s] == 2 and chrom[i, day_lacks_2s] != 2 and not (day_exceeds_2s != 0 and chrom[i, day_lacks_2s] == 1 and chrom[i, day_exceeds_2s-1] == 3) and not (day_exceeds_2s != q-1 and chrom[i, day_lacks_2s] == 3 and chrom[i, day_exceeds_2s+1] == 1)):
                    chrom[i, day_exceeds_2s], chrom[i, day_lacks_2s] = chrom[i, day_lacks_2s], chrom[i, day_exceeds_2s]
                    break
            
            if i == I-1:
                endloop=2
                
        if endloop == 2:
            break


    # do the same for 3s
    while True:
        if endloop == 1: #if ones could not be arranged to satisfy constraints - skip this part
            break
        if endloop == 2: #if twos could not be arranged to satisfy constraints - skip this part
            break
        
        number_1s = np.zeros(q)
        number_2s = np.zeros(q)
        number_3s = np.zeros(q)
        
        for k in days:
            number_1s[k-1] = np.count_nonzero(chrom[:,k-1] == 1)
            number_2s[k-1] = np.count_nonzero(chrom[:,k-1] == 2)
            number_3s[k-1] = np.count_nonzero(chrom[:,k-1] == 3) 
        
        diff_1s = number_1s - M.reshape((q,-1))[:,0] 
        diff_2s = number_2s - M.reshape((q,-1))[:,1] 
        diff_3s = number_3s - M.reshape((q,-1))[:,2]
            
        not_move_1s = list(np.where(diff_1s == 0)[0])
        not_move_2s = list(np.where(diff_2s == 0)[0])

        if diff_3s.min() >= 0:
            break
        if max(diff_3s) == 0:
            break
         
        # which day lacks 3s    
        day_lacks_3s = np.argmin(diff_3s)
        day_exceeds_3s = np.argmax(diff_3s)
        

        for i in range(I):
            if (day_lacks_3s in not_move_1s) and (day_lacks_3s in not_move_2s):
                if (chrom[i, day_exceeds_3s] == 3 and chrom[i, day_lacks_3s] != 3 and chrom[i, day_lacks_3s] != 1 and chrom[i, day_lacks_3s] != 2 and not (day_lacks_3s != q-1 and chrom[i, day_lacks_3s +1] == 1)):
                    chrom[i, day_exceeds_3s], chrom[i, day_lacks_3s] = chrom[i, day_lacks_3s], chrom[i, day_exceeds_3s]
                    break
            elif (day_lacks_3s in not_move_2s) and (day_lacks_3s not in not_move_1s):
                if (chrom[i, day_exceeds_3s] == 3 and chrom[i, day_lacks_3s] != 3 and chrom[i, day_lacks_3s] != 2 and not (day_lacks_3s != q-1 and chrom[i, day_lacks_3s +1] == 1) and not (day_exceeds_3s != 0 and chrom[i, day_lacks_3s] == 1 and chrom[i, day_exceeds_3s-1]==3) and not (chrom[i, day_lacks_3s] == 1 and day_exceeds_3s - day_lacks_3s == 1)):
                    chrom[i, day_exceeds_3s], chrom[i, day_lacks_3s] = chrom[i, day_lacks_3s], chrom[i, day_exceeds_3s]
                    break
            elif (day_lacks_3s in not_move_1s) and (day_lacks_3s not in not_move_2s):
                if (chrom[i, day_exceeds_3s] == 3 and chrom[i, day_lacks_3s] != 3 and chrom[i, day_lacks_3s] != 1 and not (day_lacks_3s != q-1 and chrom[i, day_lacks_3s +1] == 1)):
                    chrom[i, day_exceeds_3s], chrom[i, day_lacks_3s] = chrom[i, day_lacks_3s], chrom[i, day_exceeds_3s]
                    break
            else:
                if (chrom[i, day_exceeds_3s] == 3 and chrom[i, day_lacks_3s] != 3 and not (day_lacks_3s != q-1 and chrom[i, day_lacks_3s +1] == 1) and not (day_exceeds_3s != 0 and chrom[i, day_lacks_3s] == 1 and chrom[i, day_exceeds_3s-1]==3) and not (chrom[i, day_lacks_3s] == 1 and day_exceeds_3s - day_lacks_3s == 1)):
                    chrom[i, day_exceeds_3s], chrom[i, day_lacks_3s] = chrom[i, day_lacks_3s], chrom[i, day_exceeds_3s]
                    break
            if i == I-1:
                endloop=3
            
        if endloop == 3:
            break
    
    return(chrom)


###########################################################################################################

#initial population

def init_pop():
    poplist=[] #empty population list
    while len(poplist)<init_pop_size: 
    
        chromosome = np.zeros((I,q))
        
        for i in nurses:
            # chromosome[i-1,] = np.random.randint(2,size=q)
            place1 = random.sample(range(0,q),F[i-1]) #gives list of length F[i-1] of range
            list2 = [r-1 for r in place1 if r > 0] #so no night shift is placed the day before a morning shift
            list1 = [r for r in range(0, q) if r not in place1+list2]
            place3 = random.sample(list1,N[i-1])
            list3 = [r for r in range(0, q) if r not in (place1+place3)]
            place2 = random.sample(list3,A[i-1])
            
            chromosome[i-1,place1] = 1
            chromosome[i-1,place2] = 2
            chromosome[i-1,place3] = 3
          
        ##############################################################################
        chrom = copy.deepcopy(chromosome)
        make_chrom_feasible(chrom)
        
        if feasibility(chrom) == True:
            poplist.append(chrom)
        
    for i in range(len(poplist)):
        dict[str(poplist[i])]= fitnessvalue(poplist[i])
    return poplist


############################################################################################################


#crossover
def crossover(arr1,arr2):

    arr3=copy.deepcopy(arr1)
    arr4=copy.deepcopy(arr2)
    nursestocrossover=np.random.choice(I, I//2, replace=False)

    for j in range(len(nursestocrossover)):
        s=nursestocrossover[j]
        arr1[s,:] = copy.deepcopy(arr4[s,:])
        arr2[s,:] = copy.deepcopy(arr3[s,:])

    #child 1
    child1 = copy.deepcopy(arr1)
    make_chrom_feasible(child1)
        
    #child 2
    child2 = copy.deepcopy(arr2)
    make_chrom_feasible(child2)

    
    #if both children were made feasible   
    if (feasibility(child1) == True and feasibility(child2) == True):
        newchrom1,newchrom2 = crossoverbest2(arr3,arr4,child1,child2)       
    
    #if only child 1 was made feasible  
    elif (feasibility(child1) == True and feasibility(child2) == False):
        newchrom1,newchrom2 = crossoverbest2for3(arr3,arr4,child1)

        
    #if only child 2 was made feasible  
    elif (feasibility(child1) == False and feasibility(child2) == True):
        newchrom1,newchrom2 = crossoverbest2for3(arr3,arr4,child2)
        
    #if both children are still infeasible - we just add the 2 parents to the new population  
    else:
        newchrom1=arr3
        newchrom2=arr4
    
    return newchrom1,newchrom2
        
##############################################################################################################################


#mutation

def mutation(current_chrom):   
    for e in range(number_of_elements):
        #print("current pop is feasible:",feasibility(current_pop))
        nonzero_indices = [idx for idx, val in enumerate(current_chrom) if val != 0]
        chosen_indices = np.random.choice(nonzero_indices, number_of_elements, replace=False)
        current_chrom_amended = copy.deepcopy(current_chrom) 
        chosen_index = chosen_indices[e]
        current_chrom_amended[chosen_index] = 0 #changing a nonzero value to a zero
        if ((feasibility_vector(current_chrom_amended)) == True):
            current_chrom = copy.deepcopy(current_chrom_amended)

    return current_chrom

       
# ##########################################################################################################

#creating subpopulation randomly
def sub_population_random(population):
    subpop = []
    
    indices_for_subpop = np.random.choice(init_pop_size, sub_pop_size, replace=False)
    remaining_chroms=[]
    for l in range(init_pop_size):
        if l not in indices_for_subpop:
            remaining_chroms.append(population[l])
        else:
            subpop.append(population[l])

    return subpop, remaining_chroms


# ##########################################################################################################

#creating subpopulation using truncation selection
def sub_population_truncation(population):
    subpop = []
    fitness_list = []
    remaining_chroms=[]
    
    for l in range(init_pop_size):        
        fitness_list.append(dict[str(population[l])])
    fitness_list.sort()
    threshold = fitness_list[sub_pop_size-1]
    for l in range(init_pop_size):
        if dict[str(population[l])] < threshold:
            subpop.append(population[l])
        
    for l in range(init_pop_size):
        if (dict[str(population[l])] > threshold):
            remaining_chroms.append(population[l])
        elif ((dict[str(population[l])] ==threshold) and len(subpop)<sub_pop_size):
            subpop.append(population[l])
        elif ((dict[str(population[l])] ==threshold) and len(subpop)>=sub_pop_size):
            remaining_chroms.append(population[l])

    return subpop, remaining_chroms


# ##########################################################################################################


#finding probabilities and cumulative probabilities for roulette wheel selection
def probability_roulette_wheel(population):
    fitness_list = []
    probability_list = []
    cumulative_prob_list = []
    
    for l in range(init_pop_size):
        fitness_list.append(dict[str(population[l])])
    #finding the inverse since we have a minimization problem
    inverse_fitness=1/np.array(fitness_list)


    sum_of_fitness = sum(inverse_fitness)
    
    cumulative_prob = 0
    for l in range(init_pop_size):
        probability_list.append(inverse_fitness[l]/sum_of_fitness)
        cumulative_prob += probability_list[l]
        cumulative_prob_list.append(cumulative_prob)
        
    for l in range(init_pop_size):
        if str(population[l]) not in dict:
            print("fal stage 3") 

    return cumulative_prob_list

# ##########################################################################################################

#selecting two parents using roulette wheel selection
def parents_roulette_wheel(population):
    cumulative_prob_list = probability_roulette_wheel(population)
    indices=[0,0]
    
    while indices[0] == indices[1]:
        rand2 = np.random.rand(2).tolist()
        indices.clear()
        for number in rand2:
            if number <= cumulative_prob_list[0]:
                index = 0
            else:
                for j in range(0,init_pop_size-1):
                    if number > cumulative_prob_list[j] and number <= cumulative_prob_list[j+1]:
                        index = j+1
                        break
            indices.append(index)

    return indices


#####################################################################################################

def next_population(oldpopulation):
    newpop = []
    sub_pop, remaining_pop = sub_population_truncation(oldpopulation)
    
    #crossover
    for l in range(sub_pop_size//2):
        #print("l is",l,"l+sub_pop_size//2 is",l+sub_pop_size//2)
        arr1=copy.deepcopy(sub_pop[l]) #chromosome 1
        arr2=copy.deepcopy(sub_pop[l+sub_pop_size//2]) #chromosome 2
        #print("arr1 feasibility is", feasibility(arr1))        
        new_chrom1, new_chrom2 = crossover(arr1,arr2)
        newpop.append(new_chrom1)
        newpop.append(new_chrom2)
        if str(new_chrom1) not in dict:
            dict[str(new_chrom1)]=fitnessvalue(new_chrom1)
        if str(new_chrom2) not in dict:
            dict[str(new_chrom2)]= fitnessvalue(new_chrom2) 
    
    #mutation
    
    chroms_for_mutation = np.random.choice(sub_pop_size, number_of_chroms, replace=False)
    for l in range(number_of_chroms):
        chrom_to_mutate = chroms_for_mutation[l]        
        current_chrom = np.concatenate(newpop[chrom_to_mutate])
        amended_chrom = mutation(current_chrom)
        amended_chrom_array = amended_chrom.reshape((I,-1))
        newpop[chrom_to_mutate] = amended_chrom_array
        if str(newpop[chrom_to_mutate]) not in dict:
            dict[str(newpop[chrom_to_mutate])]= fitnessvalue(newpop[chrom_to_mutate])  
    
    for l in range(remaining_pop_size):
        newpop.append(remaining_pop[l]) 
            
    return newpop

#####################################################################################################

def next_population_roulette(oldpopulation):
    newpop = []    

    for l in range(init_pop_size//2):   
        parent1_ind, parent2_ind = parents_roulette_wheel(oldpopulation)
        parent1 = copy.deepcopy(oldpopulation[parent1_ind])
        parent2 = copy.deepcopy(oldpopulation[parent2_ind])  
    #crossover
        if crossover_rate >= np.random.rand():     
            new_chrom1, new_chrom2 = crossover(parent1,parent2)

            newpop.append(new_chrom1)
            newpop.append(new_chrom2)
            if str(new_chrom1) not in dict:
                dict[str(new_chrom1)]=fitnessvalue(new_chrom1)
            if str(new_chrom2) not in dict:
                dict[str(new_chrom2)]= fitnessvalue(new_chrom2) 
            
        else:
            newpop.append(parent1)
            newpop.append(parent2)
            
    
    
    #mutation 
    chroms_for_mutation = np.random.choice(init_pop_size, number_of_chroms, replace=False)
    for l in range(number_of_chroms):
        chrom_to_mutate = chroms_for_mutation[l]        
        current_chrom = np.concatenate(newpop[chrom_to_mutate])
        amended_chrom = mutation(current_chrom)
        amended_chrom_array = amended_chrom.reshape((I,-1))
        newpop[chrom_to_mutate] = amended_chrom_array
        if str(newpop[chrom_to_mutate]) not in dict:
            dict[str(newpop[chrom_to_mutate])]= fitnessvalue(newpop[chrom_to_mutate])  
    
            
    return newpop
    
#####################################################################################################
    

def generationgen(tstart):
    oldpopulation = init_pop()
    for i in range(n_gen):
        # Print iteration number.
        if (i+1) in tens:
            print("iteration",i+1)
        sortdict=sorted(dict.items(), key = operator.itemgetter(1), reverse = False)
        bestfit = sortdict[0][1]
        if (i+1) in tens:
            print("time taken:",time.time()-tstart, bestfit)
        newpop= next_population(oldpopulation)
        oldpopulation = newpop.copy()

##########################################################################################################

# Running the GA         
def GA():
    dict.clear()
    tstart= time.time()
    generationgen(tstart)
    tend= time.time()
    print("time taken", tend-tstart)
    sortdict=sorted(dict.items(), key = operator.itemgetter(1), reverse = False)
    bestfitall = sortdict[0][1]
    best_sol = sortdict[0][0]
    for item in sortdict:
        if item[1]==bestfitall:
            print(item)
        else:
            break
    return best_sol

##########################################################################################################

#next_population(init_pop())
gg=GA()
