#B&C

import gurobipy as grb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#data
# 10 nurses, 1 week - i
# 20 nurses, 1 week - h 
# 20 nurses, 2 weeks - a 
# 40 nurses, 2 weeks - b 
# 80 nurses, 2 weeks - f 
# 20 nurses, 4 weeks - g 
# 60 nurses, 4 weeks - c
# 120 nurses, 4 weeks - d 
# 200 nurses, 4 weeks - j
# 200 nurses, 6 weeks - k
# 120 nurses, 8 weeks - e 

 
# data
data1 = pd.read_csv('C:/Users/Emma/Desktop/Fourth Year/Dissertation/data/newdata2\data1k.csv')
data2 = pd.read_csv('C:/Users/Emma/Desktop/Fourth Year/Dissertation/data/newdata2\data2k.csv')
data3 = pd.read_csv('C:/Users/Emma/Desktop/Fourth Year/Dissertation/data/newdata2\data3k.csv')

n = data1.to_numpy()[:, (0)]
I = len(n)  # number of nurses
n2=data2.to_numpy()[:, (0)]
J = len(n2)  # number of shifts

# model
model = grb.Model(name='NSP')
model.Params.timeLimit = 6*60*60

# parameters
F = data1.to_numpy()[:, (1)]  # max morning shifts
A = data1.to_numpy()[:, (2)]  # max afternoon shifts
N = data1.to_numpy()[:, (3)]  # max night shifts
U = data1.to_numpy()[:, (4)]  # max total shifts
L = data1.to_numpy()[:, (5)]  # min total shifts
smax = data1.to_numpy()[:, (6)]  # max consecutive days
omin = data1.to_numpy()[:, (7)]  # min consecutive days off
wmax = data1.to_numpy()[:, (8)]  # max weekends
M = data2.to_numpy()[:, (2)]  # quota requirement parameter
h=60 #nurse preference weight
h2=60 #consecutive days worked weight
h3=30 #weekend constraint weight
h4=30#consecutive days off weight
E = data1.to_numpy()[:, (9)]  # min night shifts

cn = 10*8  # normal nurse cost per shift
co = 5*8  # normal nurse overtime cost per shift


nurses = [i for i in range(1, I+1)] #set of nurses
Shifts = [j for j in range(1, J+1)] #set of shifts
MShifts = [j for j in range(1, J+1, 3)]  # morning shifts
AShifts = [j for j in range(2, J+1, 3)]  # afternoon shifts
NShifts = [j for j in range(3, J+1, 3)]  # night shifts
Weekends = [j for j in range(1, (J // 21)+1)]  # set of weekends
indices1 = [i for i, x in enumerate(omin) if x == 2]
indices2 = [i for i, x in enumerate(omin) if x == 3]
#daysforsmax=[i for i in range(1, (J+2-3*(smax[i-1]+1)), 3)] 


# # shift off requests parameter
pp = data3.to_numpy()[:, (0)]
p = np.zeros((I, J))
nurseid = data3.to_numpy()[:, (0)]
shiftindex = data3.to_numpy()[:, (1)]


# decision variables
x = {}
for i in nurses:
    for j in Shifts:
        x[i, j] = model.addVar(vtype=grb.GRB.BINARY,
                               name='x'+str(i)+','+str(j))

k = {} #weekend decision variable
for i in nurses:
    for w in Weekends:
        k[i, w] = model.addVar(vtype=grb.GRB.BINARY,
                                name='k'+str(i)+','+str(w))

#for consecutive days worked        
y = {}
for i in nurses:
    for s in range(1, (J+2-3*(smax[i-1]+1)), 3):
        y[i, s] = model.addVar(vtype=grb.GRB.INTEGER,
                               name='y'+str(i)+','+str(s))
#for consecutive days worked        
z = {}
for i in nurses:
    for s in range(1, (J+2-3*(smax[i-1]+1)), 3):
        z[i, s] = model.addVar(vtype=grb.GRB.INTEGER,
                               name='z'+str(i)+','+str(s))

#for weekends worked as soft constraint      
u = {}
for i in nurses:
    u[i] = model.addVar(vtype=grb.GRB.INTEGER,
                               name='u'+str(i))
    
#for weekends worked as soft constraint      
v = {}
for i in nurses:
    v[i] = model.addVar(vtype=grb.GRB.INTEGER,
                               name='v'+str(i))
    
#for consecutive days off as soft constraint      
t = {}
for i in nurses:
    for q in range(1,omin[i-1]):
        for s in range(1, J//3-q): 
            t[i, q, s] = model.addVar(vtype=grb.GRB.INTEGER,
                               name='t'+str(i)+str(q)+str(s))
g = {}
for i in nurses:
    for q in range(1,omin[i-1]):
        for s in range(1, J//3-q): 
            g[i, q, s] = model.addVar(vtype=grb.GRB.INTEGER,
                               name='g'+str(i)+str(q)+str(s))

#ensuring the following decision variables are non-negative
for i in nurses:
    u[i]>=0


for i in nurses:
    v[i]>=0
    
for i in nurses:
    for s in range(1, (J+2-3*(smax[i-1]+1)), 3):
        y[i, s] >=0

for i in nurses:
    for s in range(1, (J+2-3*(smax[i-1]+1)), 3):
        z[i, s] >=0

for i in nurses:
    for q in range(1,omin[i-1]):
        for s in range(1, J//3-q): 
            t[i, q, s]>=0

for i in nurses:
    for q in range(1,omin[i-1]):
        for s in range(1, J//3-q): 
            g[i, q, s]>=0



# CONSTRAINTS

# qouta requirement
for j in Shifts:
    model.addConstr(grb.quicksum(x[i, j] for i in nurses) >= M[j-1])

# number of shifts worked per nurse
for i in nurses:
    model.addConstr(grb.quicksum(x[i, j] for j in Shifts) >= L[i-1])


# number of morning/afternoon/night shifts worked
for i in nurses:
    model.addConstr(grb.quicksum(x[i, k] for k in AShifts) <= A[i-1])

for i in nurses:
    model.addConstr(grb.quicksum(x[i, k] for k in NShifts) <= N[i-1])

for i in nurses:
    model.addConstr(grb.quicksum(x[i, k] for k in MShifts) <= F[i-1])
    
for i in nurses:
    model.addConstr(grb.quicksum(x[i, k] for k in NShifts) >= E[i-1])


# max of 1 shift per day
for i in nurses:
    for s in MShifts:
        model.addConstr(x[i, s] + x[i, s+1] + x[i, s+2] <= 1)

#defining y and z for consecutive days worked
for i in nurses:
    for s in range(1, (J+2-3*(smax[i-1]+1)), 3):
        model.addConstr(grb.quicksum(x[i, k] for k in range(s, s+3*(smax[i-1]+1))) - y[i,s] + z[i,s] == smax[i-1])


#consecutive days off
q=1
for i in nurses:
    for s in range(1, J//3-q):
        model.addConstr(1-grb.quicksum(x[i, (3*s-l)] for l in range(0, 3)) + grb.quicksum(x[i, l] for l in range(3*s+1, 3*s+3*q+1)) + 1 - grb.quicksum(x[i, (3*(s+q+1)-l)] for l in range(0, 3)) + t[i,q,s] - g[i,q,s] == 1)

q=2
for i in indices2:
    for s in range(1, J//3-q):
        model.addConstr(1-grb.quicksum(x[i+1, (3*s-l)] for l in range(0, 3)) + grb.quicksum(x[i+1, l] for l in range(3*s+1, 3*s+3*q+1)) + 1 - grb.quicksum(x[i+1, (3*(s+q+1)-l)] for l in range(0, 3)) + t[i+1,q,s] - g[i+1,q,s] == 1)


# exclusion of shift pattern: night-morning
for i in nurses:
    for s in range(3, J-2, 3):
        model.addConstr(x[i, s] + x[i, s+1] <= 1)


#weekend constraints
for i in nurses:
    for w in Weekends:
        model.addConstr(grb.quicksum(x[i, 21*w-s]
                        for s in range(0, 6)) >= k[i, w])

for i in nurses:
    for w in Weekends:
        model.addConstr(grb.quicksum(x[i, 21*w-s]
                        for s in range(0, 6)) <= 2*k[i, w])

#defining u and v for weekends constraint as soft constraint
for i in nurses:
    model.addConstr(grb.quicksum(k[i, w] for w in Weekends) + v[i] - u[i] == wmax[i-1])




# objective function
oo = [i for i in range(1, len(pp)+1)]
for i in oo:
    p[nurseid[i-1]-1, shiftindex[i-1]-1] = 1

objective = grb.quicksum(cn*x[i, j] for i in nurses for j in Shifts) + co*grb.quicksum((grb.quicksum(x[i, j]
                                                                                                      for j in Shifts)) -L[i-1] for i in nurses) + grb.quicksum(h*p[i-1, j-1]*x[i, j] for i in nurses for j in Shifts) + grb.quicksum(h2*y[i, s] for i in nurses for s in range(1, (J+2-3*(smax[i-1]+1)), 3)) + grb.quicksum(h3*u[i] for i in nurses) + grb.quicksum(h4*t[i, q, s] for i in nurses for q in range(1,omin[i-1]) for s in range(1, J//3-q)) 




# solving model
model.ModelSense = grb.GRB.MINIMIZE
model.setObjective(objective)
model.setParam('MIPGap', 0)
msolve = model.optimize()
model.optimize()

# for v in model.getVars():
#     if (v.x > 0):
#         print(v.varName, v.x)

        
#gantt chart
fig, gnt = plt.subplots(figsize=(16,10))
 
# Setting Y-axis limits
gnt.set_ylim(0, I)
 
# Setting X-axis limits
gnt.set_xlim(0, J)

gnt.set_xlabel('Shift Index')
gnt.set_ylabel('Nurse Index')
gnt.set_yticks([i for i in range(1,I+1)])
gnt.set_yticklabels(range(1,I+1))
gnt.set_xticks([j for j in range(1,J+1)])
gnt.set_xticklabels(range(1,J+1))
gnt.grid(True)

for i in nurses:
    for j in Shifts:
        if x[i,j].x==1:
            gnt.broken_barh([(j-1, 1)], (i-1, 1), facecolors =('tab:red'))
 

plt.savefig("gantt1.png")



