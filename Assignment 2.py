import numpy as np
import pandas as pd

LI = lambda i, j, M: (M) * (i - 1) + (j - 1)   #Converts point (i,j) to corresponding Linear Index.

def generate_coefficient_matrix(delta_X) :
    M=int(1/delta_X )  #Index of last-most point i.e 1,2....Mth point in the GRID
    A = np.zeros(((M+1)*(M+1),(M+1)*(M+1)))   #Taken size (M+1)^2 initially but reduced to M^2 later, to avoid indexing error when X=1 and Y=1 points are accessed.

    for i in range(1,M+1) :
        for j in range(1,M+1) :
            
            if (i == 1) & (j == 1) :
                A[LI(i,j,M)][LI(2,1,M)] = -2
                A[LI(i,j,M)][LI(1,1,M)] = 4
                A[LI(1,1,M)][LI(1,2,M)] = -2
            elif (i==1) :
                A[LI(i,j,M)][LI(2,j,M)] = -2
                A[LI(i,j,M)][LI(1,j-1,M)] = -1
                A[LI(i,j,M)][LI(i,j,M)] = 4
                A[LI(i,j,M)][LI(1,j+1,M)] = -1
            elif (j==1) :
                A[LI(i,j,M)][LI(i+1,1,M)] = -1
                A[LI(i,j,M)][LI(i,2,M)] = -2
                A[LI(i,j,M)][LI(i,j,M)] = 4
                A[LI(i,j,M)][LI(i-1,1,M)] = -1
            else :
                A[LI(i,j,M)][LI(i-1,j,M)] = -1
                A[LI(i,j,M)][LI(i,j-1,M)] = -1
                A[LI(i,j,M)][LI(i,j,M)] = 4
                A[LI(i,j,M)][LI(i,j+1,M)] = -1
                A[LI(i,j,M)][LI(i+1,j,M)] = -1
    A = A[:(M)*(M),:(M)*(M)]
    return A   



#Set Grid Size for our problem
delta_X = 0.05      #Did not take 0.01 for the time being, as it took too long to solve. But delta_X can be changed according to need here.
M=int(1/delta_X )

#Set error tolerance for numerical solution
epsilon = 0.0001

A = generate_coefficient_matrix(delta_X)
B = np.ones((M*M,1))*(delta_X*delta_X)
T_k = np.zeros((M*M,1)) 



#Setting up matrices for gauss siedel
D = np.diag(np.diag(A))    
L = np.tril(A,k=-1)
U = np.triu(A,k=1)
C= np.linalg.inv(D+L) 

#Performing gauss siedel
c=0
while True :
    T_kplus1 = (C)@(-1*(U@T_k) + B)
    error = T_kplus1-T_k
    if (np.linalg.norm(error,ord=1) <= epsilon) :
        break
    else :
        T_k = T_kplus1

Theta_final = T_kplus1.reshape(M,M)
print(f"The final solution is as follows : \nNOTE : The orientation below is of 4th quadrant; i.e point (1,1) is at top left.\n")
print(pd.DataFrame(Theta_final))


