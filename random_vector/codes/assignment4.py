import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mping
import sys
sys.path.insert(0, '/home/aryan/GVV/random_vector/codes/CoordGeo')
import math

from line.funcs import*
from triangle.funcs import*
from conics.funcs import circ_gen

#generating random vertices 
#simlen = 2
#y = np.random.randint(-6,6, size = (3,simlen))
#A = y[0]
#B = y[1]
#C = y[2]
#print("A =", A)
#print("B =", B)
#print("C =", C)
#print('')

A = np.array([-6,-3])
B = np.array([-1,0])
C = np.array([3,-5])

#Q1.1.1
ab = B - A
bc = C - B
ca = A - C

print("The direction vector of AB is ",ab)
print("The direction vector of BC is ",bc)
print("The direction vector of CA is ",ca)
print('')


#Q1.1.2
V1 = ab.reshape(-1,1)
print("The length of AB is:",math.sqrt(ab@V1))
V2 = bc.reshape(-1,1)
print("The length of BC is:",math.sqrt(bc@V2))
V3 = ca.reshape(-1,1)
print("The length of AC is:",math.sqrt(ca@V3))
print('')

#Q1.1.3
Mat = np.array([[1,1,1],[A[0],B[0],C[0]],[A[1],B[1],C[1]]])

rank = np.linalg.matrix_rank(Mat)
print("rank:",rank)
if (rank<=2):
	print("Hence proved that points A,B,C in a triangle are collinear")
else:
	print("The given points are not collinear")
print('')

#Q1.1.4
print("parametric form of equation of line AB is x:",A,"+ k",ab)
print("parametric form of equation of line BC is x:",B,"+ k",bc)
print("parametric form of equation of line CA is x:",C,"+ k",ca)
print('')

#Q1.1.5
omat = np.array([[0,1],[-1,0]])
n4 = omat@ab    #normal vector
n5 = omat@bc    #normal vector
n6 = omat@ca    #normal vector
c4 = n4@A
c5 = n5@B
c6 = n6@C
eqn1 = f"{n4}x = {c4}"
eqn2 = f"{n5}x = {c5}"
eqn3 = f"{n6}x = {c6}"
print('')
print("The normal form of equation of line AB is",eqn1)
print("The normal form of equation of line BC is",eqn2)
print("The normal form of equation of line AC is",eqn3)
print('')

#Q1.1.6
def AreaCalc(A, B, C):
    ab = B - A
    ca = A - C
    cross_product = np.cross(ab,ca)
    magnitude = np.linalg.norm(cross_product)
    area = 0.5 * magnitude
    return area

area_ABC = AreaCalc(A, B, C)
print("Area of triangle ABC:", area_ABC)
print('')

#Q1.1.7
dotA=((B-A).T)@(C-A)
dotA=dotA
NormA=(np.linalg.norm(B-A))*(np.linalg.norm(C-A))
print('value of angle A: ', np.degrees(np.arccos((dotA)/NormA)))


dotB=(A-B).T@(C-B)
dotB=dotB
NormB=(np.linalg.norm(A-B))*(np.linalg.norm(C-B))
print('value of angle B: ', np.degrees(np.arccos((dotB)/NormB)))

dotC=(A-C).T@(B-C)
dotC=dotC
NormC=(np.linalg.norm(A-C))*(np.linalg.norm(B-C))
print('value of angle C: ', np.degrees(np.arccos((dotC)/NormC)))
print('')

#Q1.2.1
D = (B + C)/2
E = (A + C)/2
F = (A + B)/2

print("The midpoint of line BC is:",D)
print("The midpoint of line AC is:",E)
print("The midpoint of line AB is:",F)
print('')

#Q1.2.2
ad = D - A
be = E - B
cf = F - C
n7 = omat@ad
n8 = omat@be
n9 = omat@cf
c7 = n7@A
c8 = n8@B
c9 = n9@C
eqn4 = f"{n7}x = {c7}"
eqn5 = f"{n8}x = {c8}"
eqn6 = f"{n9}x = {c9}"
print('')
print("The normal form of equation of line AD is",eqn4)
print("The normal form of equation of line BE is",eqn5)
print("The normal form of equation of line CF is",eqn6)
print('')

#Q1.2.3
def line_intersect(n1,A1,n2,A2):
	N=np.block([[n1],[n2]])
	p = np.zeros(2)
	p[0] = n1@A1
	p[1] = n2@A2
	#Intersection
	P=np.linalg.inv(N)@p
	return P

def norm_vec(A,B):
	return np.matmul(omat, dir_vec(A,B))

def dir_vec(A,B):
    return B - A

G=line_intersect(norm_vec(F,C),C,norm_vec(E,B),B)
print("The point of intersection of BE and CF is:",list(G))
print('')

#Q1.2.4
AG = np.linalg.norm(G - A)
GD = np.linalg.norm(D - G)

BG = np.linalg.norm(G - B)
GE = np.linalg.norm(E - G)
 
CG = np.linalg.norm(G - C)
GF = np.linalg.norm(F - G)
print("AG/GD= "+str(AG/GD))
print("BG/GE= "+str(BG/GE))
print("CG/GF= "+str(CG/GF))
print('')

#Q1.2.5
Mat = np.array([[1,1,1],[A[0],D[0],G[0]],[A[1],D[1],G[1]]])

rank = np.linalg.matrix_rank(Mat)

if (rank==2):
	print("Since rank is equal to 2, the Points A,G,D are collinear")
else:
	print("They are not collinear")
print('')

#Q1.2.6
G = (A + B + C) / 3
print("The value of (A+B+C)/3:",G,'which is same as the value of G calculated above')   
print("Hence the centroid of the triangle is A+B+C/3")   
print('')

#1.2.7
LHS=(A-F)
RHS=(E-D)
if LHS.all()==RHS.all() :
   print("A-F=E-D and AFDE is a parallelogram")
else:
    print("Not equal")
print('')


#Q1.3.1
#sides
ab=B-A
bc=C-B
ca=A-C

t=np.array([0,1,-1,0]).reshape(2,2)

#AD_1
AD_1=t@bc

#normal vector of AD_1
AD_p=t@AD_1
print("The normal vectors of AD_1:",AD_p)

#Q1.3.2/Q1.3.3
n1 = bc    #normal vector
n2 = ca    #normal vector
n3 = ab    #normal vector
c1 = n1@A
c2 = n2@B
c3 = n3@C
eqn7 = f"{n1}x = {c1}"
eqn8 = f"{n2}x = {c2}"
eqn9 = f"{n3}x = {c3}"
print('')
print("The equation of line AD_1 is",eqn7)
print('')
print("The equation of line BE_1 is",eqn8)
print("The equation of line CF_1 is",eqn9)


def alt_foot(A,B,C):
  m = B-C
  n = np.matmul(omat,m) 
  N=np.vstack((m,n))
  p = np.zeros(2)
  p[0] = m@A 
  p[1] = n@B
  #Intersection
  P=np.linalg.inv(N.T)@p
  return P
D_1 = alt_foot(A,B,C)
E_1 = alt_foot(B,A,C)
F_1 = alt_foot(C,A,B)



#Q1.3.4
A1 = np.array([[n2[0],n2[1]],[n3[0],n3[1]]])             #Defining the vector A1
B1 = np.array([c2,c3])                     #Defining the vector B1
H  = np.linalg.solve(A1,B1)                 #applying linalg.solve to find x such that (A1)x=(B1)
print('')
print('The intersection of BE_1 and CF_1 (H):',H)
print('')

#Q1.3.5
result = int(((A - H).T) @ (B - C))    # Checking orthogonality condition...

# printing output
if result == 0:
    print("(A - H)^T (B - C) = 0\nHence Verified...")

else:
    print("(A - H)^T (B - C)) != 0\nHence the given statement is wrong...")
print('')


#Q1.4.1
def midpoint(P, Q):
    return (P + Q) / 2
def perpendicular_bisector(B, C):
    midBC=midpoint(B,C)
    dir=B-C
    constant = -dir.T @ midBC
    return dir,constant
equation_coeff1,const1 = perpendicular_bisector(A, B)
equation_coeff2,const2 = perpendicular_bisector(B, C)
equation_coeff3,const3 = perpendicular_bisector(C, A)
print(f'Equation for perpendicular bisector of AB:({equation_coeff1[0]:.2f})x + ({equation_coeff1[1]:.2f})y + ({const1:.2f}) = 0')
print(f'Equation for perpendicular bisector of BC:({equation_coeff2[0]:.2f})x + ({equation_coeff2[1]:.2f})y + ({const2:.2f}) = 0')
print(f'Equation for perpendicular bisector of CA:({equation_coeff3[0]:.2f})x + ({equation_coeff3[1]:.2f})y + ({const3:.2f}) = 0')
print('')

#Q1.4.2
O = line_intersect(ab,F,ca,E)
print('The point of intersection of perpendicular bisector of AB and AC is:',O)
print('')

#Q1.4.3
result = int((O - D) @ (B - C))
if result == 0:
    print("((O - D)(B - C))= 0\nHence Verified...")

else:
    print("(((O - D)(B - C))!= 0\nHence the given statement is wrong...")
print('')

#Q1.4.4
O_1 = O - A
O_2 = O - B
O_3 = O - C
a = np.linalg.norm(O_1)
b = np.linalg.norm(O_2)
c = np.linalg.norm(O_3)
print("OA, OB, OC are respectively", a,",", b,",",c, ".")
print("Here, OA = OB = OC.")
print("Hence verified.")
print('')

#Q1.4.5
X = A - O
radius = np.linalg.norm(X)
print("the radius of the circumcircle is:",radius)
print('')

#Q1.4.6
dot_pt_O = (B - O) @ ((C - O).T)
norm_pt_O = np.linalg.norm(B - O) * np.linalg.norm(C - O)
cos_theta_O = dot_pt_O / norm_pt_O
angle_BOC = round(360-np.degrees(np.arccos(cos_theta_O)),5)  #Round is used to round of number till 5 decimal places
print("angle BOC = " + str(angle_BOC))
dot_pt_A = (B - A) @ ((C - A).T)
norm_pt_A = np.linalg.norm(B - A) * np.linalg.norm(C - A)
cos_theta_A = dot_pt_A / norm_pt_A
angle_BAC = round(np.degrees(np.arccos(cos_theta_A)),5)  #Round is used to round of number till 5 decimal places
print("angle BAC = " + str(angle_BAC))
#To check whether the answer is correct
if angle_BOC == 2 * angle_BAC:
  print("\nangle BOC = 2 times angle BAC\nHence the give statement is correct")
else:
  print("\nangle BOC ≠ 2 times angle BAC\nHence the given statement is wrong")
print('')

#Q1.5.1
def unit_vec(A,B):
	return ((B-A)/np.linalg.norm(B-A))
E1= unit_vec(A,B) + unit_vec(A,C)
F1=np.array([E1[1],(E1[0]*(-1))])
C1= F1@(A.T)
E2= unit_vec(B,A) + unit_vec(B,C)
F2=np.array([E2[1],(E2[0]*(-1))])
C2= F2@(B.T)
E3= unit_vec(C,A) + unit_vec(C,B)
F3=np.array([E3[1],(E3[0]*(-1))])
C3= F3@(A.T)
print("Internal Angular bisector of angle A is:",F1,"x = ",C1)
print("Internal Angular bisector of angle B is:",F2,"x = ",C2)
print("Internal Angular bisector of angle C is:",F3,"x = ",C3)
print('')

#Q1.5.2
t = norm_vec(B,C) 
s1 = t/np.linalg.norm(t) 
t = norm_vec(C,A)
s2 = t/np.linalg.norm(t)
t = norm_vec(A,B)
s3 = t/np.linalg.norm(t)
I=line_intersect(s1-s3,B,s1-s2,C) 
print('The point of intersection of angle bisectors of B and C:',I)
print('')

#Q1.5.3
def angle_btw_vectors(v1, v2):
    dot_product = v1 @ v2
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    angle = np.arccos(dot_product / norm)
    angle_in_deg = np.degrees(angle)
    return angle_in_deg

angle_BAI = angle_btw_vectors(A-B, A-I)
angle_CAI = angle_btw_vectors(A-C, A-I)
print("Angle BAI:", angle_BAI)
print("Angle CAI:", angle_CAI)

if np.isclose(angle_BAI, angle_CAI):
    print("Angle BAI is equal to angle CAI.")
else:
    print("error")
print('')

#q1.5.4 and Q1.5.5
t = norm_vec(B, C)
n1 = t / np.linalg.norm(t)
r = n1 @ (B-I)
print(f"Distance from I to BC= {r}")
t = norm_vec(B, A)
n1 = t / np.linalg.norm(t)
r = n1 @ (I-B)
print(f"Distance from I to AB= {r}")
t = norm_vec(A, C)
n1 = t / np.linalg.norm(t)
r = n1 @ (I-C)
print(f"Distance from I to AC= {r}")
print('')

#Q1.5.8 and Q1.5.9
p=pow(np.linalg.norm(C-B),2)
q=2*((C-B)@(I-B))
r=pow(np.linalg.norm(I-B),2)-r*r
Discre=q*q-4*p*r
print("the Value of discriminant is ",abs(round(Discre,6)))

k=((I-C)@(B-C))/((B-C)@(B-C))
print("the value of parameter k is ",k)
D3=C+(k*(B-C))
print("the point of tangency of incircle by side BC is ",D3)
print("Hence we prove that side BC is tangent To incircle and also found the value of k!")

#finding k for E_3 and F_3
k1=((I-A)@(A-B))/((A-B)@(A-B))
k2=((I-A)@(A-C))/((A-C)@(A-C))
#finding E_3 and F_3
E3=A+(k1*(A-B))
F3=A+(k2*(A-C))
print('')
print("E3 = ",E3)
print("F3 = ",F3)
print('')

#Q1.5.10
def norm(X,Y):
    magnitude=round(float(np.linalg.norm([X-Y])),3)
    return magnitude 
print('')
print("AE_3=", norm(A,E3) ,"\nAF_3=", norm(A,F3) ,"\nBD_3=", norm(B,D3) ,"\nBE_3=", norm(B,E3) ,"\nCD_3=", norm(C,D3) ,"\nCF_3=",norm(C,F3))
print('')

#Q1.5.11
a = np.linalg.norm(B-C)
b = np.linalg.norm(C-A)
c = np.linalg.norm(A-B)

#creating array containing coefficients
Y = np.array([[1,1,0],[0,1,1],[1,0,1]])

#solving the equations
X = np.linalg.solve(Y,[c,a,b])
print("m = ",X[0])
print("n = ",X[1])
print("p = ",X[2])

#plotting the graphs
#for Q1.1
plt.figure(1)
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
tri_coords = np.block([[A],[B],[C]])
plt.scatter(tri_coords[:,0], tri_coords[:,1])
vert_labels = ['A','B','C']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[i,0], tri_coords[i,1]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.savefig('/home/aryan/GVV/random_vector/figs/fig1.png')

#for Q1.2
plt.figure(2)
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)
x_AD = line_gen(A,D)
x_BE = line_gen(B,E)
x_CF = line_gen(C,F)
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_AD[0,:],x_AD[1,:],label='$AD$')
plt.plot(x_BE[0,:],x_BE[1,:],label='$BE$')
plt.plot(x_CF[0,:],x_CF[1,:],label='$CF$')
tri_coords = np.block([[A],[B],[C],[G],[D],[E],[F]])
plt.scatter(tri_coords[:,0], tri_coords[:,1])
vert_labels = ['A','B','C','G','D','E','F',]
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[i,0], tri_coords[i,1]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.savefig('/home/aryan/GVV/random_vector/figs/fig2.png')

#for Q1.3
plt.figure(3)
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)
x_AD_1 = line_gen(A,D_1)
x_AE_1 = line_gen(A,E_1)
x_BE_1 = line_gen(B,E_1)
x_CF_1 = line_gen(C,F_1)
x_AF_1 = line_gen(A,F_1)
x_CH = line_gen(C,H)
x_BH = line_gen(B,H)
x_AH = line_gen(A,H)
x_BD_1 = line_gen(B,D_1)
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_AD_1[0,:],x_AD_1[1,:],label='$AD_1$')
plt.plot(x_BE_1[0,:],x_BE_1[1,:],label='$BE_1$')
plt.plot(x_AE_1[0,:],x_AE_1[1,:],linestyle = 'dashed',label='$AE_1$')
plt.plot(x_CF_1[0,:],x_CF_1[1,:],label='$CF_1$')
plt.plot(x_AF_1[0,:],x_AF_1[1,:],linestyle = 'dashed',label='$AF_1$')
plt.plot(x_CH[0,:],x_CH[1,:],label='$CH$')
plt.plot(x_BH[0,:],x_BH[1,:],label='$BH$')
plt.plot(x_AH[0,:],x_AH[1,:],linestyle = 'dashed',label='$AH$')
plt.plot(x_BD_1[0,:],x_BD_1[1,:],linestyle = 'dashed',label='$BD_1$')
tri_coords = np.block([[A],[B],[C],[D_1],[E_1],[F_1],[H]])
plt.scatter(tri_coords[:,0], tri_coords[:,1])
vert_labels = ['A','B','C','D1','E1','F1','H']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[i,0], tri_coords[i,1]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.savefig('/home/aryan/GVV/random_vector/figs/fig3.png')

#for Q1.4
plt.figure(4)
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)
x_OD = line_gen(O,D)
x_OE = line_gen(O,E)
x_OF = line_gen(O,F)
[O,r] = ccircle(A,B,C)
x_ccirc= circ_gen(O,radius)
x_OA = line_gen(O,A)
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_OA[0,:],x_OA[1,:],label='$OA$')
plt.plot(x_OD[0,:],x_OD[1,:],label='$OD$')
plt.plot(x_OE[0,:],x_OE[1,:],label='$OE$')
plt.plot(x_OF[0,:],x_OF[1,:],label='$OF$')
plt.plot(x_ccirc[0,:],x_ccirc[1,:],label='$circumcircle$')
tri_coords = np.block([[A],[B],[C],[O],[D],[E],[F]])
plt.scatter(tri_coords[:,0], tri_coords[:,1])
vert_labels = ['A','B','C','O','D','E','F']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[i,0], tri_coords[i,1]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.savefig('/home/aryan/GVV/random_vector/figs/fig4.png')

#for Q1.5
plt.figure(5)
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)
x_BI = line_gen(B,I)
x_CI = line_gen(C,I)
x_AI = line_gen(A,I)
[I,r] = icircle(A,B,C)
x_icirc= circ_gen(I,r)
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_CI[0,:],x_CI[1,:],label='$CI$')
plt.plot(x_BI[0,:],x_BI[1,:],label='$BI$')
plt.plot(x_AI[0,:],x_AI[1,:],label='$AI$')
plt.plot(x_icirc[0,:],x_icirc[1,:],label='$incircle$')
tri_coords = np.block([[A],[B],[C],[I],[D3],[E3],[F3]])
plt.scatter(tri_coords[:,0], tri_coords[:,1])
vert_labels = ['A','B','C','I','D3','E3','F3']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[i,0], tri_coords[i,1]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.savefig('/home/aryan/GVV/random_vector/figs/fig5.png')


