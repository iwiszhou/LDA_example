from numpy import linalg as LA
import numpy as np

# The projection of a onto b 
def point_projection(a, b):
    return np.array((np.dot(a, b) / np.dot(b, b)) * b)

def array_point_projection(group_data, project_vector):
    return np.asarray([point_projection(group_data[i], project_vector) for i in range(len(group_data))])

# Change your own data points here
group1 = np.array([[4,2],[2,4],[2,3],[3,6],[4,4]])
group2 = np.array([[9,10],[6,8],[9,5],[8,7],[10,8]])

plt.scatter(group1.T[0],group1.T[1], color="red")
plt.scatter(group2.T[0],group2.T[1], color="blue")
plt.show()

S1 = np.cov(group1.T)
S2 = np.cov(group2.T)

print("Group 1 Cov Matrix (S1)  \n {} \n".format(group1_cov))
print("Group 2 Cov Matrix (S2)  \n {} \n".format(group2_cov))

Sw = np.matrix(S1 + S2)
print("Sw \n {} \n".format(Sw))

mean_group1 = group1.T.mean(axis=1)
mean_group2 = group2.T.mean(axis=1)

print("Group 1 Mean \n {} \n".format(mean_group1))
print("Group 2 Mean \n {} \n".format(mean_group2))

mean_diff=np.matrix(mean_group1-mean_group2)
Sb =  mean_diff.T @ mean_diff
print("Sb \n {} \n".format(Sb))

Sw_inverse_Sb = np.linalg.inv(Sw)*Sb
print("Sw^(-1)Sb  \n {} \n".format(Sw_inverse_Sb))

eigenvalues, eigenvectors = LA.eig(Sw_inverse_Sb)
eigenvectors = eigenvectors.T
print("eigenvalues  {} \n".format(eigenvalues))
print("eigenvectors  \n {} \n".format(eigenvectors))

max_index = eigenvalues.argmax()
projection_vector = np.squeeze(np.asarray(eigenvectors[max_index]))
print("largest eigenvector  \n {} \n".format(projection_vector))

# project original data to project vector
group1_new_data_p = array_point_projection(group1, projection_vector)
group2_new_data_p = array_point_projection(group2, projection_vector)


# scale point to line
projection_vector_set = np.vstack([projection_vector, projection_vector*-10, projection_vector*20, [0,0]])
projection_vector_array = np.asarray(projection_vector_set)

plt.clf()
plt.figure(figsize=(10,6))
plt.scatter(group1.T[0],group1.T[1], color="red")
plt.scatter(group2.T[0],group2.T[1], color="blue")
plt.plot(projection_vector_array.T[0], projection_vector_array.T[1], linestyle="-", color="green")
# new projection points
plt.scatter(group1_new_data_p.T[0],group1_new_data_p.T[1], color="green")
plt.scatter(group2_new_data_p.T[0],group2_new_data_p.T[1], color="green")

plt.show()
