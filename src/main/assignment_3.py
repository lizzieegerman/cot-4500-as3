from __init__ import *
import numpy as np
np.set_printoptions(precision=7, suppress=True, linewidth=100)

# Question 1
def eulers_method(function_euler, range_euler, iteration_euler, initial_euler):
    h_euler = (range_euler[1] - range_euler[0]) / iteration_euler
    t_previous_euler = range_euler[0]
    w_previous_euler = initial_euler

    for i in range(iteration_euler):
        t_curr_euler = t_previous_euler + h_euler
        w_curr_euler = w_previous_euler + h_euler * function_euler(t_previous_euler, w_previous_euler)
        t_previous_euler = t_curr_euler
        w_previous_euler = w_curr_euler

    return w_curr_euler


def function_euler(t, w):
    return t - w ** 2

# Question 2
def runge_kutta_method(function_r_k, range_r_k, iteration_r_k, initial_r_k):
    h_r_k = (range_r_k[1] - range_r_k[0]) / iteration_r_k
    t_previous_r_k = range_r_k[0]
    w_previous_r_k = initial_r_k

    for i in range(iteration_r_k):
        k_1 = h_r_k * function_r_k(t_previous_r_k, w_previous_r_k)
        k_2 = h_r_k * function_r_k(t_previous_r_k + h_r_k / 2, w_previous_r_k + k_1 / 2)
        k_3 = h_r_k * function_r_k(t_previous_r_k + h_r_k / 2, w_previous_r_k + k_2 / 2)
        k_4 = h_r_k * function_r_k(t_previous_r_k + h_r_k, w_previous_r_k + k_3)

        t_curr_r_k = t_previous_r_k + h_r_k
        w_curr_r_k = w_previous_r_k + (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6

        t_previous_r_k = t_curr_r_k
        w_previous_r_k = w_curr_r_k

    return w_curr_r_k


def function_r_k(t, w):
    return t - w ** 2

# Question 3:
gaussian_array = np.array([[2, -1, 1, 6], [1, 3, 1, 0], [-1, 5, 4, -3]])

for i in range(gaussian_array.shape[0]):
    max_row = i + np.argmax(np.abs(gaussian_array[i:, i]))
    gaussian_array[[i, max_row]] = gaussian_array[[max_row, i]]
    for j in range(i + 1, gaussian_array.shape[0]):
        factor = gaussian_array[j, i] / gaussian_array[i, i]
        gaussian_array[j, i:] = gaussian_array[j, i:] - factor * gaussian_array[i, i:]

x = np.zeros(gaussian_array.shape[0])
for i in range(gaussian_array.shape[0] - 1, -1, -1):
    x[i] = (gaussian_array[i, -1] - np.dot(gaussian_array[i, :-1], x)) / gaussian_array[i, i]
x = x.astype(dtype=np.double)


# Question 4:
LU_array = np.array([[1, 1, 0, 3], [2, 1, -1, 1], [3, -1, -1, 2], [-1, 2, 3, -1]])

def LU_factorization(array):
    n = array.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for k in range(n):
        L[k, k] = 1
        for j in range(k, n):
            U[k, j] = array[k, j] - np.dot(L[k, :k], U[:k, j])
        for i in range(k + 1, n):
            L[i, k] = (array[i, k] - np.dot(L[i, :k], U[:k, k])) / U[k, k]

    return L, U


L, U = LU_factorization(LU_array)
determinant = np.linalg.det(U)

# Question 5:
def diagonally_dominate(matrix):
    n = len(matrix)
    for i in range(n):
        row_sum = sum(abs(matrix[i][j]) for j in range(n) if j != i)
        if abs(matrix[i][i]) < row_sum:
            return False
    return True

# Question 6:
def positive_definite(matrix):
    if matrix.shape[0] != matrix.shape[1]:
        return False
    if not np.allclose(matrix, matrix.T):
        return False
    eigenvalues = np.linalg.eigvals(matrix)
    return np.all(eigenvalues > 0)


if __name__ == "__main__":

    # Question 1 output:
    range_euler = (0, 2)
    iteration_euler = 10
    initial_euler = 1

    print('%.5f' % eulers_method(function_euler, range_euler, iteration_euler, initial_euler))
    print()

    # Question 2 output:
    range_r_k = (0, 2)
    iteration_r_k = 10
    initial_r_k = 1

    print('%.5f' % runge_kutta_method(function_r_k, range_r_k, iteration_r_k, initial_r_k))
    print()

    # Question 3 output:
    print(x)
    print()

    # Question 4 output:
    print('%.5f' %determinant)
    print()
    print(L)
    print()
    print(U)
    print()

    # Question 5 output:
    matrix = [[9, 0, 5, 2, 1],[3, 9, 1, 2, 1],[0, 1, 7, 2, 3],[4, 2, 3, 12, 2],[3, 2, 4, 0, 8]]
    print(diagonally_dominate(matrix))
    print()

    # Question 6 output:
    matrix = np.array([[2, 2, 1],[2, 3, 0],[1, 0, 2],])

    print(positive_definite(matrix))
