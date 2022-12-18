import numpy as np
import math

def back_substitution(matrix_echelon_form, previous_solutions=[]):
    n = len(matrix_echelon_form)
    m = len(matrix_echelon_form[0])

    if len(previous_solutions) == n:
        return previous_solutions

    if not previous_solutions:
        previous_solutions = []
    
    num_solutions_found = len(previous_solutions)
    index_current_equation = n - num_solutions_found - 1
    
    sum_solutions_x_coeficients = 0
    coeficient_index = 1

    for solution in previous_solutions:
        sum_solutions_x_coeficients += solution * matrix_echelon_form[index_current_equation][m-coeficient_index-1]
        coeficient_index+=1
    current_solution = (matrix_echelon_form[index_current_equation][m-1] - sum_solutions_x_coeficients)/matrix_echelon_form[index_current_equation][m-num_solutions_found-2]
    previous_solutions.append(current_solution)
    
    return back_substitution(matrix_echelon_form, previous_solutions)

def get_matrix_elimination(matrix, row=0, col=0):
    matrix_elimination = np.identity(len(matrix))
    piv = matrix[col][col]
    multiplicator = -(matrix[row][col]/piv)
    matrix_elimination[row][col] = multiplicator
    return matrix_elimination 

def gauss_elimination(A, b):
    n_A = len(A)
    m_A = len(A[0])
    augmented_matrix = np.hstack((A,b))

    for i in range(0, m_A):
        if augmented_matrix[i][i] == 0:
            index_row_piv_not_zero = get_next_index_row_piv_not_zero(augmented_matrix, i)
            if not index_row_piv_not_zero:
                continue
            augmented_matrix = get_matrix_row_permutation(n_A, i, index_row_piv_not_zero) @ augmented_matrix
            
        for j in range(i+1,n_A):
            matrix_elimination = get_matrix_elimination(augmented_matrix, row=j, col=i)
            augmented_matrix = matrix_elimination @ augmented_matrix
            augmented_matrix = np.around(augmented_matrix,decimals=6)
            
    return augmented_matrix

# Encontra as colunas com pivôs de uma matriz
def piv_colums(A):
    pivs = []
    # Utiliza a RREF para encontrar as colunas pivo
    RREF = rref(A)
    n = len(RREF)
    m = len(RREF[0])
    piv_index = 0
    
    for i in range(0, n):
        if piv_index >= m:
            break
            
        piv = 0
        # Percorre as colunas procurando um valor diferente de zero
        while piv == 0 and piv_index < m:
            piv = RREF[i][piv_index]
            if piv != 0:
                pivs.append(piv_index)
            piv_index+=1
            
    return pivs

# Calcula a RREF de uma matriz
def rref(A):
    # Número de linhas 
    n = len(A)
    # Número de colunas
    m = len(A[0])
    rref = A.copy()

    for i in range(0, m):
        if i >= n:
            break
            
        piv = float(rref[i][i])
        if piv == 0:
            # Permutação de Linha
            index_row_piv_not_zero = get_next_index_row_piv_not_zero(rref, i)            
            if not index_row_piv_not_zero:
                # Não encontrado outro possível pivo na coluna i
                continue            
            # Permutando linhas
            rref[i], rref[index_row_piv_not_zero] = rref[index_row_piv_not_zero].copy(), rref[i].copy()       
            piv = rref[i][i]            
                
        # Se o pivo não for 1, divide pelo seu valor para que fique 1
        if piv != 1:
            rref[i] = np.true_divide(rref[i].copy(), piv)          
        
        # Elimina (zera) os outros valores naquela coluna
        for j in range(0, n):
            if j != i:
                rref = eliminate_element_by_piv(rref, row=j, col=i)
                rref = np.around(rref,decimals=3)
    
    zeros_row = np.zeros(m)
    # Permuta as linhas com valores zero para as últimas linhas
    for j in range(0, n):
        if (rref[j] == zeros_row).all():            
            rref[j], rref[n-1] = rref[n-1].copy(), rref[j].copy()       
            
    return rref

def eliminate_element_by_piv(matrix, row=0, col=0):
    piv = float(matrix[col][col])
    multiplicator = -(matrix[row][col]/piv)
    matrix[row] = matrix[col].copy() * multiplicator + matrix[row].copy()
    return matrix

def nullspace(A):
    nullspace = None
    rref_a = rref(A)
    piv_columns = piv_colums(A)
    
    n = len(rref_a)
    m = len(rref_a[0])
    
    free_columns = set(range(0, m)) - set(piv_columns)
    linha_de_zeros = np.zeros(m)
    
    for column in free_columns:
        matrix = []
        b = []    
        for i in range(0, n):
            if (rref_a[i] == linha_de_zeros).all():
                continue
            row = []
            for j in range(0, m):
                if j in piv_columns:
                    row.append(rref_a[i][j])
                elif j == column:
                    b.append([-rref_a[i][j]])                    
            matrix.append(row)
            
        matrix = np.array(matrix)
        b = np.array(b)
        
        matrix_echelon_form = gauss_elimination(matrix,b)
        solutions = back_substitution(matrix_echelon_form)
        solutions.reverse()
        for column2 in free_columns:
            if column2 == column:
                solutions.insert(column2, 1.0)
            else:
                solutions.insert(column2, 0.0)
        if nullspace is None:
            nullspace=np.array([solutions])
        else:
            nullspace = np.append(arr=nullspace, values=np.array([solutions]), axis=0)                    
    return nullspace

def get_basis(A):
    A = A.astype(np.float32)
    n = len(A)
    m = len(A[0])

    rref_a = rref(A)
    linha_de_zeros = np.zeros(m)    
    basis = np.array([A[0]])
    
    for i in range(1, n):
        if not (rref_a[i] == linha_de_zeros).all():
            basis = np.append(arr=basis, values=[A[i]], axis=0)

    nullspace_a = nullspace(A)
    if not nullspace_a is None:
        basis = np.append(arr=basis, values=nullspace_a, axis=0)
    return basis

def normalize_basis(A):
    n = len(A)
    normalized_basis = A.copy()

    for i in range(0,n):
        normalized_basis[i] = np.true_divide(A[i].copy(), np.linalg.norm(A[i]))

    return normalized_basis
    
def gram_schmidt(A):
    A = A.astype(np.float32)
    # Número de linhas 
    n = len(A)
    # Número de colunas
    m = len(A[0])
    matrix_ortoghonal = np.array([A[0]])
    matrix_ortoghonal = normalize_basis(matrix_ortoghonal)
    
    for i in range(1, n):
        proj_sum = np.zeros(m)
        for j in range(0, i):
            proj_sum = proj_sum + ( matrix_ortoghonal[j] * (np.inner(matrix_ortoghonal[j],A[i]) / np.inner(matrix_ortoghonal[j],matrix_ortoghonal[j])) )
        r = A[i] - proj_sum
        r = normalize_basis([r])
        matrix_ortoghonal = np.append(arr = matrix_ortoghonal, values = r, axis=0)    
    return matrix_ortoghonal;    
    
def mount_householder_matrix(A, j):    
    A = A.astype(np.float32)
    n_A = len(A)
    v = np.zeros(n_A)
    w = np.zeros(n_A)
    v[j+1:] = A[j+1:,j]
    w[j+1] = np.linalg.norm(v)
    n = np.array([(v - w) / np.linalg.norm(v - w)])
    H = np.identity(n_A) - 2 * (n.T @ n)
    H = np.around(H, decimals=5)
    return H

def householder_method(A):
    #print('Matriz original:\n', A)
    n_A = len(A)
    H_acc = np.identity(n_A)
    matrix = A.copy()
    
    for j in range(0, n_A - 2):
        Hj = mount_householder_matrix(matrix, j)
        #print('\n\nMatriz de Householder ( passo', j, '):\n', Hj)

        matrix = np.around(Hj @ matrix @ Hj, decimals=4)
        #print('\nMatriz modificada pela transformação de similaridade ( passo', j, '):\n', matrix)
        
        H_acc = np.around(H_acc @ Hj, decimals=4)
        #print('\nMatriz de Householder acumulada ( passo', j, '):\n', H_acc)
    
    return matrix, H_acc

def mount_householder_matrix_to_qr(A, j):
    A = A.astype(np.float32)
    n_A = len(A)
    v = np.zeros(n_A)
    w = np.zeros(n_A)
    v[j:] = A[j:,j]
    w[j] = np.linalg.norm(v)
    
    if v[j] > 0:
        w[j] = -w[j]
    n = np.array([(v - w) / np.linalg.norm(v - w)])
    H = np.identity(n_A) - 2 * (n.T @ n)
    H = np.around(H, decimals=5)
    return H
    
def qr_decomposition(A):
    n_A = len(A)
    Q = np.identity(n_A)
    R = A.copy()
    
    for j in range(0, n_A - 1):
        Qj = mount_householder_matrix_to_qr(R, j)
        R = Qj @ R
        Q = Q @ Qj        
    return Q, R
    
def qr_method_eigvalues_eigvectors(A, eps=0.0001):
    n_A = len(A)
    P = np.identity(n_A)
    dif_between_ite = 0
    qtd_ite = 0
    
    while True:
        Q, R = qr_decomposition(A)
        A = R @ Q
        P = P @ Q
        
        d_A = np.diagonal(A)
        norm_d_A = np.linalg.norm(d_A)
        
        if math.fabs((norm_d_A - dif_between_ite)) <= eps:
            break;
        else:
            dif_between_ite = norm_d_A;
        
        qtd_ite+=1
        
        if qtd_ite > 400:
            break;
      
    return A, P
