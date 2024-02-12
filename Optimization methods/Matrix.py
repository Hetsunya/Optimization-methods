import numpy as np


def transpose_matrix(matrix):
    return matrix.T


def calculate_products(matrix_a, vector_u):
    transpose_a = transpose_matrix(matrix_a)
    product_ata = np.dot(transpose_a, matrix_a)
    product_atu = np.dot(transpose_a, vector_u)
    return product_ata, product_atu


def create_augmented_matrix(product_ata, product_atu):
    return np.column_stack((product_ata, product_atu))


def gaussian_elimination(augmented_matrix):
    n = len(augmented_matrix)

    for i in range(n):
        max_row = np.argmax(np.abs(augmented_matrix[i:, i])) + i
        augmented_matrix[[i, max_row]] = augmented_matrix[[max_row, i]]

        for j in range(i + 1, n):
            ratio = augmented_matrix[j, i] / augmented_matrix[i, i]
            augmented_matrix[j, i:] -= ratio * augmented_matrix[i, i:]

    return augmented_matrix


def back_substitution(augmented_matrix):
    n = len(augmented_matrix)
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = (augmented_matrix[i, -1] - np.dot(augmented_matrix[i, i + 1:-1], x[i + 1:])) / augmented_matrix[i, i]

    return x


def main():
    matrix_a = np.array([[2, 1],
                         [1, -1],
                         [0, 1]], dtype=float)
    vector_u = np.array([1, 0, 1], dtype=float)

    product_ata, product_atu = calculate_products(matrix_a, vector_u)
    augmented_matrix = create_augmented_matrix(product_ata, product_atu)

    augmented_matrix = gaussian_elimination(augmented_matrix)

    x = back_substitution(augmented_matrix)

    print("Решение системы линейных уравнений методом Гаусса:")
    print(x)


if __name__ == "__main__":
    main()
