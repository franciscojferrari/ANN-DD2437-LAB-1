import numpy as np


def generate_linear_data(N: int, mA: list, mB: list, sigmaA: float, sigmaB: float, target_values = [-1, 1]):
    covA = np.zeros((2, 2))
    np.fill_diagonal(covA, sigmaA)
    covB = np.zeros((2, 2))
    np.fill_diagonal(covB, sigmaB)

    classA = np.random.multivariate_normal(mA, covA, N)
    classB = np.random.multivariate_normal(mB, covB, N)
    inputs = np.concatenate((classA, classB))
    targets = np.concatenate((target_values[0] * np.ones(classA.shape[0]), target_values[1] * np.ones(classB.shape[0])))

    inputs = np.append(inputs, np.ones((inputs.shape[0], 1)), axis = 1)

    indices = np.arange(inputs.shape[0])
    np.random.shuffle(indices)
    inputs = inputs[indices]
    targets = targets[indices]

    return {"inputs": inputs.T, "targets": np.atleast_2d(targets)}


def generate_nonlinear_data(N: int, mA: list, mB: list, sigmaA: float, sigmaB: float, target_values = [-1, 1]):
    covA = np.zeros((2, 2))
    np.fill_diagonal(covA, sigmaA)
    covB = np.zeros((2, 2))
    np.fill_diagonal(covB, sigmaB)

    classA_1 = np.random.multivariate_normal(mA, covA, N // 2)
    mA_2 = np.array(mA) * [-1, 1]
    classA_2 = np.random.multivariate_normal(mA_2, covA, N // 2)
    classA = np.concatenate((classA_1, classA_2))

    classB = np.random.multivariate_normal(mB, covB, N)
    inputs = np.concatenate((classA, classB))
    targets = np.concatenate((target_values[0] * np.ones(classA.shape[0]), target_values[1] * np.ones(classB.shape[0])))

    inputs = np.append(inputs, np.ones((inputs.shape[0], 1)), axis = 1)

    indices = np.arange(inputs.shape[0])
    np.random.shuffle(indices)
    inputs = inputs[indices]
    targets = targets[indices]
    return {"inputs": inputs.T, "targets": targets}
