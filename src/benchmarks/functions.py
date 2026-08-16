import numpy as np

def sphere(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))

def rastrigin(x: np.ndarray) -> float:
    n = len(x)
    return float(10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))

def rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2))

def ackley(x: np.ndarray) -> float:
    n = len(x)
    term1 = -20.0 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n))
    term2 = -np.exp(np.sum(np.cos(2 * np.pi * x)) / n)
    return float(term1 + term2 + 20.0 + np.e)

def griewank(x: np.ndarray) -> float:
    i = np.arange(1, len(x) + 1)
    return float(np.sum(x**2) / 4000.0 - np.prod(np.cos(x / np.sqrt(i))) + 1.0)

def schwefel(x: np.ndarray) -> float:
    n = len(x)
    return float(418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x)))))

def zakharov(x: np.ndarray) -> float:
    i = np.arange(1, len(x) + 1)
    sum1 = np.sum(x**2)
    sum2 = np.sum(0.5 * i * x)
    return float(sum1 + sum2**2 + sum2**4)

def levy(x: np.ndarray) -> float:
    w = 1.0 + (x - 1.0) / 4.0
    term1 = np.sin(np.pi * w[0])**2
    term3 = (w[-1] - 1.0)**2 * (1.0 + np.sin(2 * np.pi * w[-1])**2)
    term2 = np.sum((w[:-1] - 1.0)**2 * (1.0 + 10.0 * np.sin(np.pi * w[:-1] + 1.0)**2))
    return float(term1 + term2 + term3)

def styblinski_tang(x: np.ndarray) -> float:
    return float(0.5 * np.sum(x**4 - 16.0 * x**2 + 5.0 * x))

def sum_squares(x: np.ndarray) -> float:
    i = np.arange(1, len(x) + 1)
    return float(np.sum(i * x**2))

def michalewicz(x: np.ndarray, m: int = 10) -> float:
    i = np.arange(1, len(x) + 1)
    return float(-np.sum(np.sin(x) * (np.sin(i * x**2 / np.pi))**(2 * m)))

# Function configurations: metadata mapping to their standard bounds and optimum values
BENCHMARK_FUNCTIONS = {
    "Sphere": {
        "func": sphere,
        "bounds": (-100.0, 100.0),
        "optimum": 0.0
    },
    "Rastrigin": {
        "func": rastrigin,
        "bounds": (-5.12, 5.12),
        "optimum": 0.0
    },
    "Rosenbrock": {
        "func": rosenbrock,
        "bounds": (-30.0, 30.0),
        "optimum": 0.0
    },
    "Ackley": {
        "func": ackley,
        "bounds": (-32.768, 32.768),
        "optimum": 0.0
    },
    "Griewank": {
        "func": griewank,
        "bounds": (-600.0, 600.0),
        "optimum": 0.0
    },
    "Schwefel": {
        "func": schwefel,
        "bounds": (-500.0, 500.0),
        "optimum": 0.0
    },
    "Zakharov": {
        "func": zakharov,
        "bounds": (-5.0, 10.0),
        "optimum": 0.0
    },
    "Levy": {
        "func": levy,
        "bounds": (-10.0, 10.0),
        "optimum": 0.0
    },
    "Styblinski-Tang": {
        "func": styblinski_tang,
        "bounds": (-5.0, 5.0),
        "optimum": -39.16617 * 30  # Depends on dimensionality (approx for 30D is -1174.98)
    },
    "Sum Squares": {
        "func": sum_squares,
        "bounds": (-10.0, 10.0),
        "optimum": 0.0
    },
    "Michalewicz": {
        "func": michalewicz,
        "bounds": (0.0, np.pi),
        "optimum": -29.6309  # for m=10, D=30
    }
}
