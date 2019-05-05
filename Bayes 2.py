import numpy as np

X = np.array([
    ['Cerah', 'Panas', 'Tinggi', 'Stabil', 'Tidak'], # 1
    ['Cerah', 'Panas', 'Tinggi', 'Labil', 'Tidak'], # 2
    ['Mendung', 'Panas', 'Tinggi', 'Stabil', 'Ya'], # 3
    ['Hujan', 'Sedang', 'Tinggi', 'Stabil', 'Ya'], # 4
    ['Hujan', 'Dingin', 'Normal', 'Stabil', 'Ya'], # 5
    ['Hujan', 'Dingin', 'Normal', 'Labil', 'Tidak'], # 6
    ['Mendung', 'Dingin', 'Normal', 'Labil', 'Ya'], # 7
    ['Cerah', 'Dingin', 'Tinggi', 'Stabil', 'Tidak'], # 8
    ['Cerah', 'Dingin', 'Normal', 'Stabil', 'Ya'], # 9
    ['Hujan', 'Sedang', 'Normal', 'Stabil', 'Ya'], # 10
    ['Cerah', 'Sedang', 'Normal', 'Labil', 'Ya'], # 11
    ['Mendung', 'Sedang', 'Tinggi', 'Labil', 'Ya'], # 12
    ['Mendung', 'Panas', 'Normal', 'Stabil', 'Ya'], # 13
    ['Hujan', 'Sedang', 'Tinggi', 'Labil', 'Tidak'], # 14
])
test = np.array(['Cerah', 'Sedang', 'Tinggi', 'Stabil'])


def compute_prior(X):
    target_idx = len(X[0]) - 1
    classes = {}
    for x in X:
        c = x[target_idx]
        if c not in classes:
            classes[c] = 1
        else:
            classes[c] += 1
    n_total = sum(classes.values())
    for c in classes.keys():
        classes[c] /= n_total
    return classes


def compute_likelihood(X, test):
    target_idx = len(X[0]) - 1
    classes = set([x[target_idx] for x in X])
    print(classes)
    n_feature = len(X[0]) - 1

    likelihoods = {}
    for c in classes:
        likelihoods[c] = [0] * n_feature
        with_class_c = list(filter(lambda x: x[-1] == c, X))
        print(with_class_c)
        for i in range(n_feature):
            test_feature_val = test[i]
            for x in with_class_c:
                if x[i] == test_feature_val:
                    likelihoods[c][i] += 1
        n_instance = len(with_class_c)
        print(likelihoods[c])
        likelihoods[c] = [i / n_instance for i in likelihoods[c]]
    return likelihoods


def compute_posterior(prior, likelihood):
    posterior = {}
    for c in prior.keys():
        posterior[c] = np.product(likelihood[c] + [prior[c]])
    return posterior


prior = compute_prior(X)
print(prior)
likelihood = compute_likelihood(X, test)
print(likelihood)
print(compute_posterior(prior, likelihood))