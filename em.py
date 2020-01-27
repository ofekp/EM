#!/usr/bin/env python

"""
Created by Ofek P Jan 2019
Interpreter Python 3.7
"""

import time
import numpy as np
import matplotlib.pyplot as plt


def get_samples(num_of_samples):
    d = 2  # dimension
    c_1 = 0.5
    c_2 = 0.5
    vec = [1, 2]
    sigma_1 = np.diag(vec)
    vec = [2, 0.5]
    sigma_2 = np.diag(vec)
    u_1 = np.array([1, 1])
    u_2 = np.array([3, 3])

    s1 = np.random.multivariate_normal(u_1, sigma_1, num_of_samples).T
    s2 = np.random.multivariate_normal(u_2, sigma_2, num_of_samples).T

    c = np.random.choice(2, num_of_samples, p = [c_1, c_2])
    return c * s1 + (1 - c) * s2


def prob_norm(x, mean, cov):
    d = 2  # dimension
    return (1 / (np.power(np.sqrt(2*np.pi), d) * np.linalg.det(cov))) * np.exp((-0.5) * np.transpose(x - mean) * np.linalg.inv(cov) * (x - mean))


def main():
    d = 2  # dimension
    num_of_samples = 2000
    max_num_of_iterations = 110
    print_at = [2, 10, 100]

    c_1 = np.random.uniform()
    c_2 = np.random.uniform()
    cov_1 = np.diag(5.0 * np.random.rand(2))
    cov_2 = np.diag(5.0 * np.random.rand(2))
    mean_1 = 5.0 * np.random.rand(2)
    mean_2 = 5.0 * np.random.rand(2)

    # generate samples
    samples = get_samples(num_of_samples)  # 2 X 2000
    plt.plot(samples[0:1], samples[1:2], 'x')
    plt.show()

    # EM
    for i in range(max_num_of_iterations):
        # E-step

        samples_minus_mean_1 = samples - (np.ones([d, num_of_samples]).T * mean_1).T  # 2 X 2000
        p_alpha_1 = (1 / (np.power(np.sqrt(2 * np.pi), d) * np.linalg.det(cov_1))) * np.exp(
            (-0.5) * np.diag(np.dot(samples_minus_mean_1.T, np.dot(np.linalg.inv(cov_1), samples_minus_mean_1))))
        samples_minus_mean_2 = samples - (np.ones([d, num_of_samples]).T * mean_2).T  # 2 X 2000
        p_alpha_2 = (1 / (np.power(np.sqrt(2 * np.pi), d) * np.linalg.det(cov_2))) * np.exp(
            (-0.5) * np.diag(np.dot(samples_minus_mean_2.T, np.dot(np.linalg.inv(cov_2), samples_minus_mean_2))))

        alpha_1 = c_1 * p_alpha_1 / (c_1 * p_alpha_1 + c_2 * p_alpha_2)
        alpha_2 = c_2 * p_alpha_2 / (c_1 * p_alpha_1 + c_2 * p_alpha_2)

        # M - step
        # maximize c
        c_1 = np.sum(alpha_1) / num_of_samples
        c_2 = np.sum(alpha_2) / num_of_samples

        # maximize mean
        mean_1 = np.sum(samples * alpha_1, 1) / np.sum(alpha_1)
        mean_2 = np.sum(samples * alpha_2, 1) / np.sum(alpha_2)

        # maximize cov
        cov_1_1 = np.sum(np.power(samples[0], 2) * alpha_1) / np.sum(alpha_1) - np.power(mean_1[0], 2)
        cov_1_2 = np.sum(np.power(samples[1], 2) * alpha_1) / np.sum(alpha_1) - np.power(mean_1[1], 2)
        cov_1 = np.diag(np.array([cov_1_1, cov_1_2]))

        cov_2_1 = np.sum(np.power(samples[0], 2) * alpha_2) / np.sum(alpha_2) - np.power(mean_2[0], 2)
        cov_2_2 = np.sum(np.power(samples[1], 2) * alpha_2) / np.sum(alpha_2) - np.power(mean_2[1], 2)
        cov_2 = np.diag(np.array([cov_2_1, cov_2_2]))

        if i in print_at:
            print(str(mean_1) + ", " + str(mean_2))
            print(str(cov_1) + ", " + str(cov_2))
            print("--")

    # for initialization take 2 samples as the means
    mean_1 = samples[:, [0]]
    mean_2 = samples[:, [1]]
    classified_samples = np.zeros([1, num_of_samples])  # classify all as 0

    # K-Means
    for i in range(max_num_of_iterations):
        dist_1 = np.diag(np.dot((samples - mean_1).T, samples - mean_1))
        dist_2 = np.diag(np.dot((samples - mean_2).T, samples - mean_2))

        new_classified_samples = np.argmin(np.array([dist_1, dist_2]), axis=0)
        # handle dist_1 == dist_2 - do not change the class in this case
        eq_indices = (dist_1 == dist_2).astype(int)
        classified_samples = classified_samples * eq_indices + new_classified_samples * (1 - eq_indices)
        classified_samples = classified_samples.reshape((2000,))

        # calc new means
        classified_1 = samples[:, classified_samples == 0]
        mean_1 = np.sum(classified_1, axis=1, keepdims=True) / np.shape(classified_1)[1]
        classified_2 = samples[:, classified_samples == 1]
        mean_2 = np.sum(classified_2, axis=1, keepdims=True) / np.shape(classified_2)[1]

        if i in print_at:
            print(str(mean_1.T) + ", " + str(mean_2.T))
            print("--")


start_time = time.time()
main()
elapsed_time_sec = time.time() - start_time
elapsed_time_sec = float("{0:.2f}".format(elapsed_time_sec))
print("DONE! Elapsed time [" + str(elapsed_time_sec) + " sec]")