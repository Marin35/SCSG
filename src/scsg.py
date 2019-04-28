# Class for the SCSG algorithm
import numpy as np

class SCSG:
    """
    Implement the SCSG algorithm.

    Attributes:
        number_stages_T: An integer that tells the number of stages.
        x_hat: A vector which is the initial iterate.
        step_sizes: An array of float which contains the steps.
        batch_sizes: An array of int which contains batch.
        mini_batch_sizes: An array of int which contains mini batch size.
        f: An array which consists of a finite sum of functions.
        n: An integer which is the number of elements.
    """
    def __init__(self, number_stages_T, x_0, step_sizes, batch_sizes, mini_batch_sizes, f, n):
        """Initialize the class object."""
        self.number_stages_T = number_stages_T
        self.x_0 = x_0
        self.step_sizes = step_sizes
        self.batch_sizes = batch_sizes
        self.mini_batch_sizes = mini_batch_sizes
        self.n = n
        self.f = f

        self.list_of_x_hat = np.array([x_0])


    def run(self, PL_case=True):
        """Run the SCSG algorithm."""
        x_k = 3
        for j in range(1, self.number_stages_T):

            B_j, b_j = self.batch_sizes[j - 1], self.mini_batch_sizes[j - 1]
            I_j = np.random.choice((1, self.n), B_j)
            g_j = 5  # gradient differentiation

            x_hat = self.list_of_x_hat[j - 1]
            x_0 = x_hat

            N_j = np.random.geometric(p=B_j / (B_j + b_j))

            for k in range(1, N_j):
                I_hat = np.random.choice((1, self.n), b_j)

                nu = 3  # gradient differentiation

                x_k = x_0 - self.step_sizes[j - 1] * nu
                x_0 = x_k  # For next iteration

            self.list_of_x_hat = np.append(self.list_of_x_hat, x_k)  # Add the last one to the array of x hat

            if PL_case:
                return self.list_of_x_hat[-1]  # Return the last one
