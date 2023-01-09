import numpy as np
import kernel as Kernel

class SMO:
    def __init__(self, C=1.0, kernel='linear', epsilon=1e-3, random_state=None):
        self.C = C
        self.kernel = kernel
        self.epsilon = epsilon
        self.count = 0
        np.random.seed(random_state)

    def is_satisfied_KKT(self, i):
        """Check if the ith example satisfies the KKT conditions.

        Args:
            i (int): The index of the example.

        Returns:
            bool: True if the ith example satisfies the KKT conditions, False otherwise.
        """
        y_i = self.y[i]
        alpha_i = self.alpha[i]
        error_i = self.error_cache[i]
        if (alpha_i < self.C and y_i * error_i < -self.epsilon) or (alpha_i > 0 and y_i * error_i > self.epsilon):
            return False
        else:
            return True

    def initialize_parameters(self, X, y):
        """Initialize the parameters of the SVM.

        Args:
            X (numpy.ndarray): The input data.
            y (numpy.ndarray): The target values.
        """
        self.m, self.n = X.shape
        self.X = X
        self.y = y
        # Initialize the alphas
        self.alpha = np.zeros(self.m)
        # Initialize the bias term
        self.b = 0
        # Initialize the error cache
        self.error_cache = np.zeros(self.m)
        for i in range(self.m):
            self.error_cache[i] = self.update_error_cache(i)

        # Initialize weight vector for linear kernel
        if self.kernel == 'linear':
            # Initialize the weights
            self.w = np.zeros(self.n)
        
    def choose_second_alpha(self, i):
        """Choose the second Lagrange multiplier to optimize.

        Args:
            i (int): The index of the first Lagrange multiplier.

        Returns:
            int: The index of the second Lagrange multiplier.
        """
        error_i = self.error_cache[i]
        j = np.argmax(np.abs(self.error_cache - error_i))
        return j

    def examine_example(self, i):
        """Examine an example to optimize the Lagrange multipliers.

        Args:
            i (int): The index of the example to examine.

        Returns:
            int: 1 if an example is optimized, 0 otherwise.
        """
        if len(np.where((self.alpha > 0) & (self.alpha < self.C))[0]) > 1:
            # Choose the second alpha
            j = self.choose_second_alpha(i)
            if self.take_step(j, i):
                return 1

        # Loop over all non-zero and non-C alphas, starting at a random point
        for j in np.random.permutation(np.where((self.alpha > 0) & (self.alpha < self.C))[0]):
            if self.take_step(j, i):
                return 1

        # Loop over all possible i and j, starting at a random point
        for j in np.random.permutation(self.m):
            if self.take_step(j, i):
                return 1
        return 0

    def compute_L_H(self, i, j):
        """Compute the lower and upper bounds for the second Lagrange multiplier.

        Args:
            i (int): The index of the first Lagrange multiplier.
            j (int): The index of the second Lagrange multiplier.

        Returns:
            tuple: The lower and upper bounds for the second Lagrange multiplier.
        """
        if self.y[i] == self.y[j]:
            L = max(0, self.alpha[i] + self.alpha[j] - self.C)
            H = min(self.C, self.alpha[i] + self.alpha[j])
        else:
            L = max(0, self.alpha[j] - self.alpha[i])
            H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
        return L, H

    def clip_new_alpha(self, alpha, L, H):
        """Clip the new value for the second Lagrange multiplier.

        Args:
            alpha (float): The new value for the second Lagrange multiplier.
            L (float): The lower bound for the second Lagrange multiplier.
            H (float): The upper bound for the second Lagrange multiplier.

        Returns:
            float: The clipped value for the second Lagrange multiplier.
        """
        if alpha <= L:
            return L
        elif alpha >= H:
            return H
        else:
            return alpha

    def take_step(self, i1, i2):
        """Take a step to optimize the Lagrange multipliers.

        Args:
            i1 (int): The index of the first Lagrange multiplier.
            i2 (int): The index of the second Lagrange multiplier.

        Returns:
            int: 1 if the step is successful, 0 otherwise.
        """
        self.count += 1
        if i1 == i2:
            return 0
        s = self.y[i1] * self.y[i2]
        L, H = self.compute_L_H(i1, i2)
        k11 = self.kernel_func(self.X[i1], self.X[i1])
        k12 = self.kernel_func(self.X[i1], self.X[i2])
        k22 = self.kernel_func(self.X[i2], self.X[i2])
        eta = k11 + k22 - 2 * k12
        if eta > 0:
            # Compute the new value for alpha2
            alpha2_new = self.alpha[i2] + self.y[i2] * (self.error_cache[i1] - self.error_cache[i2]) / eta
            # Clip alpha2
            alpha2_new = self.clip_new_alpha(alpha2_new, L, H)
        else:
            # Compute the objective function at the bounds
            f1 = self.y[i1] * (self.error_cache[i1] + self.b) - self.alpha[i1] * k11 - s * self.alpha[i2] * k12
            f2 = self.y[i2] * (self.error_cache[i2] + self.b) - s * self.alpha[i1] * k12 - self.alpha[i2] * k22
            L1 = self.alpha[i1] + s * (self.alpha[i2] - L)
            H1 = self.alpha[i1] + s * (self.alpha[i2] - H)
            Lobj = L1 * f1 + L * f2 + 0.5 * L1 ** 2 * k11 + 0.5 * L ** 2 * k22 + s * L * L1 * k12
            Hobj = H1 * f1 + H * f2 + 0.5 * H1 ** 2 * k11 + 0.5 * H ** 2 * k22 + s * H * H1 * k12
            # Choose the bound that gives the maximum objective function
            if Lobj < Hobj - self.epsilon:
                alpha2_new = L
            elif Lobj > Hobj + self.epsilon:
                alpha2_new = H
            else:
                alpha2_new = self.alpha[i2]
        # Check if the change in alpha2 is too small
        if abs(alpha2_new - self.alpha[i2]) < self.epsilon * (alpha2_new + self.alpha[i2] + self.epsilon):
            return 0
        alpha1_new = self.alpha[i1] + s * (self.alpha[i2] - alpha2_new)

        # Update threshold to reflect change in Lagrange multipliers
        b1 = self.error_cache[i1] + self.y[i1] * (alpha1_new - self.alpha[i1]) * k11 + self.y[i2] * (
                    alpha2_new - self.alpha[i2]) * k12 + self.b
        b2 = self.error_cache[i2] + self.y[i1] * (alpha1_new - self.alpha[i1]) * k12 + self.y[i2] * (
                    alpha2_new - self.alpha[i2]) * k22 + self.b
        b_new = (b1 + b2) / 2
        self.b = b_new

        # Update weight vector to reflect change in a1 & a2, if linear SVM
        if self.kernel == 'linear':
            self.w = self.w + self.y[i1] * (alpha1_new - self.alpha[i1]) * self.X[i1] + self.y[i2] * (alpha2_new - self.alpha[i2]) * \
                     self.X[i2]

        # Update error cache using new Lagrange multipliers
        self.error_cache[i1] = self.update_error_cache(i1)
        self.error_cache[i2] = self.update_error_cache(i2)

        # Store the new Lagrange multipliers in the alpha array
        self.alpha[i1] = alpha1_new
        self.alpha[i2] = alpha2_new

        return 1

    def fit(self, X, y):
        """Fit the model using SMO algorithm.

        Args:
            X (numpy.ndarray): The input data.
            y (numpy.ndarray): The target values.
        """
        gamma = 1 / X.shape[1]
        self.kernel_func = Kernel.get_kernel(self.kernel, gamma=gamma)
        self.initialize_parameters(X, y)
        num_changed = 0
        examine_all = True
        while num_changed > 0 or examine_all:
            num_changed = 0
            if examine_all:
                for i in range(self.m):
                    if not self.is_satisfied_KKT(i):
                        num_changed += self.examine_example(i)
                examine_all = False
            else:
                for i in np.where((self.alpha > 0) & (self.alpha < self.C))[0]:
                    if not self.is_satisfied_KKT(i):
                        num_changed += self.examine_example(i)
                examine_all = True

    def update_error_cache(self, i):
        """Update the error cache for the i-th example.

        Args:
            i (int): The index of the example.

        Returns:
            float: The error of the i-th example.
        """
        u_i = int(np.sum(self.alpha * self.y * self.kernel_func(self.X, self.X[i])) - self.b)
        return u_i - self.y[i]

    def predict(self, X):
        """Predict the class labels for the provided data.

        Args:
            X (numpy.ndarray): The input data.

        Returns:
            numpy.ndarray: The predicted class labels.
        """
        if self.kernel == 'linear':
            return np.sign(np.dot(X, self.w) - self.b)
        y_pred = []
        for x in X:
            result = -self.b
            for i in range(len(self.X)):
                result += self.alpha[i] * self.y[i] * self.kernel_func(self.X[i], x)
            y_pred.append(np.sign(result))
        return np.array(y_pred)

    def score(self, X, y):
        """Compute the accuracy of the model.

        Args:
            X (numpy.ndarray): The input data.
            y (numpy.ndarray): The target values.

        Returns:
            float: The accuracy of the model.
        """
        return np.mean(self.predict(X) == y)
