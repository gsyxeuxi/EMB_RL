import numpy as np

class FI_matrix(object):

    def __init__(self) -> None:
        # Define parameters
        self.J = 4.624e-06  # Moment of inertia
        self.km = 21.7e-03  # Motor constant
        self.gamma = 1.889e-05  # Proportional constant
        self.k1 = 23.04  # Elasticity constant
        self.fc = 10.37e-3  # Coulomb friction coefficient
        self.epsilon = 0.5  # Zero velocity bound
        self.fv = 2.16e-5  # Viscous friction coefficient
        self.Ts = 1.2 * (self.fc + self.fv * self.epsilon) #Static friction torque
        self.dt = 0.001

    def f(self, x, u):
        """
        Define the system dynamics of lumped parameter EMB model
        dx/dt = f(x, u, theta)
        state variable:
            x1: motor position
            x2: motor velocity
        input:
            u: motor current
        parameter:
            theta: km, k1, fc, fv, J, gamma, epsilon
        """
        x1, x2 = x
        dx1 = x2
        #when there is clamp force, x1>0
        if x1 > 0 and x2 > self.epsilon:
            dx2 = (self.km / self.J) * u - (self.gamma * self.k1 / self.J) * x1 - (1 / self.J) * (self.fc + self.fv * x2)
        if x1 > 0 and x2 < -self.epsilon:
            dx2 = (self.km / self.J) * u - (self.gamma * self.k1 / self.J) * x1 - (1 / self.J) * (-self.fc + self.fv * x2)
        if x1 > 0 and abs(x2) <= self.epsilon:
            if (self.km / self.J) * u - (self.gamma * self.k1 / self.J) * x1 > self.Ts: #overcome the maximum static friction
                dx2 = (self.km / self.J) * u - (self.gamma * self.k1 / self.J) * x1 - self.Ts
            else: #lockup
                dx2 = 0
        #when there is no clamp force, x1<=0
        if x1 <= 0 and x2 > self.epsilon:
            dx2 = (self.km / self.J) * u - (1 / self.J) * (self.fc + self.fv * x2)
        if x1 <= 0 and x2 < -self.epsilon:
            dx2 = (self.km / self.J) * u - (1 / self.J) * (-self.fc + self.fv * x2)
        if x1 <= 0 and abs(x2) <= self.epsilon:
            if (self.km / self.J) * u > self.Ts: #overcome the maximum static friction
                dx2 = (self.km / self.J) * u - self.Ts
            else: #lockup
                dx2 = 0
        return np.array([dx1, dx2])

    def jacobian_f(self, x, u):
        """
        Define the Jacobian matrix of function f, J_f
        dx/dt = f(x, u, theta)
        output: J_f
        """
        x1, x2 = x
        df1_dx = [0, 1]

        if x1 > 0 and x2 > self.epsilon:
            df2_dx = [-self.gamma * self.k1 / self.J, -1 / self.J * self.fv]
        if x1 > 0 and x2 < -self.epsilon:
            df2_dx = [-self.gamma * self.k1 / self.J, -1 / self.J * self.fv]
        if x1 > 0 and abs(x2) <= self.epsilon:
            if (self.km / self.J) * u - (self.gamma * self.k1 / self.J) * x1 > self.Ts: #overcome the maximum static friction
                df2_dx = [-self.gamma * self.k1 / self.J, 0]
            else: #lockup
                df2_dx = [0, 0]
        #when there is no clamp force, x1<=0
        if x1 <= 0 and x2 > self.epsilon:
            df2_dx = [0, -1 / self.J * self.fv]
        if x1 <= 0 and x2 < -self.epsilon:
            df2_dx = [0, -1 / self.J * self.fv]
        if x1 <= 0 and abs(x2) <= self.epsilon:
            if (self.km / self.J) * u > self.Ts: #overcome the maximum static friction
                df2_dx = [0, 0]
            else: #lockup
                df2_dx = [0, 0]
        return np.array([df1_dx, df2_dx])
    
    def finite_difference_approximation(self, x, u, d, epsilon):
        return (self.f(x + epsilon * d, u) - self.f(x - epsilon * d, u)) / (2 * epsilon)

x0 = np.array([1.0, 0.47])
fi_matrix = FI_matrix()
u = 1
x = x0
det_T = 0.001
epsilon = 1e-6
d = np.random.rand(2,)
dx = fi_matrix.f(x, u)
x = x + det_T * dx
J_h = fi_matrix.jacobian_f(x, u)
fd_approximation = fi_matrix.finite_difference_approximation(x, u, d, epsilon)
if (np.linalg.norm(fd_approximation - np.dot(d, J_h.T) < 1e-6)):
    print('analytical equal to Finite-Difference Approximations')
    