import pandas as pd
import numpy as np
from matplotlib import pyplot
from termcolor import colored


def featureNormalize(feature):
    X_normalized = feature.copy()
    mu_mean = np.mean(X_normalized, axis=0)
    sigma_deviation = np.std(X_normalized, axis=0)
    X_normalized = X_normalized - mu_mean
    X_normalized = np.true_divide(X_normalized, sigma_deviation)
    return X_normalized, mu_mean, sigma_deviation


def computeCostMultiT(input_X, price, theta):
    m_size = price.shape[0]
    h = input_X.dot(theta)
    t = h - price
    t = np.power(t, 2)
    t = t.sum()
    J = t / (2 * m_size)
    return J


def computeCostMulti(input_X, price, theta, lambda_l):
    m_size = price.shape[0]
    h = input_X.dot(theta)
    t = h - price
    t = np.power(t, 2)
    t = t.sum()
    J = t / (2 * m_size)
    temp_theta = theta
    temp_theta = np.delete(temp_theta, 0)
    theta_power_2 = np.power(temp_theta, 2)
    s = np.sum(theta_power_2)
    J = J + lambda_l * s / (m_size * 2)
    return J


def gradientDescentMulti(input_X, price, theta, alpha_rate, num_of_iterations, lambda_l):
    m_size = price.shape[0]
    theta = theta.copy()
    J_history = []
    for term in range(num_of_iterations):
        error = (input_X.dot(theta)) - price
        temp = theta[0] - ((alpha_rate / m_size) * (np.sum(error)))
        theta = theta * (1 - (alpha_rate * lambda_l / m_size)) - ((alpha_rate / m_size) * (np.dot(input_X.T, error)))
        theta[0] = temp
        J_history.append(computeCostMulti(input_X, price, theta, lambda_l))
    return theta, J_history


def plotData(x, price, s1, s2):
    pyplot.plot(x, price, 'ro', ms=10, mec='k')
    pyplot.xlabel(s1)
    pyplot.ylabel(s2)
    pyplot.show()


# Loading Data
data = pd.read_csv("house_data_complete.csv")
data.head()
dataFrame = pd.DataFrame(data)
y = dataFrame.loc[:, 'price']
m = y.size  # training examples number
bed = dataFrame['bedrooms'].to_numpy().T
grade = dataFrame['grade'].to_numpy().T
floors = dataFrame['floors'].to_numpy().T
X = np.column_stack((bed, grade, floors))
# Displaying Data
plotData(dataFrame['bedrooms'], y, '# of Bedrooms', 'Price')
plotData(dataFrame['grade'], y, 'Grade', 'Price')
plotData(dataFrame['floors'], y, '# of Floors', 'Price')

# Dividing Data into {Training, Cross Validation, testing}
m_train = int(0.6*m)  # from 0 to 12,967                  60%
m_cv = m_test = int((m-m_train)/2)  # 8646/2 = 4323
cv_pointer = m_train+1  # from 12,968 to 17,290           20%
test_pointer = m_train+m_cv+1  # from 17,291 to 21,613    20%
# shuffling Data in case it is in an ascending form
np.random.shuffle(X)
print(colored("Size of X : ", 'red'))
print(np.shape(X))

# Normalizing Data
X_norm, mu, sigma = featureNormalize(X)
print(colored("Mean is: " + str(mu), 'green'))
print(colored("Standard Deviation is: " + str(sigma), 'green'))
# first hypo: linear X1, X2, X3
X1 = X_norm.copy()
# second hypo: Quad X1, X2, X3, X1^2, X2^2, X3^2
X2 = X_norm.copy()
X2 = np.column_stack((X2, np.power(X2[:, 0], 2)))
X2 = np.column_stack((X2, np.power(X2[:, 1], 2)))
X2 = np.column_stack((X2, np.power(X2[:, 2], 2)))
# third hypo: Cubic X1, X2, X3, X1X2, X1X3, X3^3
X3 = X_norm.copy()
X3 = np.column_stack((X3, np.multiply(X3[:, 0], X3[:, 1])))
X3 = np.column_stack((X3, np.multiply(X3[:, 0], X3[:, 2])))
X3 = np.column_stack((X3, np.power(X3[:, 2], 3)))

X_norm1 = np.column_stack(([np.ones(m), X1]))  # Adding X0
print(colored("Size of X adding X0: " + str(np.shape(X_norm1)), 'red'))
X_norm2 = np.column_stack(([np.ones(m), X2]))  # Adding X0
print(colored("Size of X adding X0: " + str(np.shape(X_norm2)), 'red'))
X_norm3 = np.column_stack(([np.ones(m), X3]))  # Adding X0
print(colored("Size of X adding X0: " + str(np.shape(X_norm3)), 'red'))

# Applying Cross Validation
alpha = 0.01
lambda_ = 5
num_iterations = 1500

theta1 = np.zeros(np.shape(X_norm1)[1])
theta2 = np.zeros(np.shape(X_norm2)[1])
theta3 = np.zeros(np.shape(X_norm3)[1])
theta1, J_history1 = gradientDescentMulti(X_norm1[0:cv_pointer-1, :], y[0:cv_pointer-1], 
                                          theta1, alpha, num_iterations, lambda_)
theta2, J_history2 = gradientDescentMulti(X_norm2[0:cv_pointer-1, :], y[0:cv_pointer-1], 
                                          theta2, alpha, num_iterations, lambda_)
theta3, J_history3 = gradientDescentMulti(X_norm3[0:cv_pointer-1, :], y[0:cv_pointer-1], 
                                          theta3, alpha, num_iterations, lambda_)
pyplot.plot(np.arange(len(J_history1)), J_history1, lw=2)
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Cost  J1')
pyplot.show()
print(colored('Theta from GD1 1st Fold: {:s}'.format(str(theta1)), 'magenta'))

pyplot.plot(np.arange(len(J_history2)), J_history2, lw=2)
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Cost  J2')
pyplot.show()
print(colored('Theta from GD2 1st Fold: {:s}'.format(str(theta2)), 'magenta'))

pyplot.plot(np.arange(len(J_history3)), J_history3, lw=2)
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Cost  J3')
pyplot.show()
print(colored('Theta from GD3 1st Fold: {:s}'.format(str(theta3)), 'magenta'))

# Decide which hypo with min COST
print(colored("ERROR J_cv for 3 thetas 1st Fold : ", 'blue'))
thetas = np.array([theta1, theta2, theta3])
J_cvs = np.array([computeCostMultiT(X_norm1[cv_pointer:test_pointer, :], y[cv_pointer:test_pointer], theta1),
                  computeCostMultiT(X_norm2[cv_pointer:test_pointer, :], y[cv_pointer:test_pointer], theta2),
                  computeCostMultiT(X_norm3[cv_pointer:test_pointer, :], y[cv_pointer:test_pointer], theta3)])
print(colored(J_cvs, 'blue'))
least_error_pos = J_cvs.argmin()
print(colored(" Best theta is of hypo " + str(least_error_pos+1) + ", with theta:" +
              str(thetas[least_error_pos]), 'green'))
# Compute & Estimate J_test
if least_error_pos == 0:
    temp_norm = X_norm1
elif least_error_pos == 1:
    temp_norm = X_norm2
else:
    temp_norm = X_norm3
J_test = computeCostMultiT(temp_norm[test_pointer:, :], y[test_pointer:], thetas[least_error_pos])
print(colored("Estimate of J_test is :" + str(J_test), 'red'))
J_individual = (temp_norm[test_pointer:, :]).dot(thetas[least_error_pos])
J_individual = J_individual.T - y[test_pointer:]
J_individual = np.absolute(J_individual)
print("Average of J_test Error is : " + str(np.average(J_individual)))
print("with minimum Error : " + str(np.min(J_individual)))
print("Samples from difference between Reality and Prediction" + str(np.sort(J_individual)[0:7]))
for i in range(5):
    print(colored("----------------------------", 'yellow'))

# Second Fold
alpha = 0.01
lambda_ = 1
num_iterations = 1000

theta1 = np.zeros(np.shape(X_norm1)[1])
theta2 = np.zeros(np.shape(X_norm2)[1])
theta3 = np.zeros(np.shape(X_norm3)[1])
theta1, J_history1 = gradientDescentMulti(X_norm1[m_cv:m_cv+m_train, :], y[m_cv:m_cv+m_train],
                                          theta1, alpha, num_iterations, lambda_)
theta2, J_history2 = gradientDescentMulti(X_norm2[m_cv:m_cv+m_train, :], y[m_cv:m_cv+m_train],
                                          theta2, alpha, num_iterations, lambda_)
theta3, J_history3 = gradientDescentMulti(X_norm3[m_cv:m_cv+m_train, :], y[m_cv:m_cv+m_train],
                                          theta3, alpha, num_iterations, lambda_)
pyplot.plot(np.arange(len(J_history1)), J_history1, lw=2)
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Cost  J1')
pyplot.show()
print(colored('Theta from GD1 2nd Fold: {:s}'.format(str(theta1)), 'magenta'))
pyplot.plot(np.arange(len(J_history2)), J_history2, lw=2)
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Cost  J2')
pyplot.show()
print(colored('Theta from GD2 2nd Fold: {:s}'.format(str(theta2)), 'magenta'))
pyplot.plot(np.arange(len(J_history3)), J_history3, lw=2)
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Cost  J3')
pyplot.show()
print(colored('Theta from GD3 2nd Fold: {:s}'.format(str(theta3)), 'magenta'))
# Decide which hypo with min COST
print(colored("ERROR J_cv for 3 thetas 1st Fold : ", 'blue'))
thetas = np.array([theta1, theta2, theta3])
J_cvs = np.array([computeCostMultiT(X_norm1[0:m_cv-1, :], y[0:m_cv-1], theta1),
                  computeCostMultiT(X_norm2[0:m_cv-1, :], y[0:m_cv-1], theta2),
                  computeCostMultiT(X_norm3[0:m_cv-1, :], y[0:m_cv-1], theta3)])
print(colored(J_cvs, 'blue'))
least_error_pos = J_cvs.argmin()
print(colored(" Best theta is of hypo " + str(least_error_pos+1) + ", with theta:" +
              str(thetas[least_error_pos]), 'green'))
# Compute & Estimate J_test
if least_error_pos == 0:
    temp_norm = X_norm1
elif least_error_pos == 1:
    temp_norm = X_norm2
else:
    temp_norm = X_norm3
J_test = computeCostMultiT(temp_norm[m_cv+m_train:, :], y[m_cv+m_train:], thetas[least_error_pos])
print(colored("Estimate of J_test is :" + str(J_test), 'red'))
J_individual = (temp_norm[test_pointer:, :]).dot(thetas[least_error_pos])
J_individual = J_individual.T - y[test_pointer:]
J_individual = np.absolute(J_individual)
print("Average of J_test Error is : " + str(np.average(J_individual)))
print("with minimum Error : " + str(np.min(J_individual)))
print("Samples from difference between Reality and Prediction" + str(np.sort(J_individual)[0:7]))
for i in range(5):
    print(colored("----------------------------", 'yellow'))

# Third Fold
alpha = 0.01
lambda_ = 0.1
num_iterations = 1000

theta1 = np.zeros(np.shape(X_norm1)[1])
theta2 = np.zeros(np.shape(X_norm2)[1])
theta3 = np.zeros(np.shape(X_norm3)[1])
theta1, J_history1 = gradientDescentMulti(X_norm1[2*m_cv:m, :], y[2*m_cv:m],
                                          theta1, alpha, num_iterations, lambda_)
theta2, J_history2 = gradientDescentMulti(X_norm2[2*m_cv:m, :], y[2*m_cv:m],
                                          theta2, alpha, num_iterations, lambda_)
theta3, J_history3 = gradientDescentMulti(X_norm3[2*m_cv:m, :], y[2*m_cv:m],
                                          theta3, alpha, num_iterations, lambda_)
pyplot.plot(np.arange(len(J_history1)), J_history1, lw=2)
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Cost  J1')
pyplot.show()
print(colored('Theta from GD1 2nd Fold: {:s}'.format(str(theta1)), 'magenta'))
pyplot.plot(np.arange(len(J_history2)), J_history2, lw=2)
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Cost  J2')
pyplot.show()
print(colored('Theta from GD2 2nd Fold: {:s}'.format(str(theta2)), 'magenta'))
pyplot.plot(np.arange(len(J_history3)), J_history3, lw=2)
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Cost  J3')
pyplot.show()
print(colored('Theta from GD3 2nd Fold: {:s}'.format(str(theta3)), 'magenta'))
# Decide which hypo with min COST
print(colored("ERROR J_cv for 3 thetas 1st Fold : ", 'blue'))
thetas = np.array([theta1, theta2, theta3])
J_cvs = np.array([computeCostMultiT(X_norm1[m_cv:2*m_cv-1, :], y[m_cv:2*m_cv-1], theta1),
                  computeCostMultiT(X_norm2[m_cv:2*m_cv-1, :], y[m_cv:2*m_cv-1], theta2),
                  computeCostMultiT(X_norm3[m_cv:2*m_cv-1, :], y[m_cv:2*m_cv-1], theta3)])
print(colored(J_cvs, 'blue'))
least_error_pos = J_cvs.argmin()
print(colored(" Best theta is of hypo " + str(least_error_pos+1) + ", with theta:" +
              str(thetas[least_error_pos]), 'green'))
# Compute & Estimate J_test
if least_error_pos == 0:
    temp_norm = X_norm1
elif least_error_pos == 1:
    temp_norm = X_norm2
else:
    temp_norm = X_norm3
J_test = computeCostMultiT(temp_norm[0:m_cv-1, :], y[0:m_cv-1], thetas[least_error_pos])
print(colored("Estimate of J_test is :" + str(J_test), 'red'))
J_individual = (temp_norm[test_pointer:, :]).dot(thetas[least_error_pos])
J_individual = J_individual.T - y[test_pointer:]
J_individual = np.absolute(J_individual)
print("Average of J_test Error is : " + str(np.average(J_individual)))
print("with minimum Error : " + str(np.min(J_individual)))
print("Samples from difference between Reality and Prediction" + str(np.sort(J_individual)[0:7]))
for i in range(5):
    print(colored("----------------------------", 'yellow'))
