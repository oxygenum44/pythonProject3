import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import curve_fit
import matplotlib as mpl
import scipy.stats as stats

cmap = mpl.colormaps['copper']
colors = cm.get_cmap('copper', 5)

# Read data from the tab-separated text file
file_path = 'DAT/90Results_combitation_failure_a_inititationR1.txt'  # Replace with the path to your text file
with open(file_path, 'r') as file:
    lines = file.readlines()

# Extract x and y values from the file
x = []
y = []
for line in lines:
    values = line.split('\t')  # Use '\t' as the delimiter for tab-separated data
    if len(values) >= 2:  # Check if the line has at least two values
        x.append(float(values[0]))
        y.append(float(values[1]))

# Create a scatter plot
plt.scatter(x, y, color=colors(1), marker='o', label='R=-1')

# Read data from the tab-separated text file
file_path = 'DAT/90Results_combitation_failure_a_inititationR005.txt'  # Replace with the path to your text file
with open(file_path, 'r') as file:
    lines = file.readlines()

# Extract x and y values from the file
x1 = []
y1 = []
for line in lines:
    values = line.split('\t')  # Use '\t' as the delimiter for tab-separated data
    if len(values) >= 2:  # Check if the line has at least two values
        x1.append(float(values[0]))
        y1.append(float(values[1]))

# Create a scatter plot
plt.scatter(x1, y1, color=colors(3), marker='h', label='R=0.05')

# Read data from the tab-separated text file
file_path = 'DAT/90Results_combitation_failure_a_inititationR05.txt'  # Replace with the path to your text file
with open(file_path, 'r') as file:
    lines = file.readlines()

# Extract x and y values from the file
x2 = []
y2 = []
for line in lines:
    values = line.split('\t')  # Use '\t' as the delimiter for tab-separated data
    if len(values) >= 2:  # Check if the line has at least two values
        x2.append(float(values[0]))
        y2.append(float(values[1]))

# whole data at once
x_total = x + x1 + x2
y_total = y + y1 + y2
# Perform linear regression
coefficients = np.polyfit(x_total, y_total, 1)
poly = np.poly1d(coefficients)
print(f"basic regression {coefficients}")

# Create a scatter plot
plt.scatter(x2, y2, color=colors(5), marker='v', label='R=0.5')

# Add linear regression line
# plt.plot(x_total, poly(x_total), color=colors(1), label='Linear Regression')

# Add labels and title
plt.xlabel('$N_{f}$ [cycles]')
plt.ylabel('$N_{CDS}$ [cycles]')
# plt.title('Scatter Plot from Tab-Separated Text File')


# Add a legend
plt.legend()


# Show the plot
# plt.show()


# Define the logarithmic function for regression
def linear_function(x, a, b):
    return a * np.multiply(1, x) + b


# Fit the data to the linear model
params, covariance = curve_fit(linear_function, x_total, y_total)
print(f" non basic regression params curve fit {params}")
# Calculate the best-fit curve
xfit = np.linspace(0, 30000, 100)  # Logarithmic scale for x-axis
yfit = linear_function(xfit, *params)

# fit the model
fit = np.polyfit(x_total, y_total, 1)

print(f" non basic regression {fit}")

# view the output of the model
print(fit)

# Calculate the prediction intervals for the best-fit curve
alpha = 0.05  # Significance level (95% confidence)
n = len(x_total)
p = len(params)
dof = n - p
t_value = stats.t.ppf(1 - alpha / 2, dof)  # T-distribution value for a 95% confidence level with n-p degrees of freedom

# Calculate the standard error of the estimate
residuals = y_total - linear_function(x_total, *params)
std = np.sum(residuals ** 2) / dof
std_error = np.sqrt(std)

# Calculate prediction intervals for the curve
prediction_intervals = t_value * std_error

# Calculate upper and lower bounds for the prediction intervals
upper_bounds = yfit + prediction_intervals
lower_bounds = yfit - prediction_intervals

# Plot the data and the best-fit curve
# plt.scatter(x_total, y_total, label='Data', color='black')
plt.plot(xfit, yfit, '-', label='Linear Regression - all data', color=colors(3))
mpl.rcParams["text.usetex"]
# Plot the boundary lines of the prediction intervals
plt.plot(xfit, upper_bounds, '--', color=colors(3))
plt.plot(xfit, lower_bounds, '--', color=colors(3))
# plt.plot(xfit, upper_bounds, 'r--', label='95% Prediction Interval')
# plt.plot(xfit, lower_bounds, 'r--', label='5% Prediction Interval')
plt.xscale('linear')  # Use a logarithmic scale for the x-axis
plt.yscale('linear')  # Use a linear scale for the y-axis
plt.xlabel('Number of Cycles to failure [-]')
plt.ylabel('Stress amplitude [MPa]')
plt.xlim(left=min(x) * 0.8, right=max(x) * 1.5)
plt.ylim(bottom=min(y) * 0.8, top=max(y) * 1.2)
plt.fill_between(xfit, lower_bounds, upper_bounds, color=colors(5), alpha=0.2, label='95% confidence interval')
plt.legend()
# Show the plot
plt.show()
