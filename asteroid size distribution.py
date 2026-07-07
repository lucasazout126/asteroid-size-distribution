# pip install requests numpy matplotlib scipy

import requests
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# Download asteroid data from Asterank (a NASA-sourced asteroid database)
# We only grab asteroids larger than 1 km because surveys are incomplete below that —
# missing small asteroids would artificially flatten our slope toward 0
data = requests.get("http://www.asterank.com/api/asterank", params={
    "query": '{"diameter":{"$gt":1}}',  # only asteroids with diameter > 1 km
    "limit": 4000,                      # maximum number to fetch
    "sort": '{"diameter":1}',           # smallest first
    "fields": '["full_name","diameter"]'
}).json()

# Pull out just the diameter values as a plain number list
diameters = [float(a["diameter"]) for a in data if a.get("diameter") and float(a["diameter"]) > 0]
diam_array = np.array(diameters)
print(f"Loaded {len(diam_array)} asteroids")

# A straight line: y = m*x + b
def line(x, m, b):
    return m * x + b

# A power law curve: y = k * x^alpha
# This is what we expect asteroid distributions to follow
def power_law(x, k, alpha):
    return k * x ** alpha

# ── Chart 1: bar chart + best fit curve ───────────────────────────────────────

# Plot a histogram — this counts how many asteroids fall into each diameter bin
counts, bins, _ = plt.hist(diam_array, label='Asteroid Data')

# Find the center of each bin (we need this to fit a curve through the bars)
bin_centers = (bins[:-1] + bins[1:]) / 2

# Take log10 of both axes, skipping empty bins (log of 0 is undefined)
log_diam   = np.log10(bin_centers[counts != 0])
log_counts = np.log10(counts[counts != 0])

# Fit a straight line through the log-log data to find the power law exponent
# curve_fit returns the best [slope, intercept] for our line function
popt, pcov = optimize.curve_fit(line, log_diam, log_counts)
slope, intercept = popt
slope_err = np.sqrt(np.diag(pcov))[0]

# Convert the line back into a power law curve and draw it over the histogram
diam_vals = np.linspace(np.min(bin_centers), np.max(bin_centers))
plt.plot(diam_vals, power_law(diam_vals, 10**intercept, slope), label='Best Fit')
plt.xlabel('Diameter [km]')
plt.ylabel('Number of Asteroids')
plt.title('Diameter Distribution of Asteroids')
plt.legend()
plt.show()

# ── Chart 2: log-log view ─────────────────────────────────────────────────────

# Plotting log(diameter) vs log(count) turns the curved power law into a straight line
# The slope of that line is the power law exponent — we expect it to be close to -2.5
log_diam_vals = np.linspace(np.min(log_diam), np.max(log_diam), num=10)
plt.plot(log_diam, log_counts, "o", label='Asteroid Data')
plt.plot(log_diam_vals, line(log_diam_vals, slope, intercept), label='Best Fit')
plt.xlabel('log$_{10}$ (Diameter [km])')
plt.ylabel('log$_{10}$ (Number of Asteroids)')
plt.title('Diameter Distribution of Asteroids (Log-Log)')
plt.legend()
plt.show()

print(f"Best-fit slope: {slope:.3f} ± {slope_err:.3f}")

# python "c:\Users\lucas\OneDrive\Desktop\Data Science Projects\Python\Asteroid Mass Distribution.py"
