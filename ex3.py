import ROOT as r
import numpy as np
from iminuit import Minuit
from iminuit.cost import LeastSquares

# Extract histograms from "fitInputs.root":
tf = r.TFile("fitInputs.root")
raw_dat = tf.Get("hdata")
hbkg = tf.Get("hbkg")

# Convert 2D histograms to np arrays:
def hist2np_2d(h):
    nbinx = h.GetNbinsX()
    nbiny = h.GetNbinsY()
    
    x = np.zeros(nbinx)
    y = np.zeros(nbiny)
    z = np.zeros((nbinx, nbiny))
    ez = np.zeros((nbinx, nbiny))
    
    # Get bin centers for x and y axes
    for i in range(1, nbinx + 1):
        x[i-1] = h.GetXaxis().GetBinCenter(i)
    
    for j in range(1, nbiny + 1):
        y[j-1] = h.GetYaxis().GetBinCenter(j)
    
    # Get bin contents and errors
    for i in range(1, nbinx + 1):
        for j in range(1, nbiny + 1):
            z[i-1, j-1] = h.GetBinContent(i, j)
            ez[i-1, j-1] = h.GetBinError(i, j)
    
    return x, y, z, ez

# Convert histograms to np.arrays:
x_bkgrd, y_bkgrd, z_bkgrd, ez_bkgrd = hist2np_2d(hbkg)
x_data, y_data, z_data, ez_data = hist2np_2d(raw_dat)

# Define the signal form
def detection(x, y, A, mu1, mu2, sig1, sig2, norm):
    X, Y = np.meshgrid(x, y, indexing='ij')
    full = A * np.exp(- (X - mu1)**2 / sig1**2) * np.exp(- (Y - mu2)**2 / sig2**2) + norm*z_bkgrd
    return full

def signal(x, y, A, mu1, mu2, sig1, sig2):
    X, Y = np.meshgrid(x, y, indexing='ij')
    full = A * np.exp(- (X - mu1)**2 / sig1**2) * np.exp(- (Y - mu2)**2 / sig2**2)
    return full

# Define chi-squared function for fitting
def chi_squared(A, mu1, mu2, sig1, sig2, norm):
    """
    Calculate chi-squared between data and model (signal + normalized background)
    """
    model = detection(x_data, y_data, A, mu1, mu2, sig1, sig2, norm)
    
    # Calculate chi-squared with proper error weighting
    # Avoid division by zero by adding small epsilon where errors are zero
    errors = np.where(ez_data > 0, ez_data, 1.0)
    chi2 = np.sum(((z_data - model) / errors)**2)
    
    return chi2

# Estimate initial parameters from the data
print("Estimating initial parameters from data...")
print("=" * 60)

# Find approximate signal location (max of data - background)
signal_estimate = z_data - z_bkgrd
max_idx = np.unravel_index(np.argmax(signal_estimate), signal_estimate.shape)
mu1_init = x_data[max_idx[0]]
mu2_init = y_data[max_idx[1]]
A_init = np.max(signal_estimate)

# Estimate widths from data spread
x_weighted = np.sum(x_data[:, np.newaxis] * signal_estimate) / np.sum(signal_estimate)
y_weighted = np.sum(y_data[np.newaxis, :] * signal_estimate.T) / np.sum(signal_estimate)
sig1_init = np.sqrt(np.sum((x_data[:, np.newaxis] - x_weighted)**2 * signal_estimate) / np.sum(signal_estimate))
sig2_init = np.sqrt(np.sum((y_data[np.newaxis, :] - y_weighted)**2 * signal_estimate.T) / np.sum(signal_estimate))

print(f"Initial estimates:")
print(f"  A ~ {A_init:.2f}")
print(f"  mu1 ~ {mu1_init:.2f}")
print(f"  mu2 ~ {mu2_init:.2f}")
print(f"  sig1 ~ {sig1_init:.2f}")
print(f"  sig2 ~ {sig2_init:.2f}")
print()

# Set up MINUIT fitter with data-driven initial guesses
m = Minuit(chi_squared, 
           A=A_init,         # Signal amplitude from data
           mu1=mu1_init,     # Mean in x from data
           mu2=mu2_init,     # Mean in y from data
           sig1=max(sig1_init, 0.5),  # Width in x from data
           sig2=max(sig2_init, 0.5),  # Width in y from data
           norm=1.0)         # Background normalization

# Set parameter limits if needed (optional but often helpful)
m.limits['A'] = (0, None)        # Amplitude must be positive
m.limits['sig1'] = (0.1, 10.0)   # Width must be positive with upper bound
m.limits['sig2'] = (0.1, 10.0)   # Width must be positive with upper bound
m.limits['norm'] = (0.5, 2.0)    # Background normalization constrained
m.limits['mu1'] = (x_data.min(), x_data.max())  # Means within data range
m.limits['mu2'] = (y_data.min(), y_data.max())

# Set error definition for chi-squared (1 sigma = delta chi2 = 1)
m.errordef = Minuit.LEAST_SQUARES

# Set tolerances for more robust convergence
m.tol = 0.1  # Increase tolerance (default is 0.1)

# Perform the fit with multiple strategies
print("Starting MINUIT fit...")
print("=" * 60)

# Strategy 1: Run migrad
m.migrad()

# If it didn't converge, try running migrad again
if not m.valid:
    print("\nFirst attempt didn't fully converge. Trying again...")
    m.migrad()

# If still not converged, try simplex first, then migrad
if not m.valid:
    print("\nTrying simplex algorithm first...")
    m.simplex()
    m.migrad()

# Final check with hesse to get better error estimates
if m.valid:
    print("\nCalculating improved error estimates with Hesse...")
    m.hesse()

# Print results
print("\nFit Results:")
print("=" * 60)
print(m.params)  # Print parameter values and errors

print("\nFit Quality:")
print("=" * 60)
print(f"Minimum chi-squared: {m.fval:.2f}")
print(f"Number of degrees of freedom: {len(z_data.flatten()) - m.nfit}")
print(f"Reduced chi-squared: {m.fval / (len(z_data.flatten()) - m.nfit):.2f}")
print(f"Fit converged: {m.valid}")

# Get the best-fit parameters
best_params = m.values
print("\nExtracted Signal Parameters:")
print("=" * 60)
print(f"Amplitude (A):     {best_params['A']:.3f} ± {m.errors['A']:.3f}")
print(f"Mean X (mu1):      {best_params['mu1']:.3f} ± {m.errors['mu1']:.3f}")
print(f"Mean Y (mu2):      {best_params['mu2']:.3f} ± {m.errors['mu2']:.3f}")
print(f"Width X (sig1):    {best_params['sig1']:.3f} ± {m.errors['sig1']:.3f}")
print(f"Width Y (sig2):    {best_params['sig2']:.3f} ± {m.errors['sig2']:.3f}")
print(f"BG norm (norm):    {best_params['norm']:.3f} ± {m.errors['norm']:.3f}")


# Now plot the results

best_A = best_params['A']
best_mu1 = best_params['mu1']
best_mu2 = best_params['mu2']
best_sig1 = best_params['sig1']
best_sig2 = best_params['sig2']
best_norm = best_params['norm']


# Function to convert numpy array to ROOT TH2D
def np2hist_2d(x, y, z, name, title):

    nbinx = len(x)
    nbiny = len(y)
    
    # Calculate bin edges from bin centers
    x_edges = np.zeros(nbinx + 1)
    y_edges = np.zeros(nbiny + 1)
    
    # Calculate edges assuming uniform binning
    dx = x[1] - x[0] if nbinx > 1 else 1.0
    dy = y[1] - y[0] if nbiny > 1 else 1.0
    
    x_edges[0] = x[0] - dx/2
    for i in range(nbinx):
        x_edges[i+1] = x[i] + dx/2
    
    y_edges[0] = y[0] - dy/2
    for j in range(nbiny):
        y_edges[j+1] = y[j] + dy/2
    
    # Create histogram
    hist = r.TH2D(name, title, nbinx, x_edges, nbiny, y_edges)
    
    # Fill histogram
    for i in range(nbinx):
        for j in range(nbiny):
            hist.SetBinContent(i+1, j+1, z[i, j])
    
    return hist

# Calculate the fitted model
z_fit = detection(x_data, y_data, best_A, best_mu1, best_mu2, 
                  best_sig1, best_sig2, best_norm)

# Calculate residuals
z_residuals = z_data - z_fit

# Calculate data with background subtracted
z_bkg_subtracted = signal(x_data, y_data, best_A, best_mu1, best_mu2, 
                  best_sig1, best_sig2)


# Slightly out of order from how the problem is worded:
# Need to estimate the total number of signal events
# Total signal = sum over all bins
N_signal = np.sum(z_bkg_subtracted)

print(f"Total signal events: {N_signal:.2f}")

# Now calculate the uncertainty using error propagation
# We need the gradient of N_signal with respect to each parameter

# Small step size for numerical derivatives
epsilon = 1e-6

# Calculate partial derivatives numerically
gradients = {}
params = ['A', 'mu1', 'mu2', 'sig1', 'sig2']

for param in params:
    # Get current parameter value
    p_val = m.values[param]
    
    # Calculate N_signal with parameter shifted up
    m_values_up = {p: m.values[p] for p in params}
    m_values_up[param] = p_val + epsilon
    z_up = signal(x_data, y_data, m_values_up['A'], m_values_up['mu1'], 
                    m_values_up['mu2'], m_values_up['sig1'], m_values_up['sig2'])
    N_up = np.sum(z_up)
    
    # Calculate N_signal with parameter shifted down
    m_values_down = {p: m.values[p] for p in params}
    m_values_down[param] = p_val - epsilon
    z_down = signal(x_data, y_data, m_values_down['A'], m_values_down['mu1'], 
                    m_values_down['mu2'], m_values_down['sig1'], m_values_down['sig2'])
    N_down = np.sum(z_down)
    
    # Numerical derivative
    gradients[param] = (N_up - N_down) / (2 * epsilon)

# Error propagation: σ²(N) = Σᵢⱼ (∂N/∂pᵢ) Cᵢⱼ (∂N/∂pⱼ)
# where Cᵢⱼ is the covariance matrix

variance = 0.0
for i, param_i in enumerate(params):
    for j, param_j in enumerate(params):
        # Get covariance between parameters i and j
        cov_ij = m.covariance[param_i, param_j]
        variance += gradients[param_i] * cov_ij * gradients[param_j]

N_signal_error = np.sqrt(variance)

print(f"Uncertainty: ± {N_signal_error:.2f}")
print(f"\nResult: N_signal = {N_signal:.2f} ± {N_signal_error:.2f}")
print(f"Relative uncertainty: {100*N_signal_error/N_signal:.2f}%")

# Show breakdown of contributions
print("\nUncertainty contributions from each parameter:")
for param in params:
    # Variance contribution from this parameter alone
    var_contribution = gradients[param]**2 * m.covariance[param, param]
    sigma_contribution = np.sqrt(var_contribution)
    print(f"  {param:5s}: ± {sigma_contribution:.2f} ({100*sigma_contribution/N_signal_error:.1f}% of total)")



# Create ROOT histograms
h_data = np2hist_2d(x_data, y_data, z_data, "h_data", 
                     "Data;X;Y;Counts")
h_fit = np2hist_2d(x_data, y_data, z_fit, "h_fit", 
                    "Best Fit (Signal + Background);X;Y;Counts")
h_residuals = np2hist_2d(x_data, y_data, z_residuals, "h_residuals", 
                          "Residuals (Data - Fit);X;Y;Residual")
h_signal = np2hist_2d(x_data, y_data, z_bkg_subtracted, "h_signal",
                      "Fitted Signal Only;X;Y;Counts")

# Create canvas divided into 2x2
canvas = r.TCanvas("c1", "Fit Results", 1400, 1200)
canvas.Divide(2, 2)

# Set ROOT to batch mode if you don't want windows popping up
# r.gROOT.SetBatch(True)

# Plot 1: Data
canvas.cd(1)
r.gPad.SetTheta(30)  # Set viewing angle
r.gPad.SetPhi(30)
h_data.SetLineColor(r.kBlue)
h_data.Draw("LEGO2")

# Plot 2: Fitted result
canvas.cd(2)
r.gPad.SetTheta(30)
r.gPad.SetPhi(30)
h_fit.SetLineColor(r.kRed)
h_fit.Draw("LEGO2")

# Plot 3: Residuals
canvas.cd(3)
r.gPad.SetTheta(30)
r.gPad.SetPhi(30)
h_residuals.SetLineColor(r.kGreen+2)
h_residuals.Draw("LEGO2")

# Plot 4: Background subtracted
canvas.cd(4)
r.gPad.SetTheta(30)
r.gPad.SetPhi(30)
h_signal.SetLineColor(r.kMagenta)
h_signal.Draw("LEGO2")

# Update canvas
canvas.Update()

# Save the canvas
canvas.SaveAs("ex3.png")
canvas.SaveAs("ex3.pdf")

print("2x2 plot saved as ex3.png and ex3.pdf")

# Keep the canvas open (comment out if running in batch mode)
input("Press Enter to close...")