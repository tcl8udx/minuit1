# minuit1

- expFit.cpp: example using the more general fitting interface in minuit
- expFit.ipynb: equivalent example using lmfit
- SimultaneousExps(lm).ipynb: generation of histograms with correlated signals for simultaneous fit exercise
- rootExample.cpp: just another example of using ROOT classes in a C++ program
- *.root files: various input histograms for fitting exercises

-----

Team member names and computing IDs: Ali Ahmad (mdk3qf), Tristen Lowrey (tcl8udx)

-----

Exercise 1 comments:
--
We can see from our plots in ex1.pdf that the two Gaussians fit, despite appearing to be good, is actually horrible. It has a chi-squared of 258.05 and a p-value of 0.0000. In contrast, the Gumbel distribution returns a chi-squared of 21.51 and a p-value of 0.9733. This suggests that the data itself is closely related to a Gumbel distribution. The NLL analysis shows similar results.

-----

Exercise 2 comments:
--
Looking at our plots in SimultaenousExps.ipynb, we see the signal has a mean of 76.60 +- 0.45, a sigma of 4.60 +- 0.44, and a p-value of 0.883. The consistent results for both experiment 1 and 2, in addition to the high p-value indicates that this is a good fit.

-----

Exercise 3 comments:
--
Extracted Signal Parameters:
Amplitude (A):     52.979 ± 0.634
Mean X (mu1):      3.477 ± 0.006
Mean Y (mu2):      2.340 ± 0.013
Width X (sig1):    0.780 ± 0.006
Width Y (sig2):    1.410 ± 0.014
BG norm (norm):    0.500 ± 0.000
Total signal events: 18125.08
Uncertainty: ± 189.01

Result: N_signal = 18125.08 ± 189.01
Relative uncertainty: 1.04%

Uncertainty contributions from each parameter:
  A    : ± 216.93 (114.8% of total)
  mu1  : ± 0.00 (0.0% of total)
  mu2  : ± 5.92 (3.1% of total)
  sig1 : ± 147.69 (78.1% of total)
  sig2 : ± 171.12 (90.5% of total)


The number of signal events was calculated by summing over all of the bins in the signal histogram.

The error was a little more complicated. In short, we calculated it from error propagation through the error associated with each of the fitting parameters.
-----
