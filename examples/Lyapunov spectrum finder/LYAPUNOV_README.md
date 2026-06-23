# Finding the Lyapunov Spectrum in `reservoirpy`

## Additions to `observables.py`
Key functions and classes:
- `_lyapunov()`: General-purpose method for finding the first $k$ elements of the Lyapunov spectrum for a discrete-stepped dynamical system.
- `_ReservoirStepper`: Wrapper class to make the internal states of a `reservoirpy` model available to `_lyapunov()`. Can accommodate models with multiple state-bearing components (e.g. a reservoir computer with multiple reservoirs)
- `lyapunov()`: Main function for finding the Lyapunov spectrum of a reservoir computer, using `_lyapunov()` and `_ReservoirStepper` above.

A more detailed description of the method used is provided in the docstring for `_lyapunov()`.


## Demonstration in `lyapunov_demo.py`

`lyapunov_demo.py` tests both the ability of `observables._lyapunov()` to find the Lyapunov spectrum for a system of differential equations, and the ability of `observables.lyapunov()` to find the Lyapunov spectrum of a trained reservoir computer.

Lyapunov spectrum finding algorithms have a number of parameters that can significantly affect the outputs, particularly the length of the reorthonormalization cycle. If the cycle is too long, positive growth saturates and positive exponents returned are lower than their true values. If the cycle is too short, low-magnitude exponents can be obscured by numerical error and can take many cycles to converge. For a detailed look at the impact of cycle length on estimated exponents in one system, see [Edson et al 2019](https://arxiv.org/pdf/1902.09651), figure 3.

`observables._lyapunov()` attempts to make a good quality default choice for cycle length without user input based on the characteristic timescale of the system (the Lyapunov time, $1/\lambda_1$), which can be roughly estimated quickly. In general, this demo file aims to show the quality of the default parameters of `_lyapunov()`.
### Finding Lyapunov spectra of known systems
#### Systems Used

| Name                 | Type | Dimension                   | Parameters                                       |
|----------------------|------|-----------------------------|--------------------------------------------------|
| Lorenz 63            | ODE  | 3                           | $\sigma=10, \rho=28, \beta=8/3$                  |
| Lorenz 96            | ODE  | 10                          | $F=8$                                            |
| Mackey-Glass         | DDE  | 1                           | $\beta=0.2, \gamma=0.1, n=10, \theta=1, \tau=20$ |
| Kuramoto-Sivashinsky | PDE  | 1D Periodic Domain $[0, L)$ | $L=22$                                           |



`lyapunov_demo.py` finds the spectra for all of these systems using `observables._lyapunov()` with default parameters.

#### Numerica methods for known systems
`observables.lyapunov_()` requires a function that takes a system state and steps it forward some increment in time. Each of the known system class instances `Lorenz`, `Lorenz96`, `MackeyGlass`, `KuramotoSivashinsky` in `lyapunov_test.py` contains a method `step()` that does this, using a fixed-step 4th order Runge-Kutta method. For `MackeyGlass` this includes spline interpolation to find values between past steps, since it is a delay differential equation. For `KuramotoSivashinsky` this method is `ETDRK4`, Exponential Time Differencing, which evolves the system in Fourier space. For this work the KS Lyapunov spectra is calculated using values from the real space, though either real or fourier space calculation should give the same result.



#### Comparisons from sources in existing literature
In the existing literature, there are Lyapunov spectra for the Lorenz 63 and Kuramoto-Sivashinsky (KS) systems, with the exact parameters used in this demonstration. These spectra appear in the printout of `lyapunov_demo.py` to allow for comparison.
- Lorenz 63: [Sprott (2003)](https://sprott.physics.wisc.edu/chaos/lorenzle.htm): Apparently high-quality. Uses several orders of magnitude more iterations than are used in this demonstration. Can be assumed to be a ground truth.
- KS: [Edson et al (2019)](https://arxiv.org/abs/1902.09651): Contains no elements exactly equal to zero, which is a mark of insufficient convergence for continuous time systems. The true KS Lyapunov spectrum should contain at least two zero elements. Useful for comparison purposes, but should not be used as a ground truth.

### Finding Lyapunov spectra of Reservoir Computers
`lyapunov_demo.py` also builds and trains reservoir computers on data from the same systems listed above, and then finds the lyapunov spectra of those reservoir computers in closed-loop mode. For Lorenz 63, it trains several different variants:
- `lorenz_f`: ESN trained on all three variables ("full")
- `lorenz_x`: ESN trained on $x$ alone
- `lorenz_x_2_res`: 2-reservoir RC (averaging outputs of two separate ESNs), trained on $x$ alone, demonstrating the ability of `observables.lyapunov()` to handle reservoir computers with multiple elements with internal states.

For each of these systems, high quality hyperparameters have been selected, and are drawn from the file `hyperparameters.csv`. Systems trained on Lorenz 63 and Mackey-Glass systems are trained using ridge regression; systems trained on Lorenz 96 and Kuramoto-Sivashinsky systems are trained using noise regularization.

For each system, a table prints with the spectrum from literature (if available), the spectrum from the true ODE/DDE/PDE system found above, and the spectrum from trained reservoir computer.

### Discussion of Results
- Reservoir computers provided with all variables as input (`lorenz_f`, `lorenz96`) closely replicate the Lyapunov spectra of the system on which they were trained
- Reservoirs can replicate Lyapunov spectra of length greater than their input dimension. For example, the Mackey-Glass system ESN has a 1D input but accurately replicates the first 3 lyapunov exponents. ESNs `mackey_glass` and `lorenz_x` both tend to underestimate the magnitude of negative lyapunov exponents.
- In the Kuramoto-Sivashinsky (KS) system, the results from literature, the true PDE, and ESN all have different numbers of near-zero elements. Accurately handling near-zero elements is a known difficulty for lyapunov exponent finding, and in particular having reservoir computers replicate fewer zero elements has been observed (see [Pathak et al 2017](https://arxiv.org/pdf/1710.07313) figure 7 for an example). However, if these near-zero elements are removed from each spectrum, the agreement is quite good up to 7 exponents (see second table for KS).


