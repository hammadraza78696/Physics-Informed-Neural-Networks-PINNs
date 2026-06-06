# Physics-Informed Neural Networks (PINNs) for 1D Heat Equation

## Overview
This repository implements a Physics-Informed Neural Network (PINN) to solve a 1D partial differential equation (PDE) governing thermodynamic heat transfer without relying on labeled mesh datasets.

## Mathematical Formulation
The network models the functional approximation $u(x, t)$ parameterized by weights $\theta$. The loss objective embeds the residual of the Heat Equation directly via automatic differentiation:

$$f = \frac{\partial u}{\partial t} - \alpha \frac{\partial^2 u}{\partial x^2} = 0$$

## Technical Architecture
- **Framework:** PyTorch (Autograd Engine)
- **Optimization Target:** Mean Squared Error ($MSE_f$) of PDE collocation residuals.
- **Hidden Layers:** Multi-Layer Perceptron (MLP) mapping spatial coordinates and temporal constraints.
