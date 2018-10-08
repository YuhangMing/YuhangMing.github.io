---
layout: post
title:  "Graph-Based SLAM"
date:   2018-10-07 16:17:00 -0200
description: details about the graph based SLAM and its probabilistic formulation.
permalink: /graph-based-SLAM/
---

### Probabilistic Formulation:

  Along a trajectory, given:
  
  -> A series of camera poses: $$ \boldsymbol{x_{1:T}} = \{ \boldsymbol{x_1}, \dots, \boldsymbol{x_T} \} $$;
  
  -> A series of measurements from motion sensors (like wheel odometry or IMU): $$ \boldsymbol{u_{1:T}} = \{ \boldsymbol{u_1}, \dots, \boldsymbol{u_T} \} $$;

  -> A series of perceptions of the environment (the landmarks observed by the camera at each time step): $$ \boldsymbol{z_{1:T}} = \{ \boldsymbol{z_1}, \dots, \boldsymbol{z_T} \} $$;

  -> A map of the environment $$ \boldsymbol{m} $$ which consists of $$N$$ landmarks $$ \boldsymbol{l_{1:N}} = \{ \boldsymbol{l_1}, \dots, \boldsymbol{l_N} \} $$;

  The SLAM problem can be formulated as: 

  *Estimating the posterior probability of the trajectory* $$\boldsymbol{x_{1:T}}$$ *and the map* $$\boldsymbol{m}$$ *given all the measurements plus an initial pose* $$\boldsymbol{x_0}$$.

  | $$ p(\boldsymbol{x_{1:T}}, \boldsymbol{m} \vert \boldsymbol{z_{1:T}}, \boldsymbol{u_{1:T}}, \boldsymbol{x_0}) $$ |


### Dynamic Bayesian Network (DBN)

  DBN describs a stochastic process as a directed graph where an arrow in the graph indicats the dependency between two nodes. E.g. an arrow pointed from $$\boldsymbol{x_0}$$ to $$\boldsymbol{x_1}$$ means $$p(\boldsymbol{x_1} \vert \boldsymbol{x_0})$$.

  ![Image](\assets\img\graph-based-slam\DBN.png)

  Expressing SLAM as a DBN highlights its temporal structure. Hence this formulization is well-suited for the filtering processes that can be used to tackle SLAM problems via the MAP (Maximize-A-Posterior) scheme. 

  | $$ \{ \boldsymbol{x}, \boldsymbol{l} \}^* = \mathrm{argmax}(\boldsymbol{x_0}) \prod P(\boldsymbol{x_k} \vert \boldsymbol{x_{k-1}, \boldsymbol{u_k}}) \prod P(\boldsymbol{z_k} \vert \boldsymbol{x_i}, \boldsymbol{l_j}) $$ |


### Graph-Based / Network-Based Formulation

  This formulation highlights the underlying spatial structure of the SLAM system. This is usually divided into a 2-step tast: 1. constructing the graph from raw measurements (front-end); 2. determining the MOST LIKELY configuration of the poses given constraints (edges) of the graph.

  - Data Association [Front-End]

    An edge between 2 nodes is labelled with a probatility distribution over teh relative transformation of 2 poses, conditioned on their mutual measurements. One needs to determine the Most Likely constraint resulting from an observation. More details about probablistic data association refer to [3].

    | $$ \mathcal{D}_{t, t+1} = \underset{\mathcal{D}}{\mathrm{argmax}} \, p(\mathcal{D} \vert \boldsymbol{x_0}, \boldsymbol{z_{t, t+1}}, \boldsymbol{u_{t, t+1}}) $$ |

  - Graph-Based Mapping [Back-End]

    *Assuming that Gaussian noise is added to the observations and the data association is known. The GOAL is to find a Gaussian approximation of the posterior over the trajectory.*

    First some new notations are introduced here:

    -> let $$ \boldsymbol{x} = (\boldsymbol{x_1}, \dots, \boldsymbol{x_T})^T $$ be the vector of parameters where $$ \boldsymbol{x_i} $$ stands for the pose of node $$ i $$; 

    -> let the noise be zero-mean Gaussian with information matrix $$ \Omega_{ij} $$, so the transformation that makes the observation acquired from $$i$$ maximally overlap with overvation acquired form $$j$$ follows the Gaussian distribution $$ \mathcal{N}(T_{ij}, \Omega_{ij}^{-1}) $$; 

    -> let $$ \hat{T_{ij}}(\boldsymbol{x_i}, \boldsymbol{x_j}) $$ be the prediction transformation between node $$i$$ and $$j$$ (note that this is a random variable).

    Therefore, we have the distribution of the random variable $$ \hat{T_{ij}} $$:

    $$ P_T( \hat{T_{ij}} ) = \frac{ \mathrm{exp} (-\frac{1}{2} (\hat{T_{ij}} - T_{ij})^T \Omega_{ij} (\hat{T_{ij}} - T_{ij})) }{\sqrt{ (2\pi)^k \vert\Sigma_{ij}\vert }} $$

    assuming that $$ \hat{T_{ij}} $$ is a $$k$$ dimensional vector. Then, the negative log-likelihood is

    $$ \mathcal{L}_{ij} \propto (T_{ij} - \hat{T_{ij}})^T \Omega_{ij} (T_{ij} - \hat{T_{ij}}) $$

    By defining $$ \boldsymbol{e_{ij}} = \boldsymbol{e_{ij}}(\boldsymbol{x_i}, \boldsymbol{x_j}) = T_{ij} - \hat{T_{ij}} $$, we have the final objective function:

    $$ \boldsymbol{\mathrm{F}}(\boldsymbol{x}) = \sum_{<i, j> \in C} \boldsymbol{e_{ij}}^T \Omega_{ij} \boldsymbol{e_{ij}} $$

    where $$C$$ is the set of pairs of indices for which a constraint (ovservation) exits. Under the MLE scheme, we can find the optimal values for $$ \boldsymbol{x} $$ by:

    | $$ \boldsymbol{x}^* = \underset{\boldsymbol{x}}{\mathrm{argmin}} \, \boldsymbol{\mathrm{F}}(\boldsymbol{x}) $$ |

  - Solving for The Optimal

    Given a good initial guess of the poses $$\breve{\boldsymbol{x}}$$, usually can be obtained using linear estimation like SVD, the numerical solution of $$ \underset{\boldsymbol{x}}{\mathrm{argmin}} \, \boldsymbol{\mathrm{F}}(\boldsymbol{x}) $$ can be found using following methods:

    1. First-order and Second-order Gradient Descent

       The most straight-forward way is performing Taylor expansion around the initial guess $$\breve{\boldsymbol{x}}$$, we have:

       $$ 
       \begin{split}
         & \boldsymbol{x}^* = \underset{\boldsymbol{x}}{\mathrm{argmin}} \, \boldsymbol{\mathrm{F}} (\boldsymbol{x}) \\
         \Rightarrow \quad
         & \breve{\boldsymbol{x}} + \Delta\boldsymbol{x}^* = \underset{\Delta\boldsymbol{x}}{\mathrm{argmin}} \, \boldsymbol{\mathrm{F}} (\breve{\boldsymbol{x}} + \Delta\boldsymbol{x})
       \end{split}
       $$

       $$ 
       \begin{split}
         \boldsymbol{\mathrm{F_{ij}}}(\breve{\boldsymbol{x_i}} + \Delta\boldsymbol{x_i}, \breve{\boldsymbol{x_j}} + \Delta\boldsymbol{x_j}) 
         &= \boldsymbol{\mathrm{F_{ij}}}(\breve{\boldsymbol{x}} + \Delta\boldsymbol{x}) \\
         &= \boldsymbol{e_{ij}}^T \Omega_{ij} \boldsymbol{e_{ij}} \\
         &\simeq \boldsymbol{\mathrm{F_{ij}}}(\breve{\boldsymbol{x}}) + \boldsymbol{J} \Delta\boldsymbol{x} + \frac{1}{2} \Delta\boldsymbol{x}^T\boldsymbol{H}\Delta\boldsymbol{x}
       \end{split}
       $$

       where $$\boldsymbol{J}$$ and $$\boldsymbol{H}$$ are the Jacobian matrix and Hessian matrix of $$\boldsymbol{\mathrm{F_{ij}}}$$. To find minimum, simply take derivative w.r.t. $$\Delta\boldsymbol{x}$$ and set the equation equal to 0.

       When keeping first-order gradient only, we have **Steepst Descent Method**, with

       $$ \Delta\boldsymbol{x}^* = -\boldsymbol{J}^T(\breve{\boldsymbol{x}}) $$

       When keeping the additional second-order gradient, we have **Newton Method**, with

       $$ \boldsymbol{H}\Delta\boldsymbol{x} = -\boldsymbol{J}^T $$

       *Problems with* 

       -> Steepest Descent Method: too greedy, leaning to zig-zag desent;

       -> Newton Method: huge computation complexity when calculating Hessian.

    2. Gauss-Newton Method

       To avoid the computation complexity when calculating Hessian, we tried to use first-order gradient (Jacobian) to approximate second-order gradient (Hessian). To begin with, we take Taylor expansion of the error term around the initial guess $$\breve{\boldsymbol{x}}$$:

       $$ \boldsymbol{e_{ij}}(\breve{\boldsymbol{x}} + \Delta\boldsymbol{x}) \simeq \boldsymbol{e_{ij}}(\breve{\boldsymbol{x}}) + \boldsymbol{J_{ij}}\Delta\boldsymbol{x} $$

       Then substituting the error term back into the objective funciton, we have the new objective function:
       
       $$ 
       \begin{split}
        & \Delta\boldsymbol{x}^* = \underset{\Delta\boldsymbol{x}}{\mathrm{argmin}} \, \boldsymbol{\mathrm{F}} (\breve{\boldsymbol{x}} + \Delta\boldsymbol{x}) \\
        \Rightarrow \quad
        & \Delta\boldsymbol{x}^* = \underset{\Delta\boldsymbol{x}}{\mathrm{argmin}} \, \sum_{<i, j> \in C} (\boldsymbol{e_{ij}}(\breve{\boldsymbol{x}}) + \boldsymbol{J_{ij}}\Delta\boldsymbol{x})^T \Omega_{ij} (\boldsymbol{e_{ij}}(\breve{\boldsymbol{x}}) + \boldsymbol{J_{ij}}\Delta\boldsymbol{x})
       \end{split}
       $$

       after expanding and combining terms, we have:

       $$
       \begin{split}
         \boldsymbol{\mathrm{F_{ij}}}(\breve{\boldsymbol{x}} + \Delta\boldsymbol{x}) 
         &= \boldsymbol{e_{ij}}(\breve{\boldsymbol{x}} + \Delta\boldsymbol{x})^T \Omega_{ij} \boldsymbol{e_{ij}}(\breve{\boldsymbol{x}} + \Delta\boldsymbol{x}) \\
         &\simeq (\boldsymbol{e_{ij}}(\breve{\boldsymbol{x}}) + \boldsymbol{J_{ij}}\Delta\boldsymbol{x})^T \Omega_{ij} (\boldsymbol{e_{ij}}(\breve{\boldsymbol{x}}) + \boldsymbol{J_{ij}}\Delta\boldsymbol{x}) \\
         &= \boldsymbol{e_{ij}}(\breve{\boldsymbol{x}})^T\Omega_{ij}\boldsymbol{e_{ij}}(\breve{\boldsymbol{x}}) + 2\boldsymbol{e_{ij}}(\breve{\boldsymbol{x}})^T\Omega_{ij}\boldsymbol{J_{ij}}\Delta\boldsymbol{x} + \Delta\boldsymbol{x}^T\boldsymbol{J_{ij}}^T\Omega_{ij}\boldsymbol{J_{ij}}\Delta\boldsymbol{x}
       \end{split}
       $$

       Again, to find the optimal, we take derivative w.r.t. $$\Delta\boldsymbol{x}$$ and set the equation equal to 0, we have:

       $$ 
       \begin{split}
         & \boldsymbol{e_{ij}}(\breve{\boldsymbol{x}})^T\Omega_{ij}\boldsymbol{J_{ij}} + \boldsymbol{J_{ij}}^T\Omega_{ij}\boldsymbol{J_{ij}}\Delta\boldsymbol{x} = \boldsymbol{0} \\
         \Rightarrow \quad
         & \boldsymbol{J_{ij}}^T\Omega_{ij}\boldsymbol{J_{ij}}\Delta\boldsymbol{x} = -\boldsymbol{e_{ij}}(\breve{\boldsymbol{x}})^T\Omega_{ij}\boldsymbol{J_{ij}}
       \end{split}
       $$

       Let $$\,\boldsymbol{H} = \boldsymbol{J_{ij}}^T\Omega_{ij}\boldsymbol{J_{ij}}$$, $$\,\boldsymbol{g} = -\boldsymbol{e_{ij}}(\breve{\boldsymbol{x}})^T\Omega_{ij}\boldsymbol{J_{ij}}$$, 

       | $$\boldsymbol{H}\Delta\boldsymbol{x} = \boldsymbol{g}$$ |

       which is called **Augmented Equation**, a.k.a. Gauss Newton Equation or Normal Equation.

       **Solving the augmented equation is the core of the optimization.** The complete optimization algorithm is:

       *i. Calculate the initial guess* $$\,\breve{\boldsymbol{x}}$$;

       *ii. For kth iteration, calculate Jacobian matrix and the error term*;

       *iii. Sovle the augmented equation*;

       *iv. If* $,\Delta\boldsymbol{x_k}\,$$ *is small enough, stop; else,* $$\,\Delta\boldsymbol{x_{k+1}} = \boldsymbol{x_k} + \Delta\boldsymbol{x_k}$$.

       *Problems with* Gauss-Newton: $$\boldsymbol{H}$$ should be positive definite while $$\boldsymbol{J}^T\Omega\boldsymbol{J}$$ is positive semi-definite (may be a singular matrix or in ill-condition); the algorithm may not converge due to the unstable augmented value.

    3. Levenberg-Marquardt Method
       
       To get a better approximation of the Hessian matrix, a Trust Region $$\mu$$ is added to the $$\Delta\boldsymbol{x}$$. The new objective function is defined as:

       $$ \Delta\boldsymbol{x}^* = \underset{\Delta\boldsymbol{x}}{\mathrm{argmin}} \, \sum (\boldsymbol{e_{ij}}(\breve{\boldsymbol{x}}) + \boldsymbol{J_{ij}}\Delta\boldsymbol{x})^T \Omega_{ij} (\boldsymbol{e_{ij}}(\breve{\boldsymbol{x}}) + \boldsymbol{J_{ij}}\Delta\boldsymbol{x}), \,\, s.t. \,\, \parallel D\Delta\boldsymbol{x} \parallel^2 \leq \mu $$

       where $$D$$ can either be identity or tha square root of the diagonal elements from $$\boldsymbol{J}^T\boldsymbol{J}$$. We define:

       $$ \rho = \frac{\boldsymbol{e_{ij}}(\breve{\boldsymbol{x}} + \Delta\boldsymbol{x}) - \boldsymbol{e_{ij}}(\breve{\boldsymbol{x}})}{\boldsymbol{J}(\breve{\boldsymbol{x}}) \Delta\boldsymbol{x}} $$

       -> $$ \rho \, \rightarrow \, 1$$, meaning the approximation is good;

       -> $$\rho$$ is too small, meaning the real descent is far smaller than the approximated descent, the range needs to be narrowed down;

       -> $$\rho$$ is too large, meaning the real descent is far larger than the approximated descent, the range needs to be expanded.



    Gauss-Newton or Levernberg-Marquardt algorithms. The main idea behind these algorithms is to approximate the error function by its first order Taylor expansion around the current initial guess $$\breve{\boldsymbol{x}}$$.


### Reference:

[1] 高翔；张涛；刘毅；严沁睿: "视觉SLAM十四讲，从理论到实践".

[2] G. Grisetti, R. Kümmerle, C. Stachniss, and W. Burgard. "A Tutorial on Graph-Based SLAM". IEEE Intelligent Transportation Systems Magazine. 2(4):31-43, December 2010.

[3] S. Bowman, N. Atanasov, K. Daniilidis, and G. Pappas. "Probabilistic Data Association for Semantic SLAM". IEEE International Conference on Robotics and Automation (ICRA). May 2017.


### Appendix
  



