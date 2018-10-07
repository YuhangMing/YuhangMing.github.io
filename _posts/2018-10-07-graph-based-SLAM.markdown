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

  DBN describs a stochastic process as a directed graph where an arrow in the graph indicats the dependency between two nodes. E.g. arrow pointed from $$\boldsymbol{x_0}$$ to $$\boldsymbol{x_1}$$ means $$p(\boldsymbol{x_1} \vert \boldsymbol{x_0})$$.

  ![Image](\assets\img\graph-based-slam\DBN.png)

  Expressing SLAM as a DBN highlights its temporal structure. Hence this formulization is well-suited for the filtering processes that can be used to tackle SLAM problems via the MAP (Maximize-A-Posterior) scheme. 

  | $$ \{ \boldsymbol{x}, \boldsymbol{l} \}^* = \mathrm{argmax}(\boldsymbol{x_0}) \prod P(\boldsymbol{x_k} \vert \boldsymbol{x_{k-1}, \boldsymbol{u_k}}) \prod P(\boldsymbol{z_k} \vert \boldsymbol{x_i}, \boldsymbol{l_j}) $$ |




### Reference:

[1] 高翔；张涛；刘毅；严沁睿: "视觉SLAM十四讲，从理论到实践".

[2] G. Grisetti, R. Kümmerle, C. Stachniss, and W. Burgard. "A Tutorial on Graph-Based SLAM". IEEE Intelligent Transportation Systems Magazine. 2(4):31-43, December 2010.


### Appendix
  



