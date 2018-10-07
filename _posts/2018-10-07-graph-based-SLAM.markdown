---
layout: post
title:  "Graph Based SLAM"
date:   2018-10-07 16:17:00 -0200
description: details about the graph based SLAM and its probabilistic formulation.
permalink: /graph-based-SLAM/
---

### Probabilistic Formulation:

  Along a trajectory, given:
  
  -> A series of camera poses: $$ \boldsymbol{x_{1:T}} = \{ \boldsymbol{x_1}, \dots, \boldsymbol{x_T} \} $$;
  
  -> A series of measurements from motion sensors (like wheel odometry or IMU): $$ \boldsymbol{u_{1:T}} = \{ \boldsymbol{u_1}, \dots, \boldsymbol{u_T} \} $$;

  -> A series of perceptions of the environment (the landmarks observed by the camera at each time step): $$ \boldsymbol{z_{1:T}} = \{ \boldsymbol{z_1}, \dots, \boldsymbol{z_T} \} $$;

  -> A map of the environment $$ \boldsymbol{m} $$ which consists of N landmarks $$ \boldsymbol{y_{1:N}} = \{ \boldsymbol{y_1}, \dots, \boldsymbol{y_N} \} $$;

  The SLAM problem can be formulated as: 

  *Estimating the posterior probability of the trajectory* $$\boldsymbol{x_{1:T}}$$ *and the map* $$\boldsymbol{m}$$ *given all the measurements plus an initial pose* $$\boldsymbol{x_0}$$.

  $$ p(\boldsymbol{x_{1:T}}, \boldsymbol{m} \vert \boldsymbol{z_{1:T}}, \boldsymbol{u_{1:T}}, \boldsymbol{x_0}) $$


### Reference:

[1] 高翔；张涛；刘毅；严沁睿: "视觉SLAM十四讲，从理论到实践".

[2] G. Grisetti, R. Kümmerle, C. Stachniss, and W. Burgard. "A Tutorial on Graph-Based SLAM". IEEE Intelligent Transportation Systems Magazine. 2(4):31-43, December 2010.


### Appendix
  



