---
layout: post
title:  "REVIEW: Semantic SLAM (Keep Updating)"
date:   2018-10-09 12:00:00 -0200
description: review papers about semantic SLAM, and summarize what people are doing in this area.
permalink: /review-semantic-SLAM/
---


Why higher level features are needed:

More discriminative, which helps data association;

Serve to inform robotic tasks that require higher level information.


### HOW Semantic Information Is Used

- Objection Recognition

  - Feature-Based Recognition

  - Use Pre-Defined Models

  - Use Deep Learning Methods

- Segmentation

  - Plane Segmentation

     1. [5] considers all homogeneous planes as planar landmarks and the remaining non-planar regions as the regions corresponding to potential objects of interest. The ICP algorithm is then used to find matches between objects of interest. In this manner, no pre-defined object models are needed, and a dataset of objects as the system is running.

     2. [10] uses CNN to detect planes and then incorporating them into points-based LSD-SLAM to achieve better performance in low texture environment.

  - Scene Segmentation

     1. [4] tackles the problem of medium-term continuous tracking. Patch / feature-point based tracking may fail due to severe scale variation, but if we have the information about the semantic identity of the patch / feature-point, we can still perform correct matching. Meaning that although the patch appearance changes drastically, its semantic identity remains the same, for example from the same vehicle.

     2. [11] proposes that the error in outdoor-senario tracking is most likely caused by sky-region and car-region, based on its empirical studies. So the authors propose to use DNN to segment these regions out of the input image, by using a mask, and then perform ORB-SLAM on the rest regions of the image.



### Reference:

[1] R. F. Salas-Moreno, R. A. Newcombe, H. Strasdat, P. H. J. Kelly, and A. J. Davison. "SLAM++: Simultaneous Local- isation and Mapping at the Level of Objects". In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2013.

[2] M. Bloesch, J. Czarnowski, R. Clark, S. Leutenegger, and A. J. Davison. "CodeSLAM - Learning a Compact, Optimisable Representation for Dense Visual SLAM". In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.

[3] S. L. Bowman, N. Atanasov, K. Daniilidis, and G. J. Pappas. "Probabilistic Data Association for Semantic SLAM". IEEE/RSJ International Conference on Robotics and Automation (ICRA), 2017.

[4] K. Lianos, J. L. Schönberger, M. Pollefeys, and T. Sattler. "VSO: Visual Semantic Odometry". European Conference on Computer Vision (ECCV), 2018.

[5] S. Choudhary, A. J. B. Trevor, H. I. Christensen, and F. Dellaert. "SLAM with Object Discovery, Modeling and Mapping" IEEE/RSJ International Conference on Intelligent Robotics and Systems (IROS), 2014.

[6] N. Sünderhauf, T. T. Pham, Y. Latif, M. Milford, and I. Reid. "Meaningful Maps with Object-Oriented Semantic Mapping". IEEE/RSJ International Conference on Intelligent Robotics and Systems (IROS), 2017.

[7] M. Hosseinzadeh, Y. Latif, T. Pham, N. Sünderhauf, and I. Reid. "Towards Semantic SLAM: Points, Planes and Objects". IEEE/RSJ International Conference on Intelligent Robotics and Systems (IROS), 2017.

[8] B. Mu, S. Liu, L. Paull, J. Leonard, and J. P. How. "SLAM with Objects using a Nonparametric Pose Graph". IEEE/RSJ International Conference on Intelligent Robotics and Systems (IROS), 2016.

[9] C. Yu, Z. Liu, X. Liu, F. Xie, Y. Yang, Q. Wei, and Q. Fei. "DS-SLAM: A Semantic Visual SLAM Towards Dynamic Environments". IEEE/RSJ International Conference on Intelligent Robotics and Systems (IROS), 2018.

[10] S. Yang, Y. Song, M. Kaess, adn S. Scherer. "Pop-up SLAM: Semantic Monocular Plane SLAM for Low-texture Environments". arXiv, 2017.

[11] M. Kaneko, K. Iwami, T. Ogawa, T. Yamasaki, and K. Aizawa. "Mask-SLAM: Robust Feature-Based Monocular SLAM by Masking Using Semantic Segmentation". In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.

[12] J. Civera, D. Gálvez-López, L. Riazuelo, J. D. Tardós, and J. M. M. Montiel. "Towards Semantic SLAM Using a Monocular Camera". IEEE/RSJ International Conference on Robotics and Automation (ICRA), 2011. 

[13] S. Pillai and J. J. Leonard. "Monocular SLAM Supported Object Recognition". Robotics: Science and Systems, 2015.


### Appendix
  



