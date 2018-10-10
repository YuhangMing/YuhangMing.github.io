---
layout: post
title:  "REVIEW: Semantic SLAM (Keep Updating)"
date:   2018-10-09 12:00:00 -0200
description: review papers about semantic SLAM, and summarize what people are doing in this area.
permalink: /review-semantic-SLAM/
---


### Towards Semantic SLAM

What is semantic SLAM: 

[6] states that semantic SLAM requires information flow in both directions: SLAM helping semantics and semantics helping SLAM.

Why detecting objects helps: 

[1] states that tracking one object in 6DoF is enough to localise a camera, and reliable relocalisation of a lost camera or loop closure detection can be performed on the basis of just a small number of object measurements due to their high saliency. Further, and crucially, instant recognition of objects provides great efficiency and robustness benefits via the active approaches it permits to tracking and object detection, guided entirely by the dense predictions we can make of the positions of known objects.

Why higher level features are needed: 

First of all, as stated above, they are more discriminative, which helps data association; Secondly, they can serve to inform robotic tasks that require higher level information, saying if a robot that needs to reason about moving from point A to B needs access to place identities (room, corridor, kitchen etc), while a robot that manipulates objects needs information about object identities and affordances (What can be done with the object? How to grab it? How is it supported in space?) [6].

Challenges with object SLAM:

When using objects as higher level features, there are 2 major challenges, as mentioned in [8]. 
First, there are multiple instances of the object class. Without correct data association, it is hard to distinguish different object instances. Standard pose-graph SLAM algorithms can only optimize poses with exact data association, such as g2o, isam, gtsam. 
The second challenge is high false positive rates. Blindly using these unfiltered detections in standard SLAM algorithms will lead to “non-exist” nodes and cause loop closure failures.


### Semantic Information
#### --- How it is extracted and used in SLAM systems

<u>Objection Recognition</u>

- Feature-Based Recognition

  [12] proposes to use SURF features and reconstruct the object in a SfM manner.

- Use Pre-Defined Models

  [1] builds models of objects using KinectFusion, and then 3D objects are detected by matching to these pre-defined models. The camera poses are estimated using a fast dense ICP and then updated with detected objects. Graph optimization is formulized to jointly optimize camera and object poses. The pipeline of the SLAM++ system is shown below.

  ![Image](\assets\img\posts\SLAM++.png)

  [3] doesn't mention how the object detection is performed but the objects to be detected are limited to doors and red chairs only. So a reasonable guess is that the model of doors and red chairs are built before-hand and then perform 3D object detection. With objects obtained, the data association and optimization are formulized into a EM style formulation, by performing object classes and data association in the E-step and pose graph optimization in M-step.

- Use Deep Learning Methods

  [6] uses SLAM to help create accurate maps with objects. Due to the online requirements, the authors model objects as separate entities in space instead of generating object instances from point-wise labelling map. The objects are detected using Single Shot MultiBox Detector (SSD) and 3D unsupervised segmentation is performed leveraging depth information. Then a 3D point cloud is assigned to every detected objected, and add new map objects if necessary.

  [7] proposes a more general setup of semantic SLAM which uses points, planes and objects. <u>This is the first real-time semantic SLAM system proposed in literature that uses previously unseen objects as landmarks</u>. The authors use Faster-RCNN to perform object recognition and then instead of bounding boxes, ellipsoids are used to represent objects. Points, planes, and objects are jointly optimized in a factor-graph. The pipeline of the semantic point-plane-object SLAM system is shown below.

  ![Image](\assets\img\posts\point-plane-object-SLAM.png)

  [8] also uses Faster-RCNN to detec objects and then a novel nonparametric pose graph that models data association and SLAM in a single framework.

<u>Semantic Segmentation</u>

- Plane Segmentation

  [5] considers all homogeneous planes as planar landmarks and the remaining non-planar regions as the regions corresponding to potential objects of interest. The ICP algorithm is then used to find matches between objects of interest. In this manner, no pre-defined object models are needed, and a dataset of objects as the system is running. The discovered objects are used as landmarks to help SLAM systems perform localization in an online manner.

  [10] uses CNN to detect planes and then incorporating them into points-based LSD-SLAM to achieve better performance in low texture environment. A planar SLAM formulation is proposed.

- Scene Segmentation

  [4] tackles the problem of medium-term continuous tracking. Patch / feature-point based tracking may fail due to severe scale variation, but if we have the information about the semantic identity of the patch / feature-point, we can still perform correct matching. Meaning that although the patch appearance changes drastically, its semantic identity remains the same, for example from the same vehicle. (Methods used for segmentation are not mentioned.) The authors also propose a semantic cost function for optimization.

  [9] aims to improve SLAM performance in the dynamic environments. The authors incorporate the semantic information with optical flow to determine whether the object is moving. The segmentation is performed using SegNet by *Badrinarayanan et al.* Given results from pixel-wise semantic segmentation and moving consistency check, the authors decide if an object is moving based on the following criteria: if the number of dynamic points producedby moving consistency check fall in the contours of a segmented object is larger than a specific threshold, then this object is determined to be moving. If the segmented object is determined to be moving, then all the feature points located in the object’s contour is considered as outlier and will be removed.

  [11] proposes that the error in outdoor-senario tracking is most likely caused by sky-region and car-region, based on its empirical studies. So the authors propose to use DNN (DeepLab v2 by *Chen et al.*) to segment these regions out of the input image, by using a mask, and then perform ORB-SLAM on the rest regions of the image.


### Optimization


### Map


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


### Appendix
  



