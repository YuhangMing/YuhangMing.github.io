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

<u>Objection Recognition</u>

- Feature-Based Recognition

     [12] proposes to use SURF features and reconstruct the object in a SfM manner.

- Use Pre-Defined Models

     1. [1] builds models of objects using KinectFusion, and then 3D objects are detected by matching to these pre-defined models. The pipeline of the SLAM++ system is shown below.

        ![Image](\assets\img\posts\SLAM++.png)

     2. [3] doesn't mention how the object detection is performed but the objects to be detected are limited to doors and red chairs only. So a reasonable guess is that the model of doors and red chairs are built before-hand and then perform 3D object detection.

- Use Deep Learning Methods

     1. [6] uses SLAM to help create accurate maps with objects. Due to the online requirements, the authors model objects as separate entities in space instead of generating object instances from point-wise labelling map. The objects are detected using Single Shot MultiBox Detector (SSD) and 3D segmentation is performed leveraging depth information.

     2. [7] proposes a more general setup of semantic SLAM which uses points, planes and objects. **This is the first real-time semantic SLAM system proposed in literature that uses previously unseen objects as landmarks**. The authors use Faster-RCNN to perform object recognition and then instead of bounding boxes, ellipsoids are used to represent objects. The pipeline of the semantic point-plane-object SLAM system is shown below.

        ![Image](\assets\img\posts\point-plane-object-SLAM.png)

    3. [8] also uses Faster-RCNN to detec objects and then a novel nonparametric pose graph that models data association and SLAM in a single framework.

<u>Semantic Segmentation</u>

- Plane Segmentation

     1. [5] considers all homogeneous planes as planar landmarks and the remaining non-planar regions as the regions corresponding to potential objects of interest. The ICP algorithm is then used to find matches between objects of interest. In this manner, no pre-defined object models are needed, and a dataset of objects as the system is running.

     2. [10] uses CNN to detect planes and then incorporating them into points-based LSD-SLAM to achieve better performance in low texture environment.

- Scene Segmentation

     1. [4] tackles the problem of medium-term continuous tracking. Patch / feature-point based tracking may fail due to severe scale variation, but if we have the information about the semantic identity of the patch / feature-point, we can still perform correct matching. Meaning that although the patch appearance changes drastically, its semantic identity remains the same, for example from the same vehicle.

     2. [9] aims to improve SLAM performance in the dynamic environments. The authors incorporate the semantic information with optical flow to determine whether the object is moving. Given results from pixel-wise semantic segmentation and moving consistency check, the authors decide if an object is moving based on the following criteria: if the number of dynamic points producedby moving consistency check fall in the contours of a segmented object is larger than a specific threshold, then this object is determined to be moving. If the segmented object is determined to be moving, then all the feature points located in the object’s contour is considered as outlier and will be removed.

     3. [11] proposes that the error in outdoor-senario tracking is most likely caused by sky-region and car-region, based on its empirical studies. So the authors propose to use DNN to segment these regions out of the input image, by using a mask, and then perform ORB-SLAM on the rest regions of the image.


### HOW Optimization Is Performed


### HOW The Map Is Built


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
  



