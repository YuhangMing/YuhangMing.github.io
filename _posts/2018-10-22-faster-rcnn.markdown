---
layout: post
title:  "Delving into the R-CNN Family"
date:   2018-10-22 16:00:00 -0200
description: Explain faster R-CNN in details along with seudo-code in python/pytorch
permalink: /faster-rcnn/
---


### General Information

With the booming advance of the deep neural networks (DNNs), more and more researchers have been applying DNNs in various vision problems. Among all, Regions with CNN features (R-CNN) [2] by *R. Girshick et al.* is the first to show a CNN can lead to dramatically outstanding object detection performance on PASCAL VOC challenge. However, R-CNN has a significant drawback: detection with VGG16 takes 47s/image (on a GPU). Hence, an improved version of R-CNN is proposed by sharing convolutions across all region proposals, known as Fast R-CNN [3]. The Fast R-CNN system is able to achieve real-tine detection when ignoring time spent on region proposals, with 0.3s/image (excluding object proposal time). Furthermroe, to break the bottleneck of improving detection time, *S. Ren et al.* proposed Faster R-CNN [4] with the key contribution of replacing Selective Search in Fast R-CNN with Region Proposal Network (RPN), which shares the convolutional layers with Fast R-CNN. This improvement equips the system with the detection frame rate of 5 frame per second (fps) on a GPU. Moving on from object detection to semantic segmentation, Mask R-CNN [5] extends Faster R-CNN by adding a branch for predicting object mask. The architecture of these 4 networks are shown in the figure below and these 4 netowrks together forms the famous R-CNN family of object detection.

![Image](\assets\img\posts\rcnn-family.jpg)

For semantic SLAM, work proposed by *M. Hosseinzadeh et al.* [7] and *B. Mu et al.* [8] utilize Faster R-CNN to provide object information for their systems.

In the following sections, I will use Faster R-CNN as the example to delve into the R-CNN family. The Faster R-CNN is diveded into the following parts: R-CNN base which is the convolutional layers that computes the feature maps; RoI pooling that resize all RoIs into the same size, RPN that proposals regions of interest, and the R-CNN head which predicts the class probabilities and regresses the coordinates of the bounding boxes. Finally, the training procedure and the evaluation metic will be discussed.


### R-CNN Base




### Region-of-Interest (RoI) Pooling




### Region Proposal Network (RPN)




### R-CNN Head




### Training Procedure



### Evaluation Metric

mAP



### Reference:

[1] VGG16

[2] R-CNN

[3] Fast R-CNN

[4] Faster R-CNN

[5] Mask R-CNN

[6] FCN

[7] M. Hosseinzadeh, Y. Latif, T. Pham, N. Sünderhauf, and I. Reid. “Towards Semantic SLAM: Points, Planes and Objects”. IEEE/RSJ International Conference on Intelligent Robotics and Systems (IROS), 2017.

[8] B. Mu, S. Liu, L. Paull, J. Leonard, and J. P. How. “SLAM with Objects using a Nonparametric Pose Graph”. IEEE/RSJ International Conference on Intelligent Robotics and Systems (IROS), 2016.


### Appendix

<u>Objection Recognition</u>

![Image](\assets\img\posts\SLAM++.png)

![Image](\assets\img\posts\point-plane-object-SLAM.png)
  



