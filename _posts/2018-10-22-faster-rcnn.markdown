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

For semantic SLAM, work proposed by *M. Hosseinzadeh et al.* [8] and *B. Mu et al.* [9] utilize Faster R-CNN to provide object information for their systems.

In the following sections, I will use Faster R-CNN as the example to delve into the R-CNN family. The Faster R-CNN is diveded into the following parts: R-CNN base which is the convolutional layers that computes the feature maps; RoI pooling that resize all RoIs into the same size, RPN that proposals regions of interest, and the R-CNN head which predicts the class probabilities and regresses the coordinates of the bounding boxes. Finally, the training procedure and the evaluation metic will be discussed.


### R-CNN Base

The convolutional layers of VGG16 (up to conv5, before the max pooling layer) are used to form the R-CNN Base, i.e. it is a fully convolutional network (FCN) [6]. The structure of VGG16 is shown below.

![Image](\assets\img\posts\vgg16-structure.jpg)

<u>What is FCN?</u> FCN is a net with only layers of the form below which computes a nonlinear filter.

$$ 
\begin{cases}
y_{ij} = f_{ks}(\{ x_{si+\delta i, sj+\delta j} \}_{0\leq\delta i, \delta j \leq k}) \\
f_{ks} \circ g_{k's'} = (f \circ g)_{k'+(k-1)s', ss'}
\end{cases}
$$

i.e. there are no fully connected layers attached to the convolutional layers.

A code snippet of creating the R-CNN base with PyTorch is shown below:

{% highlight python %}
# In __init__():
    self.bn_flag = bn
    if bn:
        model = models.vgg16_bn(pretrained=pretrain)
        self.layer_dict = {
            'conv1_1': 2, 'conv1_2': 5, 'conv2_1': 9, 'conv2_2': 12,
            'conv3_1': 16, 'conv3_2': 19, 'conv3_3': 22,
            'conv4_1': 26, 'conv4_2': 29, 'conv4_3': 32,
            'conv5_1': 36, 'conv5_2': 39, 'conv5_3': 42
        }
    else:
        model = models.vgg16(pretrained=pretrain)
        self.layer_dict = {
            'conv1_1': 1, 'conv1_2': 3, 'conv2_1': 6, 'conv2_2': 8,
            'conv3_1': 11, 'conv3_2': 13, 'conv3_3': 15,
            'conv4_1': 18, 'conv4_2': 20, 'conv4_3': 22,
            'conv5_1': 25, 'conv5_2': 27, 'conv5_3': 29
        }  
    self.RCNN_base = nn.Sequential(*list(model.features.children())[:-1])

# In forward():
    # get the base feature map
    feature_map = self.RCNN_base(img)
{% endhighlight %}


### Region Proposal Network (RPN)

As defined in [4], a RPN taks an image (of any size) as input and outputs a set of rectangular object proposals along with repective objectness scores. This process is also a FCN, and it shares a common set of convolutional layers with the R-CNN base (13 shareable convolutional layers with VGG16).

Follow the procedure in [4], a small network is slided over the feature map from R-CNN base, which takes as input an $$n \times n$$ (e.g. $$3 \times 3$$) spatial window. The window is mapped to a lower-dimensional feature vector (512-d for VGG with ReLU following), and then 2 sibling $$1 \times 1$$ convolutional layers (i.e. FC layers) are used for box-regression and box-classification. Furthermore, at each sliding-window location, mutiple region proposals are predicted simultaneously adn the maximum possible proposals for each location is denoted as $$k$$. **The** $$k$$ **proposals are parameterized RELATIVE to** $$k$$ **reference boxes, which are called ANCHORs**. By default, 3 scales ($$128^2,\, 256^2,\, 512^2$$) and 3 aspect ratios ($$1:1,\, 1:2,\, 2:1$$) are associated with each anchor, yeilding $$k=9$$ anchors at each sliding position. Following is a figure from [4] illustrating architechture of RPN.

![Image](\assets\img\posts\rpn-structure.jpg)

However, I found this figure alone is not so easy to understand how RPN acutally works. So I created the flow-chart of the architecture of RPN based on my understanding, showing below.

![Image](\assets\img\posts\rpn-flowchart.jpg)

A code snippet for this procedure is shown below:

{% highlight python %}
class RPN(nn.Module):

    def __init__(self, s=3, r=3, n=3, bn=True, in_channels=512, out_channels=512):
        super(RPN, self).__init__()
        # number of proposed boxes #
        self.scale = [128, 258, 512] if s=3 else None  # meaning size of bbox in original image
        self.ratio = [1, 0.5, 2] if r=3 else None      # meaning 1:1, 1:2, 2:1
        self.k = s*r
        
        # n*n small network (sliding window) #
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=n, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True)
        
        # 1*1 convolutional layer for reg and cls, both take in the feature patch #
        # predicts 2 class probabilities(object is present / no object) for each anchor
        # as 2-class softmax layer
        # outputs the probabilities (dim = 2*k)
        self.conv1_cls = nn.Conv2d(out_channels, 2*self.k, kernel_size=1, stride=1, padding=0)
        # predicts 4 coordinates of a bounding box relative to each anchor
        # outputs the coordinates (dim = 4*k)
        self.conv1_reg = nn.Conv2d(out_channels, 4*self.k, kernel_size=1, stride=1, padding=0)
        
    def forward(self, feat_patch):
        # pass through the small network mentioned in the paper
        interval = self.conv3(feat_patch)
        interval = self.bn(interval)
        interval = self.relu(interval)
        
        # regressor for bbox coordinates
        reg = self.conv1_reg(interval)
        # softmax classifier for probabilities
        cls = self.conv1_cls(interval)
        
        return cls, reg
{% endhighlight python %}






### Region-of-Interest (RoI) Pooling

Since the input of the Faster R-CNN network is arbitrary, unlike the fixed input size of 224 for VGG16, RoI pooling layer is implemented to convert the features inside any valid region of interest into small feature map of fixed size of $$H \times W$$ (e.g. $$7 \times 7$$) [3]. It works bi dividing $$h \times w$$ RoI window into an $$H \times W$$ and then max-pooling the values in each sub-window of each channel independently into corresponding output grid cell. The RoI pooling layer can be considered as a special case of SPPNet [7], illustrated below:

![Image](\assets\img\posts\roi-pooling.jpg)

A code snippet of the PyTorch implementation is shown below. Note that the RoI pooling layer takes the feature map extracted from a mini-batch of images (bsz=1, feat_chan, feat_h, feat_w) and a list of proposed regions (roi_topleft_r, roi_topleft_c, roi_h, roi_w) as inputs, outputs a list of features of fixed size (bsz=1, feat_chan, H, W). Since the sizes of the input images are arbitrary, we only consider the case of 1 image per mini-batch for now.

{% highlight python %}
class RoI_Pooling(nn.Module):

    def __init__(self, H=7, W=7):
        super(RoI_Pooling, self).__init__()
        self.H = H
        self.W = W
        
    def forward(self, feature_map, in_tuple):
        # in_tuple is a list of four-tuple (r, c, h, w), which are the region proposals
        # instead output a list of feature vectors, concatenate all roi feat vector along the batch dimension
        # get first RoI
        roi_r, roi_c, h, w = in_tuple[0]
        fixed_size_feat = MaxPooling(feature_map[:, :, roi_r:roi_r+h, roi_c:roi_c+w])
        for i in range(1, len(in_tuple)):
            roi_r, roi_c, h, w = in_tuple[i]
            fixed_size_feat_tmp = MaxPooling(feature_map[:, :, roi_r:roi_r+h, roi_c:roi_c+w])
            fixed_size_feat = torch.cat((fixed_size_feat, fixed_size_feat_tmp), 0)
        # output feature is a torch tensor of size (bsz=#_of_rois, chans, H, W)
        return fixed_size_feat
    
    def MaxPooling(self, feat_roi):
        bsz, chans, h, w = feat_roi.size()
        fixed_size_feat = torch.zeros(bsz=1, chans, self.H, self.W)
        # sub_window size
        sub_win_height = h/self.H
        sub_win_width = w/self.W
        # max pooling
        for r in range(self.H):
            start_r = self.H*r
            end_r = start_r + sub_win_height
            for c in range(self.W):
                start_c = self.W*c
                end_c = start_c + sub_win_width
                # get the patch
                patch_feat = feat_roi[:,:, start_r:end_r, start_c:end_c]
                # find maximum in each channel
                max_along_h = torch.max(patch_feat, 2)
                max_final = torch.max(max_along_h[0], 2)
                fixed_size_feat[:, :, r, c] = max_final[0]
        # return feature of (bsz, chans, H, W)
        return fixed_size_feat
{% endhighlight %}

{% highlight python %}
# In __init__():
    self.RoI_pooling = RoI_Pooling(7, 7)

# In forward():
    RoI_feat = self.RoI_pooling(feature_map, proposals)
{% endhighlight %}


### R-CNN Head

R-CNN head consists of two parts. The first part contains 2 fully connected layers which takes in the fixed-length feature vector of RoIs as input. The second part has 2 parallel sibling output layers: one uses softmax for objectness estimation (produces the probability over all object classes plus a background class) and another regressor to predict the position of bounding boxes (4 values tuple). Since I'm using VGG16 for the R-CNN base, so I just adopt the first two FC layers from the VGG16 and then attaches two parallel siblings outputs to the FC layers. The code snippet is shown below:


{% highlight python %}
# In __init__():
    self.RCNN_base = nn.Sequential(*list(model.classifier.children())[:-1])
    # feature dimension of 4096 for VGG16, K is the # of object classes
    self.cls = nn.Linear(in_features=4096, out_features=K+1, bias=True)
    self.reg = nn.Linear(in_features=4096, out_features=4, bias=True)

# In forward():
    RoI_feat = self.RCNN_base(RoI_feat)
    # softmax score
    class_scores = self.cls(RoI_feat)
    # bbox coordinates
    bbox_coords = self.reg(RoI_feat)
{% endhighlight python %}



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

[7] SPPNet

[8] M. Hosseinzadeh, Y. Latif, T. Pham, N. Sünderhauf, and I. Reid. “Towards Semantic SLAM: Points, Planes and Objects”. IEEE/RSJ International Conference on Intelligent Robotics and Systems (IROS), 2017.

[9] B. Mu, S. Liu, L. Paull, J. Leonard, and J. P. How. “SLAM with Objects using a Nonparametric Pose Graph”. IEEE/RSJ International Conference on Intelligent Robotics and Systems (IROS), 2016.


### Appendix

<u>Objection Recognition</u>

![Image](\assets\img\posts\SLAM++.png)

![Image](\assets\img\posts\point-plane-object-SLAM.png)
  



