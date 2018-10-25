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


### Pre-processing

All image are resize to make the shorter side be 600 pixels.

{% highlight python %}
PILimg = PIL.Image.open("cars.jpg").convert("RGB")
img_w, img_h = PILimg.size
if img_w < img_h:
    scaled_w = 600
    scaled_h = img_h*600/img_w
else:
    scaled_w = img_w*600/img_h
    scaled_h = 600
print((img_w, img_h, scaled_w, scaled_h))

mytransform = transforms.Compose(
            [
                # Resize is for the purpose of speeding up
                # scale the shorter side to 600 pixels
                transforms.Resize((scaled_h,scaled_w)),
                
                # for the purpose of data augmentation
                # transforms.RandomHorizontalFlip(),
                
                # (H x W x C) in the range [0, 255] to (C x H x W) in the range [0.0, 1.0].
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ]
        )

in_img = mytransform(PILimg)
in_img = in_img.unsqueeze(0)  # add an additional dimension as first dimension
{% endhighlight python %}

And the whole structure of Faster R-CNN is as below:

{% highlight python %}
# due to the inconsistent input size, process 1 img per batch
class faster_rcnn(nn.Module):
    def __init__(self, bn=True, pretrain=True):
        super(faster_rcnn, self).__init__()
        
        #### RCNN base ####
        # load vgg16 with/without batch_normalization
        print('Using conv layers from vgg16...')
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
        # discard last max pooling layer
        # will be replaced with a RoI pooling layer
        self.RCNN_base = nn.Sequential(*list(model.features.children())[:-1])
        
        #### RPN ####
        self.RPN = RPN(s=[128, 256, 512], r=[1, 0.5, 2], n=3, bn=False, in_channels=512)
        
        #### RoI pooling ####
        self.RoI_pooling = RoI_Pooling(7, 7)
        
        #### RCNN head ####
        self.RCNN_head = nn.Sequential(*list(model.classifier.children())[:-1])
        # feature dimension of 4096 for VGG16, K is the # of object classes
        self.cls = nn.Linear(in_features=4096, out_features=K+1, bias=True)
        self.reg = nn.Linear(in_features=4096, out_features=4, bias=True)
            
    def forward(self, img):
        # get the base feature map
        feature_map = self.RCNN_base(img)
        
        # find regions of interests
        proposals = self.RPN(feature_map, nms_topN=2000, nms_thresh=0.7)
        
        # reshape all RoI to the same size
        RoI_feat = self.RoI_pooling(feature_map, proposals)
        # then pass the extracted roi feature through fc layers
        RoI_feat = self.RCNN_head(RoI_feat)
        
        # classification, output softmax scores
        class_scores = self.cls(RoI_feat)
        class_scores = F.softmax(class_scores)
        # regression, output bbox offsets to the gt
        bbox_coords = self.reg(RoI_feat)
        
        return class_scores, bbox_coords
{% endhighlight python %}


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
self.RCNN_base = nn.Sequential(*list(model.features.children())[:-1])
{% endhighlight %}


### R-CNN Head

R-CNN head consists of two parts. The first part contains 2 fully connected layers which takes in the fixed-length feature vector of RoIs as input. The second part has 2 parallel sibling output layers: one uses softmax for objectness estimation (produces the probability over all object classes plus a background class) and another regressor to predict the position of bounding boxes (4 values tuple). Since I'm using VGG16 for the R-CNN base, so I just adopt the first two FC layers from the VGG16 and then attaches two parallel siblings outputs to the FC layers. 

For the purpose of bounding box (b-box) refinement, since the proposed region of interest may not fully coincide with the object b-box, a class-specific b-box regressor is implemented with the goal of learning a transformation that maps the proposed b-box $$P$$ to a ground-truth b-box $$G$$ [2]. The transformation is parameterized as $$d_x(P), d_y(P), d_w(P), d_h(P)$$, and the map function can be formulated as:

$$
\begin{cases}
\hat{G_x} = P_w d_x(P) + P_x \\
\hat{G_y} = P_h d_y(P) + P_y \\
\hat{G_w} = P_w exp(d_w(P)) \\
\hat{G_h} = P_h exp(d_h(P))
\end{cases}
$$

where the first two equations compute the new b-box center and the second two find the new height and width. Using linear funciton, we have $$ d(P) = \boldsymbol{x}^T\phi_5(P) $$ where $$ \phi_5(P) $$ is the roi feature vector from CNN. Formulized in least squares, we have:

$$
\begin{cases}
\boldsymbol{w}^* = \underset{\hat{\boldsymbol{w}}}{\mathrm{argmin}} \sum_i^N (t^i - d(P^i))^2 \\
t_x = (G_x - P_x)/P_w \\
t_y = (G_y - P_y)/P_h \\
t_w = log(G_w/P_w) \\
t_h = log(G_h/P_h)
\end{cases}
$$


The code snippet is shown below:

{% highlight python %}
self.RCNN_head = nn.Sequential(*list(model.classifier.children())[:-1])
# feature dimension of 4096 for VGG16, K is the # of object classes
self.cls = nn.Linear(in_features=4096, out_features=K+1, bias=True)
self.reg = nn.Linear(in_features=4096, out_features=4, bias=True)
{% endhighlight python %}


### Region Proposal Network (RPN)

As defined in [4], a RPN taks an image (of any size) as input and outputs a set of rectangular object proposals along with repective objectness scores. This process is also a FCN, and it shares a common set of convolutional layers with the R-CNN base (13 shareable convolutional layers with VGG16).

Follow the procedure in [4], a small network is slided over the feature map from R-CNN base, which takes as input an $$n \times n$$ (e.g. $$3 \times 3$$) spatial window. The window is mapped to a lower-dimensional feature vector (512-d for VGG with ReLU following), and then 2 sibling $$1 \times 1$$ convolutional layers (i.e. FC layers) are used for box-regression and box-classification.

Furthermore, at each sliding-window location, mutiple region proposals are predicted simultaneously and the maximum possible proposals for each location is denoted as $$k$$. **The** $$k$$ **proposals are parameterized RELATIVE to** $$k$$ **reference boxes, which are called ANCHORs**. By default, 3 scales ($$128^2,\, 256^2,\, 512^2$$) and 3 aspect ratios ($$1:1,\, 1:2,\, 2:1$$) are associated with each anchor, yeilding $$k=9$$ anchors at each sliding position. Similar to the b-box regression formulation from R-CNN head, a transformation is learned to map b-boxes from the FCN to the anchors, and the output is the final region proposals needed later.

Following is a figure from [4] illustrating architechture of RPN. 

![Image](\assets\img\posts\rpn-structure.jpg)

However, I found this figure alone is not so easy to understand how RPN acutally works. So I created the flow-chart of the architecture of RPN based on my understanding, showing below.

![Image](\assets\img\posts\rpn-flowchart.jpg)

A code snippet for this procedure is shown below:

{% highlight python %}
class RPN(nn.Module):

    def __init__(self, s=[128, 256, 512], r=[1, 0.5, 2], n=3, bn=False, in_channels=512):
        super(RPN, self).__init__()
        # number of proposed boxes #
        self.scale = s  # meaning size of bbox in original image
        self.ratio = r  # meaning 1:1, 1:2, 2:1
        self.k = len(s)*len(r)
        self.anchors = self.ref_anchors(s, r)
        
        # n*n small network (sliding window) #
        # scale down from whatever dimension to 512
        self.conv3 = nn.Conv2d(in_channels, 512, kernel_size=n, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(512, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True)
        
        # 1*1 convolutional layer for reg and cls #
        # predicts 2 class probabilities(object is present / no object) for each anchor
        self.conv1_cls = nn.Conv2d(512, 2*self.k, kernel_size=1, stride=1, padding=0)
        # predicts 4 coordinates of a bounding box relative to each anchor ("t")
        self.conv1_reg = nn.Conv2d(512, 4*self.k, kernel_size=1, stride=1, padding=0)
        
    def forward(self, feat_map, nms_topN=2000, nms_thresh=0.7):
        ########## following is Network operation, all variables in torch!!
        ## scale dimension: (bsz=1, feat_chan, H, W) -> (bsz=1, 512, , H, W)
        interval = self.conv3(feat_patch)
        interval = self.bn(interval)
        interval = self.relu(interval)        
        ## cls
        # output tensor of shape (bsz=1, 18, H, W)
        cls_scores = self.conv1_cls(interval)
        _, cls_s_chan, H, W = cls_scores.size()
        # reshape to (bsz=1, 2, 9H, W)
        cls_scores_reshape = cls_scores.view(1, 2, H*self.k, W)
        # output tensor of shape (bsz=1, 2, 9H, W)
        cls_prob_reshape = F.softmax(cls_scores_reshape, 1)
        # reshape back to !!(bsz=1, 18, H, W)!!
        t_cls_prob = cls_prob_reshape.view(1, cls_s_chan, H, W)
        ## reg
        # output tensor of shape !!(bsz=1, 36, H, W)!!
        t_reg_bbox = self.conv1_reg(interval)
        
        ########## following is Post-network operation, 
        ########## torch variables are specified with "t_"
        ## get anchors at each position:
        # 1 pixel in conv5 is a 16x16 patch in original img, use (7,7) as center
        shift_row = np.arange(0, H) * 16 + 7
        shift_col = np.arange(0, W) * 16 + 7
        shift_row, shift_col = np.meshgrid(shift_row, shift_col)
        # new_centers is in shape (HxW, 4)
        new_centers = np.vstack(shift_row.ravel(), shift_col.ravel(), 
                                shift_row.ravel(), shift_col.ravel()).transpose()
        # reshape to: anchor (1, 9, 4), centers (HxW, 1, 4)
        t_centers = torch.from_numpy(new_centers).float().view(H*W, 1, 4)
        t_ref_anchors = torch.from_numpy(self.anchors).float().view(1, 9, 4)
        # anchors at each location, (bsz=1, HxW, 9, 4)
        t_anchors = (t_anchors + t_centers).unsqueeze(0).expand(1, -1, -1, -1)
        
        ## map the proposed bbox into the anchor boxes
        # reshape output of network to the same format
        t_reg_bbox = t_reg_bbox.permute(0, 2, 3, 1).view(1, H*W, 9, 4)
        # [object prob, non-obj prob]
        t_cls_prob = t_cls_prob.permute(0, 2, 3, 1).view(1, -1, 2)
        # refine bbox using regression result, in shape of (bsz=1, H*W*9, 4)
        t_init_proposals = self.inverse_parameterize(t_reg_bbox, t_anchors)
        # [start_r, start_c, end_r, end_c]
        t_init_proposals = t_init_proposals.view(1, -1, 4)
        
        ## ignore cross-boundary proposals FOR TRAINING, test clip to bd, torch.clamp()
        img_h, img_w = H*16, W*16
        # cross-bd ones: 1, in-bd ones: 0
        comp_r_lower = torch.lt(t_init_proposals[0,:,0], torch.zeros(H*W*9))
        comp_r_higher = torch.gt(t_init_proposals[0,:,2], torch.ones(H*W*9)*img_h-1)
        comp_c_lower = torch.lt(t_init_proposals[0,:,1], torch.zeros(H*W*9))
        comp_c_higher = torch.gt(t_init_proposals[0,:,3], torch.ones(H*W*9)*img_c-1)
        comp_cross_bd = torch.le(comp_r_lower + comp_r_higher + 
                                 comp_c_lower + comp_c_higher, torch.zeros(H*W*9))
        t_inbd_proposals = t_init_proposals[:, comp_cross_bd, :]
        t_inbd_scores = t_cls_prob[:, comp_cross_bd, :]
        
        ## non-maximum suppression
        # sort by the probability a object presents
        _, order = torch.sort(t_inbd_scores[:,:, 0], descending=True)
        roi_proposals = []
        for idx in order.numpy():
            t_tmp_roi = t_inbd_proposals[0, idx, :]
            b_accept_roi = True
            for t_roi in roi_proposals:
                iou_score = self.calculate_IoU(t_tmp_roi, t_roi)
                if iou_score > nms_thresh:
                    accept_roi = False
                    break
            roi_proposals.push_back(t_tmp_roi) if b_accept_roi
            break if len(roi_proposals) >= nms_topN
        
        return roi_proposals
    
    ## More about staticmethod : ##
    # 1. Static methods don’t have access to cls or self, won't modify class state
    # 2. They work like regular functions but belong to the class’s namespace.
    #   can be called without initialize class
    # 3. This can have maintenance benefits.
    @staticmethod
    def ref_anchors(scale, ratio):
        # this method generates the desired anchors at location (0,0)
        # anchors in the form of (start_row, start_col, end_row, end_col)
        anchors = np.empty((len(scale)*len(ratio), 4), dtype=int)
        count = 0
        for s in scale:
            area = s*s
            for r in ratio:
                # calculate width and height
                width = np.sqrt(area/r)
                height = width*r
                # calculate start and end coords
                half_wid = int(round(width/2))
                half_hei = int(round(height/2))
                anchors[count, :] = [-half_hei+1, -half_wid+1, half_hei, half_wid]
                count = count + 1
        return anchors
        
    @staticmethod
    def inverse_parameterize(reg_bbox, anchor_bbox):
        # (bsz=1, H*W, 9, 4), 
        # in order of: tx, ty, tw, th; start_row, start_col, end_row, end_col
        # all in float type
        # get center_row, center_col, height, width for anchor boxes
        an_height = anchor_bbox[:, :, :, 2] - anchor_bbox[:, :, :, 0] + 1.
        an_width = anchor_bbox[:, :, :, 3] - anchor_bbox[:, :, :, 1] + 1.
        an_center_row = anchor_bbox[:, :, :, 0] + an_height*0.5
        an_center_col = anchor_bbox[:, :, :, 1] + an_width*0.5
        
        # for every anchor, calculate the final coordinates of initial proposals
        prop_x = reg_bbox[:,:,:,0] * an_width + an_center_col
        prop_y = reg_bbox[:,:,:,1] * an_height + an_center_row
        prop_w = torch.exp(reg_bbox[:,:,:,2]) * an_width
        prop_h = torch.exp(reg_bbox[:,:,:,3]) * an_height
        
        # get to the format of start_row, start_col, end_row, end_col
        prop_start_r = (prop_y - 0.5*prop_h).unsqueeze(3)
        prop_end_r = (prop_y + 0.5*prop_h).unsqueeze(3)
        prop_start_c = (prop_x - 0.5*prop_w).unsqueeze(3)
        prop_end_c = (prop_x + 0.5*prop_w).unsqueeze(3)
        init_proposals = torch.cat((prop_start_r, prop_start_c, prop_end_r, prop_end_c), 3)
        
        # return in shape of (bsz=1, H*W, 9, 4), start_row, start_col, end_row, end_col
        return init_proposals
        
    @staticmethod
    def calculate_IoU(bbox1, bbox2):
        # input type: torch tensors of size 4
        # convert to numpy
        np_bbox1 = bbox1.numpy()
        np_bbox2 = bbox2.numpy()
        # get width and height
        w1 = bbox1[3] - bbox1[1] + 1.
        h1 = bbox1[2] - bbox1[0] + 1.
        w2 = bbox2[3] - bbox2[1] + 1.
        h2 = bbox2[2] - bbox2[0] + 1.
        # get area of overlap
        upperleft_r = max(bbox1[0], bbox2[0])
        upperleft_c = max(bbox1[1], bbox2[1])
        lowerright_r = min(bbox1[2], bbox2[2])
        lowerright_c = min(bbox1[3], bbox2[3])
        return 0. if upperleft_r >= lowerright_r or upperleft_c >= lowerright_c
        
        area_inter = (lowerright_r-upperleft_r+1.) * (lowerright_c-upperleft_c+1.)
        area_union = (w1*h1) + (w2*h2) - area_inter
        return float(area_inter) / float(area_union)
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
        
    def forward(self, feature_map, list_rois):
        # list_rois is a list of torch_tensors (r_s, c_s, h_e, w_e), which are the region proposals
        # instead output a list of feature vectors, concatenate all roi feat vector along the batch dimension
        # get first RoI
        roi_r, roi_c, roi_r_end, roi_c_end = list_rois[0].numpy()
        feat_patch = feature_map[:, :, round(roi_r/16):round(roi_r_end/16),
                                 round(roi_c/16):round(roi_c_end/16)]
        fixed_size_feat = MaxPooling(feat_patch)
        for i in range(1, len(list_rois)):
            roi_r, roi_c, roi_r_end, roi_c_end = list_rois[i].numpy()
            feat_patch = feature_map[:, :, round(roi_r/16):round(roi_r_end/16),
                                     round(roi_c/16):round(roi_c_end/16)]
            fixed_size_feat_tmp = MaxPooling(feature_patch)
            fixed_size_feat = torch.cat((fixed_size_feat, fixed_size_feat_tmp), 0)
        # output feature is a torch tensor of size (bsz=#_of_rois, chans, H, W)
        return fixed_size_feat
        
    def MaxPooling(self, feat_roi):
        bsz, chans, h, w = feat_roi.size()
        # padded with zeros if h,w < H,W
        fixed_size_feat = torch.zeros(bsz, chans, self.H, self.W)
        # sub_window size
        sub_win_height = int(h/self.H)
        sub_win_height = 1 if sub_win_height == 0
        sub_win_width = int(w/self.W)
        sub_win_width = 1 if sub_win_width == 0
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


### Training Procedure



### Evaluation Metric

mean Average Precision (mAP): 

It's the mean of AP. 

Average Precision (AP) computes the average of the maximum precisions at different recall values.

True Positive (TP): Predicted as positive while it's positive in GT.

False Positive (FP): ........... positive .......... negative ......

False Negative (FN): ........... negative .......... positive ......

True Negative (TN): ............ negative .......... negative ......

$$
\begin{cases}
precision = \frac{TP}{TP + FP} \\
recall = frac{TP}{TP + FN} \\
F1 = 2 \frac{precision \times recall}{precision + recall} \\
mAP = \frac{\sum_{q=1}^Q Avg(P(q))}{Q}
\end{cases}
$$

*e.g.* Given 3 classes of objects in an image, calss 1 has 3 instances, class 2 has 5 instances and class 3 has 1 instance. We obtain 10 detections for each class, o for correct detection and x for misses.


---------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- 
class 1:   |  o  |  x  |  o  |  x  |  x  |  o  |  x  |  x  |  o  |  o
precision: | 1.0 | 0.5 | 0.67| 0.5 | 0.4 | 0.5 | 0.43| 0.38| 0.44| 0.5
recall:    | 0.2 | 0.2 | 0.4 | 0.4 | 0.4 | 0.6 | 0.6 | 0.6 | 0.8 | 1.0
---------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- 
class 2:   |  x  |  o  |  x  |  x  |  o  |  x  |  o  |  x  |  x  |  x
precision: | 0.0 | 0.5 | 0.33| 0.25| 0.4 | 0.33| 0.43| 0.38| 0.33| 0.3
recall:    | 0.0 | 0.33| 0.33| 0.33| 0.67| 0.67| 1.0 | 1.0 | 1.0 | 1.0
---------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- 
class 3:   |  x  |  x  |  x  |  o  |  x  |  x  |  x  |  x  |  x  |  x
precision: | 0.0 | 0.0 | 0.0 | 0.25| 0.2 | 0.17| 0.14| 0.13| 0.11| 0.1
recall:    | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0

$$
\begin{cases}
AP_1 &= \frac{1.0\times 2 + 0.67\times 3 + 0.5\times 3 + 0.44 + 0.5}{10} = 0.645 \\
AP_2 &= \frac{0 + 0.5\times 3 + 0.4\times 2 + 0.43\times 4}{10} = 0.502 \\ 
AP_3 &= \frac{0\times 3 + 0.25\times 7}{10} = 0.175
mAP  &= \frac{0.645 + 0.502 + 0.175}{3} = 0.441
\end{cases}
$$


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

  



