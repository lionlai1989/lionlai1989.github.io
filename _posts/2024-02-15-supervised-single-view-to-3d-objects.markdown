---
layout: single
title: Supervised Single-View to 3D Objects
categories: []
description: "."
toc: true
wip: true
date: 2024-02-15
---

Work in progress.

3D reconstruction from a single view is very similar to the process through which we
recognize objects in the real world. When we look at a chair from one angle, we know it
is a chair and can intuitively _imagine_ what it would look like from other angles. It's
not like a chair viewed from one angle will look like an airplane from another angle.
That being said, if you were determined to design an airplane that looks like a chair
from a specific viewpoint, then everything in this post is inapplicable. 🤣

# How is it Possible to Know the Depth from a Single View?

Deriving depth information from a single view is an ill-posed problem, meaning there are
infinitely many possible solutions for reconstructing the 3D scene from a 2D image. This
complexity arises because a single ray projected from the camera center to a pixel in
the image can intersect with any point along its path, making the depth information
inherently ambiguous without additional cues.

<p align="center">
<img alt="camera" src="/assets/images/2024-02-15/single_view_projection.png" width="45%">
<br>
</p>
The image above is adapted from [here](http://www.cs.toronto.edu/~fidler/slides/2015/CSC420/lecture12_hres.pdf).

Despite this being an ill-posed problem, it is still possible to infer depth from a
single view by exploiting the regularities and cues present in the natural world. For
instance, depth can be inferred from single images through cues such as occlusion (where
one object blocks another), relative size (larger objects are perceived as closer),
perspective (parallel lines appear to converge with distance), and shading/lighting.

Furthermore, we can train models to recognize patterns and learn these visual cues from
extensive datasets where the true depth or shape information is known.

In this post, I will show how numerous images and objects' 3D structures can be used to
train a neural network to deduce a 3D shape from a single 2D image.

# What Coordinate Systems Should We Predict 3D Models In?

Different coordinates systems serves different purposes in 3D world. Here, three
coordinate systems related to this subject are brought up, each with different
charateristics.

1.  **Camera Coordinate System**: it is inherently aligned with our perspective. This
    system adjusts the coordinates of an object based on its movements or rotations
    relative to the camera. This means that any change in the object's position or
    orientation directly alters its coordinates within this system. However, this
    approach has its limitations. It can introduce confusion between the actual shape of
    an object and its perceived location from the camera's viewpoint. This overlap of
    shape and position uncertainties can make it challenging to accurately predict a 3D
    model's shape.

    <p align="center">
    <img alt="camera" src="/assets/images/2024-02-15/camera_coordinate_system.png" width="45%">
    <br>
    Camera Coordinate System. The image is adapted from "Methods for Structure from Motion" by Henrik Aanæs.
    </p>

2.  **View-aligned Object-centric Coordinate System**: it centers inside the object,
    usually at the average location of its parts. This system decouples the object's
    shape from its spatial position.

    What makes it "view-aligned" is that the object's coordinates adjust based on where
    we're looking from, ensuring that the object's coordinates always relate directly to
    our viewpoint. This means that the object's coordinates don't change if we only vary
    the distance between the observer and the object along the direction from the camera
    to the object. A major limitation of this approach is that we need to generate a
    distinct 3D shape for each viewpoint of the object, complicating the model
    prediction process.

3.  **Object-centric "Canonical" Coordinate System**: In this system, a "canonical"
    definition for axis orientation is established where the Y-axis points up, and the
    -Z-axis faces the front of the object. Defining the "front" of an object can
    sometimes be challenging and is often determined by convention or the dataset
    creator. This system helps in standardizing object representation across various
    observations. This system ensures uniform object representation across different
    viewpoints by keeping the object's coordinates constant, regardless of its movement
    or rotation relative to the observer.

**Visualizing coordinate systems:**

<p align="center">
<img alt="camera" src="/assets/images/2024-02-15/view_centric_object_centric1.png" width="100%">
<br> This visualization is adapted from "Sym3DNet: Symmetric 3D Prior Network for Single-View 3D Reconstruction."
</p>

The left image illustrates the view-aligned object-centric coordinate system (let's
pretend the coordinate system centered within a chair,) indicating how this system
adapts to the viewer's perspective.

The middle image depicts the object-centric canonical coordinate system, showcasing a
method where the object's orientation and position are standardized, irrespective of the
viewer's perspective.

The right image demonstrates that in the canonical view, all 3D shapes are uniformly
aligned within the world's 3D space, offering a consistent framework for object
representation.

**Visualizing ground truth w.r.t. the coordinate system:**

<p align="center">
<img alt="camera" src="/assets/images/2024-02-15/view_centric_object_centric2.png" width="100%">
<br> This visualization is adapted from "On the generalization of learning-based 3D reconstruction."
</p>

The image above shows that a view-aligned object-centric coordinate system dynamically
adjusts the ground truth coordinate system to match the orientation of the input view.
In contrast, an object-centric canonical coordinate system maintains the ground truth
anchored to a canonical frame, unaffected by the perspective of 2D input view.

Now, the goal is clearly defined: given an object image view from an arbitrary angle,
predict the object's 3D shape in the object-centric canonical coordinate system.
Predictions in the object-centric canonical coordinate system should be invariant to the
observed viewpoint, ensuring consistent and accurate 3D models across different
observations.

# Dataset Visualization

In this project, I use the `r2n2_shapenet_dataset`, a synthetic dataset designed in
structured environments. While synthetic data offers benefits such as ease of
prototyping and testing, it also comes with its challenges. One significant issue is
that synthetic data may favor artificial categories, leading to biases that do not
accurately represent the diversity and variability found in real-life objects and
environments.

For each training example, the dataset provides up to 24 views of a chair image, along
with mesh objects and voxels. It does not include point clouds; however, we can use
`sample_points_from_meshes()` to generate point clouds from the ground truth meshes.

Below are examples of multiple views from three different chairs:

<table>
  <tr>
    <td><img src="/assets/images/2024-02-15/dataset_visualization/groundtruth_image_2_view_1.png" width="100%"/></td>
    <td><img src="/assets/images/2024-02-15/dataset_visualization/groundtruth_image_2_view_2.png" width="100%"/></td>
    <td><img src="/assets/images/2024-02-15/dataset_visualization/groundtruth_image_2_view_3.png" width="100%"/></td>
    <td><img src="/assets/images/2024-02-15/dataset_visualization/groundtruth_mesh_2.gif" width="100%"/></td>
  </tr>
  <tr>
    <td><img src="/assets/images/2024-02-15/dataset_visualization/groundtruth_image_11_view_1.png" width="100%"/></td>
    <td><img src="/assets/images/2024-02-15/dataset_visualization/groundtruth_image_11_view_2.png" width="100%"/></td>
    <td><img src="/assets/images/2024-02-15/dataset_visualization/groundtruth_image_11_view_3.png" width="100%"/></td>
    <td><img src="/assets/images/2024-02-15/dataset_visualization/groundtruth_mesh_11.gif" width="100%"/></td>
  </tr>
  <tr>
    <td><img src="/assets/images/2024-02-15/dataset_visualization/groundtruth_image_14_view_1.png" width="100%"/></td>
    <td><img src="/assets/images/2024-02-15/dataset_visualization/groundtruth_image_14_view_2.png" width="100%"/></td>
    <td><img src="/assets/images/2024-02-15/dataset_visualization/groundtruth_image_14_view_3.png" width="100%"/></td>
    <td><img src="/assets/images/2024-02-15/dataset_visualization/groundtruth_mesh_14.gif" width="100%"/></td>
  </tr>
</table>

During the training phase, the data loader randomly selects one view for the input. An
important observation is that all meshes' vertices are positioned close to the origin,
with their centers very near to `(0, 0, 0)` and scales within `[-1, 1]`.

# Model building

now we are ready to build the model for processing point cloud.

## defining loss for the point clouds.

To calculate the loss between two pointclouds, the chamfer distance is usually used.
here is the equation. The Chamfer distance between point cloud is defined as:

$$
d_{\text{CD}}(S_1, S_2) = \frac{1}{|S_1|} \sum_{x \in S_1} \min_{y \in S_2} \|x - y\|^2 + \frac{1}{|S_2|} \sum_{y \in S_2} \min_{x \in S_1} \|x - y\|^2
$$

which is implemented as following with the help of `knn_points` from PyTorch3D.

```python
def chamfer_loss(point_cloud_src, point_cloud_tgt):
    # point_cloud_src, point_cloud_src: (batch, n_points, 3)

    k = 1  # the number of nearest neighbors
    # knn_points returns K-Nearest neighbors on point clouds.
    src_dists, _, _ = knn_points(point_cloud_src, point_cloud_tgt, K=k)
    tgt_dists, _, _ = knn_points(point_cloud_tgt, point_cloud_src, K=k)
    # src_dists, tgt_dists: (batch, n_points, k)

    return src_dists.mean() + tgt_dists.mean()  # Calculate the mean distance.
```

## fitting 3d point clouds with torch tensor.

one way to verify if the loss function is correct is that we can fit a random point
cloud with the model using our loss function. the simplified code looks like the
following:

```python
n_points = 10000
pointclouds_source = torch.randn([1, n_points, 3], requires_grad=True, device="cuda")
optimizer = torch.optim.Adam([pointclouds_source], lr=1e-4)

for step in range(0, 50000):
    loss = chamfer_loss(pointclouds_source, pointclouds_groundtruth)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

Here is how the output looks like:

<table>
  <tr>
    <td><img src="/assets/images/2024-02-15/dataset_visualization/pointcloud_gt_2.gif" width="100%"/></td>
    <td><img src="/assets/images/2024-02-15/dataset_visualization/pointcloud_fitted_2.gif" width="100%"/></td>
  </tr>
  <tr>
    <td><img src="/assets/images/2024-02-15/dataset_visualization/pointcloud_gt_11.gif" width="100%"/></td>
    <td><img src="/assets/images/2024-02-15/dataset_visualization/pointcloud_fitted_11.gif" width="100%"/></td>
  </tr>
  <tr>
    <td><img src="/assets/images/2024-02-15/dataset_visualization/pointcloud_gt_14.gif" width="100%"/></td>
    <td><img src="/assets/images/2024-02-15/dataset_visualization/pointcloud_fitted_14.gif" width="100%"/></td>
  </tr>
</table>

## Defining `PointModel`

`PointModel` inherits from `torch.nn.Module`. It builds the overall architecure of the
model. It consists of two parts, an encoder and a decoder:

1. **2D Encoder**: Transforms an image into a latent representation, capturing the
   essential features required for 3D reconstruction. I use resnet modoel from
   `torchvision.models`.
2. **3D Decoder**: Converts the latent representation into a discrete 3D point clouds,
   where each point represents the location in the 3d world.

here is what my model looks like:

```python
class PointModel(nn.Module):
    def __init__(self, arch, n_points):
        super(PointModel, self).__init__()
        self.device = args.device
        vision_model = torchvision_models.__dict__[arch](pretrained=True)
        self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        # Encoder output (batch, 512) if resnet18
        #
        # Input: b x 512
        # Output: b x n_points x 3

        self.n_point = n_points
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, n_points * 3),
            torch.nn.Tanh()
        )

    def forward(self, images, args):
        images_normalize = self.normalize(images.permute(0, 3, 1, 2))
        encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1)  # b x 512
        pointclouds_pred = self.decoder(encoded_feat)
        return pointclouds_pred.view(-1, self.n_point, 3)
```

# evaluation metric

given a predicted point cloud and a ground truth point cloud. how can i calculate the
evaluation metric. for example, how can i know the precision, recall and f1? my idea is
for each point in predicted point cloud, I first found the closest ground truth and
their distance, if this distance is smaller than a threshold, then I say this point in
predicted point cloud is a true positive. Does it make sense?

Your idea for calculating precision, recall, and F1 score for evaluating a predicted 3D
point cloud against a ground truth point cloud by using a distance threshold to
determine true positives makes sense and is a practical approach in the context of 3D
point clouds. This method is somewhat analogous to object detection metrics in 2D image
processing, where predictions are compared to ground truth data to determine true
positives, false positives, and false negatives. Here's how you can conceptualize and
implement these calculations:

### Definitions

-   **True Positive (TP):** A point in the predicted point cloud that is within a
    specified distance threshold of any point in the ground truth point cloud.
-   **False Positive (FP):** A point in the predicted point cloud that is not within the
    threshold distance of any point in the ground truth point cloud.
-   **False Negative (FN):** A point in the ground truth point cloud that is not within
    the threshold distance of any point in the predicted point cloud.

### Calculating Precision, Recall, and F1 Score

-   **Precision** measures the proportion of predicted points that are true positives
    out of all points predicted.
-   **Recall** measures the proportion of true positives out of the points in the ground
    truth point cloud.
-   **F1 Score** is the harmonic mean of precision and recall, providing a balance
    between them.

### Implementation Steps

1. **Determine TP, FP, and FN:**

    - For each point in the predicted point cloud, find the closest point in the ground
      truth point cloud. If the distance is less than the threshold, count it as TP;
      otherwise, it's FP.
    - For each point in the ground truth point cloud, find if there's a corresponding TP
      in the predicted point cloud (within the threshold). If not, it's an FN.

2. **Calculate Precision and Recall:**

    - Precision = TP / (TP + FP)
    - Recall = TP / (TP + FN)

3. **Calculate F1 Score:**
    - F1 Score = 2 _ (Precision _ Recall) / (Precision + Recall)

### Code Example

Here's a simplified example assuming you have distance functions and your point clouds
are in PyTorch tensors:

```python
import torch
from pytorch3d.ops import knn_points

def calculate_metrics(pred_cloud, gt_cloud, threshold=0.1):
    # Find closest points and distances from predicted to ground truth
    knn_result = knn_points(pred_cloud.unsqueeze(0), gt_cloud.unsqueeze(0), K=1)
    distances_pred_to_gt = knn_result.dists.squeeze(0)  # Batch size of 1
    tp_fp_mask = distances_pred_to_gt < threshold**2  # True for TP+FP, squared threshold for squared distances

    # True Positives + False Positives
    tp_plus_fp = tp_fp_mask.sum().item()

    # False Negatives
    knn_result_gt_to_pred = knn_points(gt_cloud.unsqueeze(0), pred_cloud.unsqueeze(0), K=1)
    distances_gt_to_pred = knn_result_gt_to_pred.dists.squeeze(0)
    fn_mask = distances_gt_to_pred >= threshold**2  # True for FN
    fn = fn_mask.sum().item()

    # True Positives
    tp = tp_plus_fp  # Assuming each TP in pred has a unique closest point in GT

    # Precision, Recall, and F1
    precision = tp / (tp_plus_fp) if tp_plus_fp > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score
```

### Notes

-   This approach assumes a one-to-one mapping for TP determination. Depending on your
    application, you might adjust how TPs, FPs, and FNs are counted, especially if the
    density of points or the distribution of points in space significantly differs
    between the predicted and ground truth point clouds.
-   The use of a squared threshold for distance comparisons is a minor optimization to
    avoid unnecessary square root calculations since `knn_points` returns squared
    distances by default. Adjust the threshold according to your specific needs and
    distance metric.

**references:**

sphere mesh to dolphin mesh
https://pytorch3d.org/tutorials/deform_source_mesh_to_target_mesh

-   "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network"
-   "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot
    Cross-dataset Transfer"
-   "Learning a predictable and generative vector representation for objects"
-   "Occupancy Networks: Learning 3D Reconstruction in Function Space"
-   "AtlasNet: A Papier-Mâché Approach to Learning 3D Surface Generation"
-   "Pixel2Mesh: Generating 3D Mesh Models from Single RGB Images"

be carefor of this imageio error:

```
TypeError: Cannot handle this data type: (1, 1, 3), <f4
```

https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/

<!--
Scale and Shift Invariant Objectives: As you correctly noted, depth estimation from a
single image suffers from scale and depth ambiguity. To mitigate this, researchers have
proposed scale-invariant loss functions for training deep learning models. These loss
functions are designed to minimize the relative error in depth prediction rather than
the absolute error, which helps the model learn to predict depth up to a scale factor.
This approach aligns with how humans perceive depth—relative to other objects rather
than in absolute terms.

Inverse Depth Representation: Some methods represent depth in terms of inverse depth
(disparity) rather than absolute depth. This representation naturally emphasizes closer
objects (which have higher disparity values) and can be more robust to the scale
ambiguity problem. Loss functions based on disparity can be designed to be scale and
shift invariant, improving the robustness of depth predictions.

Incorporating Additional Constraints: Other approaches to single-view depth estimation
incorporate additional constraints or information, such as known object sizes, geometric
models of the scene, or semantic information about the scene (e.g., sky is far, cars are
at a certain range on the road). These methods use a combination of cues and assumptions
to refine depth predictions. -->
