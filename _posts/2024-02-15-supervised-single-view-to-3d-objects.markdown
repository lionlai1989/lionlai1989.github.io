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
