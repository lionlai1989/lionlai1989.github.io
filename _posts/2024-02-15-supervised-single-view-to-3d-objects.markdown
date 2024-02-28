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

Furthermore, we can train model to recognize patterns and learn these visual cues from
extensive datasets where the true depth or shape information is known.

In this post, I will show how numerous images and objects' 3D structures can be used to
train a neural network to deduce a 3D shape from a single 2D image.
