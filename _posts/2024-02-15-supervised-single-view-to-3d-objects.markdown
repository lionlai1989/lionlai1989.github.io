---
layout: single
title: Supervised Single-View to 3D Objects
categories: []
description: "."
toc: true
wip: true
date: 2024-02-15
---

3D reconstruction from a single view is very similar to the process through which we
recognize objects in the real world. When we look at a chair from one angle, we know it
is a chair and can intuitively _imagine_ what it would look like from other angles. It's
not like a chair viewed from one angle will look like an airplane from another angle.
That being said, if you were determined to design an airplane that looks like a chair
from a specific viewpoint, then everything in this post is inapplicable. 🤣

Work in progress.

$$
d_{\text{Chamfer}}(S_1, S_2) = \frac{1}{|S_1|} \sum_{x \in S_1} \min_{y \in S_2} \|x - y\|^2 + \frac{1}{|S_2|} \sum_{y \in S_2} \min_{x \in S_1} \|x - y\|^2
$$


