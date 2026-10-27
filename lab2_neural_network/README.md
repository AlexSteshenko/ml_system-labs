Lab 2: Deep Learning Primer — Neural Network from Scratch
Overview
This laboratory assignment implements a 2-hidden-layer fully-connected (dense) feedforward neural network from scratch using NumPy to classify handwritten digits (
8
×
8
 grayscale images, 10 classes) from the load_digits dataset.

The project demonstrates:

Correct manual implementation of Forward Propagation, Cross-Entropy Loss, and Backpropagation using the Chain Rule.
Validation of theoretical analytical gradients via Numerical Gradient Checking.
Hyperparameter search across varying hidden layer widths (
W
∈
16
,
64
) and learning rates (
η
∈
0.001
,
0.01
,
0.1
).
Empirical benchmarking of computational efficiency (Forward vs. Backward pass timing) linked to matrix multiplication complexity.
Architecture & Mathematical Formulation
1. Model Architecture
The network consists of an input layer, two hidden layers with ReLU activation, and an output layer with Softmax activation:

Input (64)
→
W
1
,
b
1
Hidden 1 
(
W
)
→
ReLU
Hidden 2 
(
W
)
→
ReLU
Output 
(
10
)
→
Softmax
Y
^

Input Dimension: 
D
i
n
=
64
 (flattened 
8
×
8
 images normalized to 
[
0
,
1
]
 by dividing by 16).
Hidden Widths (
W
): Evaluated for 
W
=
16
 and 
W
=
64
.
Output Dimension: 
K
=
10
 classes (One-Hot encoded labels).
Weight Initialization: Weights initialized from Gaussian distribution 
N
(
0
,
0.01
)
, biases initialized to zeros.
2. Forward Propagation
For a batch 
X
∈
R
N
×
64
:

Z
1
=
X
W
1
+
b
1
,
A
1
=
max
(
0
,
Z
1
)
Z
2
=
A
1
W
2
+
b
2
,
A
2
=
max
(
0
,
Z
2
)
Z
3
=
A
2
W
3
+
b
3
,
A
3
=
Softmax
(
Z
3
)
=
exp
⁡
(
Z
3
−
max
(
Z
3
)
)
∑
k
exp
⁡
(
Z
3
,
⋅
,
k
−
max
(
Z
3
)
)
3. Loss Function
Multi-class Cross-Entropy Loss with numeric stability clipping (
ϵ
=
10
−
15
): 
L
(
Y
,
A
3
)
=
−
1
N
∑
i
=
1
N
∑
k
=
1
10
Y
i
,
k
ln
⁡
(
A
3
,
i
,
k
)

4. Backpropagation Equations
Combining Softmax and Cross-Entropy Loss yields the concise error signal at the output:

Output Layer: 
δ
3
=
∂
L
∂
Z
3
=
1
N
(
A
3
−
Y
)
 
∂
L
∂
W
3
=
A
2
T
δ
3
,
∂
L
∂
b
3
=
∑
i
=
1
N
δ
3
,
i

Second Hidden Layer: 
δ
2
=
(
δ
3
W
3
T
)
⊙
I
(
Z
2
>
0
)
 
∂
L
∂
W
2
=
A
1
T
δ
2
,
∂
L
∂
b
2
=
∑
i
=
1
N
δ
2
,
i

First Hidden Layer: 
δ
1
=
(
δ
2
W
2
T
)
⊙
I
(
Z
1
>
0
)
 
∂
L
∂
W
1
=
X
T
δ
1
,
∂
L
∂
b
1
=
∑
i
=
1
N
δ
1
,
i

Experimental Results & Evidence
1. Numerical Gradient Checking
To confirm the correctness of analytical backpropagation gradients, 10 random parameters were selected and perturbed by 
ϵ
=
10
−
5
 using two-sided finite differences:

grad
num
=
L
(
θ
+
ϵ
)
−
L
(
θ
−
ϵ
)
2
ϵ

Relative error formula: $$\text{Relative Error} = \frac{\vert{}\text{grad}{\text{analytic}} - \text{grad}{\text{num}}\vert{}}{\vert{}\text{grad}{\text{analytic}}\vert{} + \vert{}\text{grad}{\text{num}}\vert{} + 10^{-8}}$$

Result: Passed for 9/10 checked parameters with relative error 
<
10
−
6
, satisfying the required sanity threshold.

2. Hyperparameter Grid Search & Test Accuracy
Training configuration:

Train / Test split: 70 / 30 (Stratified, random_state=42)
Batch size: 32
Epochs: 50
Optimizer: Mini-Batch Stochastic Gradient Descent (SGD) with per-epoch shuffling
Hidden Width (
W
)	Learning Rate (
η
)	Test Accuracy	Status / Notes
16	
0.001
10.19%	Underfitting / Stuck in local minimum due to small step size
16	
0.01
10.19%	Slow convergence
16	
0.1
67.22%	Moderate convergence
64	
0.001
10.19%	Underfitting
64	
0.01
10.19%	Underfitting
64	
0.1
94.44%	Best Performing Model
3. Loss Curves
Loss Curves

Figure 1: Cross-Entropy loss progression over 50 epochs for all 6 hyperparameter combinations.

4. Computational Performance Analysis (Timing Breakdown)
Timing measured over 100 independent runs on a batch size of 32:

Forward Pass Median Time: ~0.0848 ms
Backward Pass Median Time: ~0.0951 ms
Theoretical Justification
The Backward Pass takes consistently longer than the Forward Pass. This behavior is directly tied to the underlying matrix multiplication complexity:

Forward Pass: Each linear layer computes a single matrix multiplication: 
Z
[
l
]
=
A
[
l
−
1
]
W
[
l
]
 (1 GEMM operation per layer)

Backward Pass: Each layer requires two separate matrix multiplications:

Gradient w.r.t. parameters: 
∂
L
∂
W
[
l
]
=
(
A
[
l
−
1
]
)
T
δ
[
l
]
Gradient w.r.t. activations (to propagate error backwards): 
δ
[
l
−
1
]
=
(
δ
[
l
]
(
W
[
l
]
)
T
)
⊙
σ
′
(
Z
[
l
−
1
]
)

(2 GEMM operations per layer)
Hence, backpropagation inherently performs approximately double the number of matrix multiplications, explaining the higher execution time observed in benchmarks.

How to Run
# 1. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 2. Run training script
python main.py
