# Terms

## Week 2

- **Dropout**
    - A regularization technique. Involves randomly "hiding" a predefined
      proportion of nodes to be 0 at each iteration of the gradient descent.
      Note that the cost function may not always go down at each iteration. In
      prediction, all the nodes are turned on. 
- **Exponentially weighted average**
    - moving average. $v_t = p * v_{t-1} + (1-p) x_t$, where $x_t$ is the
      variable of interest. Higher $p$ means more weight to previous times.
      e.g. $p=2$ means roughly average over last two days.
- **Gradient descent with Momentum**
    - In gradient descent, instead of directly using the gradient to 
      update the state of the parameter, update the parameter
      using a moving average of the gradient. This smooths out the
      change in parameters per iteration, you will probably converge 
      faster.
- **RMSprop** 
    - Allows you to update proportional to the gradient with a larger learning
      rate. Basically like using different learning rates for each parameter.
- **Adam (adaptive momentum) Optimization**
    - Combines gd with momentum and RMSprop.
- **Learning rate decay**
    - decrease learning rate over time (not among the first things to try).
- **Local Optima in NN?**
    - Unlikely to get stuck in local optima in high dimensions 
      (most gradient=0 points are **saddle points**)
    - But there may be long plateaus, making learning slower
    - Adam may help

## Week 3

- ** Batch normalization**
    - Standardizing the hidden units. This speeds up the optimization process.
      Usually used in mini-batch gradient descent.
    - At testing, use moving average to get mean and var for centering and
      scaling.
