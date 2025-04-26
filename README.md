# Non linear systems in ML & Resorvoir Computing

This project, explores reservoir computing with a special focus on analyzing and using various non linear systems and hopefully able to create a general resorvoir for various different tasks.

## Overview

Reservoir computing is a framework for designing recurrent neural networks, particularly effective for time-series processing and complex dynamic systems. This study investigates:
- The principles behind Echo State Networks (ESN)
- Advanced learning strategies and online/offline training methods
- Exploration of hyper-parameters optimization

One of the best resource for overview on RC: [lit](papers/RC_Intro.pdf)

## Papers & Projects

Some literature:
- Many other in the [papers directory](papers/)
- Evolving Reservoirs for Meta Reinforcement Learning. (EvoAPPS 2024)  
    [HAL](https://inria.hal.science/hal-04354303) • [Code](https://github.com/corentinlger/ER-MRL)
- From Implicit Learning to Explicit Representations.  
    [arXiv](https://arxiv.org/abs/2204.02484) • [PDF](https://arxiv.org/pdf/2204.02484)


## Expt results

Below are some preliminary results of implementations and runs

**From [resorvoir_py.ipynb](resorvoir_py.ipynb):**

![Lorenz system evolution xt vs yt zt](results_0/lorenz%20system%20pred.png)

![Lorenz System evolution](results_0/lorenz%20attractor%20rc.png)


**From [simple_pendulum_RC.ipynb](simple_pendulum_RC.ipynb):**

![Simple Pendulum RC](results_0/simple%20pendulum%20track.png)


## Acknowledgment

This project is conducted under the guidance of Prof. Gaurav Dar as part of my Study Oriented Project. 