# Laplace approximation considered useful

Laplace approximation is a useful trick that bridges the loss-function-optimization and Bayesian views of machine learning. That unlocks some possibilities that I personally find interesting, and in this repo I want to go over some of them.

[1 Background](Background.ipynb) Deriving Laplace approximation from scratch

[2 Sequential Learning](Sequential%20Learning.ipynb) Often, data arrives in batches and we want to update our model incrementally, without re-training on the entire collected dataset - that simply can take too much time. Turns out, we can train pretty complex models incrementally, thanks to Laplace approximation.

[4 Parallel Learning]() Alternatively, sometimes we want to split a large dataset in a set of smaller dataset and learn on them in parallel. How do the merge the solutions of these subproblems? Just average them? Let's try combining them using the Laplace approximation!

[5 Multi-task Learning]() In another scenario, we want to sequentially train our model to perform several different tasks. For instance, one day we train it to detect cats in images, the next day we want to train it to detect dogs and - the tricky part - not to forget how to detect cats!

[6 Thompson Sampling]() Thompson sampling is an elegant explore/exploit strategy that works by sampling from the posterior.  Laplace approximation allows us to approximate the posterior in cases where it's hard to do it other way.


#References

