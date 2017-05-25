# Laplace approximation considered useful

The Laplace approximation is a useful trick that bridges the risk minimization and Bayesian views of machine learning. It unlocks some possibilities that I personally find interesting, and in this repo I want to go over some of them.

[1 Background](Background.ipynb) Deriving the Laplace approximation from scratch.

[2 Sequential Learning](Sequential%20Learning.ipynb) Often, data arrives in batches and we want to update our model incrementally, without re-training on the entire collected dataset - that simply can take too much time. Turns out, we can train pretty complex models incrementally, thanks to the Laplace approximation.

[4 Parallel Learning ]() [To be added] Alternatively, sometimes we want to split a large dataset in a set of smaller dataset and learn on them in parallel. How do the merge the solutions of these subproblems? Just average them? Let's try combining them in a smarter way.

[5 Multi-task Learning]() [To be added] In another scenario, we want to sequentially train our model to perform several different tasks. For instance, one day we train it to detect cats in images, the next day we want to train it to detect dogs and - the tricky part - not to forget how to detect cats!

[6 Thompson Sampling]() [To be added] Thompson sampling is an elegant explore/exploit strategy that works by sampling from the posterior. The Laplace approximation allows us to approximate the posterior in cases where it's hard to do it other way.


# References

[Pattern Recognition and Machine Learning, Bishop, C.M.; S4.4](https://www.microsoft.com/en-us/research/people/cmbishop/)

[Comment on "Overcoming catastrophic forgetting in NNs": Are multiple penalties needed? Ferenc Huszár](http://www.inference.vc/comment-on-overcoming-catastrophic-forgetting-in-nns-are-multiple-penalties-needed-2/)

[Overcoming catastrophic forgetting in neural networks, Kirkpatrick et al.](https://arxiv.org/abs/1612.00796)

[An Empirical Evaluation of Thompson Sampling, Chapelle O., Li L.](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/thompson.pdf)

[Information theory, inference and learning algorithms, D. MacKay](www.inference.phy.cam.ac.uk/itila/book.html)

[Thompson Sampling and Bayesian Factorization Machines, Berzan C.](http://tech.adroll.com/blog/data-science/2017/03/06/thompson-sampling-bayesian-factorization-machines.html)
