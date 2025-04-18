"""
STAGE 1: JUST A GRAPH OF VFE
- Accuracy is way bigger than complexity, at least in the linear case

STAGE 2: LEARNING (second graph)
- Correct learning of p(y|x) hinges on p(x) being reasonably close to the mean. Presumably, vice versa too.
- Kind of obvious, but in b1x + b0, b1 will be adjusted more than b0 if |x| is usually more than 1
- Learning in general seems unstable. It is preferred to bootstrap the Agent with reasonable p(y|x) and p(x).
-- What really happens is that p(y|x) and p(x) are trained against each other. y is just training data, and q(x) is a glue that transfers signal from p(x) to p(y|x). "From my p(y|x) I'm getting something unusual, so it's either not unusual or I don't understand how the world works. Let me do both"
-- Which is why it's vital to have optimally trained q(x) before even attempting to learn. Else, noise of q(x) will kill the signal from p(x) / p(y|x)

- The lower variance on p(x), the better learning for p(y|x) will be. Presumably, vice verse


Quadratic:
- Unstable gradients
- Local minima
- Learning rate matters a lot
- It might make much more sense to set learning rate to 1/2 and then do it twice, or a geometrically decaying learning rate
- Accuracy is a quartic (it was a quadratic for linear)

-Now:
-- Migrate from tensorflow
-- Finish up the cleaning
-- Get to a maze


"""
