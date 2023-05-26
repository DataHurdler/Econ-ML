<!-- omit in toc -->
Discrete Choice, Classification, and Tree-Based Ensemble Algorithms
=============================================================

*Zijun Luo*

## Introduction

Suppose you are the owner of an e-commerce website that sells a musical instrument in 5 sizes: soprano, alto, tenor, bass, and contrabass. You are now considering to open up some physical stores. With physical stores, inventory is an important consideration. Based on your past experience in shipping out your instruments, you are convinced that different communities can have different tastes toward different sizes. And you want to make inventory decision accordingly.

If the stake for a musical store is too low, consider the classic discrete choice example: automobiles. Ultimately, we want to understand the question of "who buys what?" This can inform many business decisions, especially inventory, since the aggregation of individuals can tell us the demand of a market. Discrete choice is different from decisions about "how much," for example, do I exercise 15 minutes or 25 minutes tonight?

In economics and social sciences, the popular approaches are what I call a "top-down" approaches: it starts with a understanding of the data-generating process and some assumptions. If the error term is believed to follow a logistic distribution, then use the logit model or logistic regression. If the error term is believed to follow a normal distribution, then use the probit model. If there is nested structure, then use nested-logit or nested-probit. And so on.

This is fine, since the focus of economics and social sciences is hypothesis testing and understanding the mechanism. On the other hand, the machine learning approach is more bottom-up: all it cares about is to make good predictions. In a way, we can say that the economics approach of discrete choice cares more about "bias" whereas the machine learning approach considers the bias-variance tradeoff more holistically.

## The Bias-Variance Tradeoff

Let's begin with *Variance*. A model with high variance is sensitive to the *training* data and can capture the fine details in the training data. However, such model is usually difficult to generalize. On the one hand, the *test* data, or the data that the model is actually applied to, may lack such fine detail. On the other hand, those fine details may only be important in the actual data.

A model that can capture fine details is almost guaranteed to have low *bias*. A model with low bias is one that explains the known, or training, data well. In order to predict, we need our machine learning model to learn from known data. A model with high bias usually does not predict well.

While models with low bias *and* low variance do exist, they are rare. Since a model with high bias almost always does not work well, lowering bias is often considered a first-order task. One way to do so is using models, or specifying hyperparameters of a model, so that more fine details in the data are taken in to consideration. By doing so, higher variance is introduced. And hence the trade off.

Consider the following example: a zoo wants to build a machine learning algorithm to detect penguin species and deploy it on their smart phone application. Let's say all that the zoo and its users care about is to tell apart King, Magellanic, and Macaroni penguins. The zoo's staffs and data scientists took hundreds of photos of penguins in their aquarium, split the penguins into training and test datasets, and build the machine learning model. In their test, they find that the algorithm is able to identify the penguins correctly 98%.

However, when their users use the algorithm to identify penguins in other zoos, the algorithm fails miserably. Why? It turns out that the machine learning algorithm was not learning to identify penguins by their different features such as head, neck, and tails. Instead, the algorithm identifies the different species of penguins by the tag on their wings: blue is for King penguin, red of Magellanic, and yellow for Macaroni. Different zoos have different tags, so this algorithm, which has low bias but high variance, is unable to predict or detect the species of penguins outside of the zoo where photos were taken.

As we will see next, tree-based algorithms are extremely prone to high variance, or *over-fitting*.

## Decision Tree

Let's first learn about the basic decision tree algorithm.

## Bagging

## Random Forest

## Boosting

## AdaBoost

## Gradient Boosting

## XGBoost

## Comparison the Algorithms

## Causal Tree

## Summary

## References

* https://scikit-learn.org/stable/modules/tree.html#tree
* https://xgboost.readthedocs.io/en/stable/tutorials/model.html
* https://www.nvidia.com/en-us/glossary/data-science/xgboost/