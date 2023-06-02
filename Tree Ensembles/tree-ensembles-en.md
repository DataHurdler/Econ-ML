<!-- omit in toc -->
Discrete Choice, Classification, and Tree-Based Ensemble Algorithms
=============================================================

*Zijun Luo*

## Introduction

Suppose you are the owner of an e-commerce website that sells a musical instrument in 5 sizes: soprano, alto, tenor, bass, and contrabass. You are now considering to open up some physical stores. With physical stores, you need to make more careful inventory decisions. Based on your past experience in shipping out your instruments, you are convinced that different communities can have different tastes toward different sizes. And you want to make inventory decisions accordingly.

If the stake for a musical store is too low, consider the classic discrete choice example: automobiles. Ultimately, we want to understand the question of "who buys what." This can inform many business decisions, not only inventory, since the aggregation of individuals can tell us the demand of a product in a market. Discrete choice is different from decisions about "how much," for example, do I exercise 15 minutes or 25 minutes tonight?

In economics and social sciences, the popular approaches are what I call a "top-down" approaches: it starts with a understanding of the data-generating process and some assumptions. Logit and Probit are two widely used models in economics. If the error term is believed to follow a logistic distribution, then use the logit model or logistic regression. If the error term is believed to follow a normal distribution, then use the probit model. If there is nested structure, then use nested-logit or nested-probit. And so on.

This is fine, since the focus of economics and social sciences is hypothesis testing and the understanding of mechanisms. On the contrary, the machine learning approach is more *bottom-up*: it cares about making good predictions. In a way, we can say that the economics approach of discrete choice cares more about "bias" whereas the machine learning approach considers the bias-variance tradeoff more holistically.

## The Bias-Variance Tradeoff

Let's begin with *Variance*. A model with high variance is sensitive to the *training* data and can capture the fine details in the training data. However, such model is usually difficult to generalize. On the one hand, the *test* data, or the data that the model is actually applied to, may lack such fine detail. On the other hand, those fine details may not be as important in the actual data than in the training data.

A model that can capture fine details is almost guaranteed to have low *bias*. A model with low bias is one that explains the known, or training, data well. In order to predict, we need our machine learning model to learn from known data. A model with high bias normally can not predict well.

While models with low bias *and* low variance do exist, they are rare. Since a model with high bias almost always does not work well, lowering bias is often considered a first-order task. One way to do so is using models, or specifying hyperparameters of a model, so that more fine details in the data are taken in to consideration. By doing so, higher variance is introduced. And hence the trade off.

Consider the following example: a zoo wants to build a machine learning algorithm to detect penguin species and deploy it on their smart phone application. Let's say all that the zoo and its users care about is to tell apart King, Magellanic, and Macaroni penguins. The zoo's staffs and data scientists took hundreds of photos of penguins in their aquarium, split the penguins into training and test datasets, as how tasks like are usually performed, and build the machine learning model. In their test, with photos that they have set aside earlier, they find that the algorithm is able to identify the penguins correctly 98%.

However, when their users use the algorithm to identify penguins in other zoos, the algorithm fails miserably. Why? It turns out that the machine learning algorithm was not learning to identify penguins by their different features such as head, neck, and tails. Instead, the algorithm identifies the different species of penguins by the tag on their wings: blue is for King penguin, red for Magellanic, and yellow for Macaroni. These are the colors used by the zoo who developed the algorithm, but different zoos have different tags. As a result, this algorithm, which has low bias but high variance, is unable to predict or detect the species of penguins outside of the zoo where photos for developing the machine learning algorithm were taken.

As we will see next, tree-based algorithms are extremely prone to high variance, or *over-fitting*.

## Decision Tree

Let's first talk about the basic decision tree algorithm. Because we will be using [scikit-learn](https://scikit-learn.org/stable/) for `Python` implementation in this chapter, I am using notations and languages similar to that in scikit-learn's documentations. Also, since this is not a comprehensive lecture on decision tree, as you can easily find more in-depth discussions in books such as [An Introduction to Statistical Learning](https://www.statlearning.com/) and [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/), or other online learning sources, my focus is what matters most for the applications of these algorithms in business and economics.

The simplest way to think about a decision tree algorithm is to consider a flow-chart, especially one that is for diagnostic purposes. Instead of someone building a flow-chart from intuition or experience, we feed data into the computer and the decision tree algorithm would build a flow-chart to explain the data. For example, if we know some characteristics of consumers of the music store, and want to know who is more likely to buy the smaller size instruments, a flow-chart, built by the decision tree algorithm, may look like this:
* Is the customer under 30?
  * Yes: is the customer female?
    * Yes: has the customer played any instrument before?
      * Yes: the customer has a 90% chance buying a smaller instruments
      * No: the customer has 15% chance buying a smaller instruments
    * No: is the customer married?
      * Yes: the customer has a 5% chance buying a smaller instruments
      * No: the customer has 92% chance buying a smaller instruments
  * No: is the customer under 50?
    * Yes: the customer has a 10% chance buying a smaller instruments
    * No: has the customer played any instrument before?
      * Yes: the customer has a 100% chance buying a smaller instruments
      * No: the customer has a 20% chance buying a smaller instruments

You can see several basic elements of a decision tree algorithm:
1. As expected, the tree algorithm resulted in a hierarchical structure that can be easily represented by a tree diagram;
2. The tree structure does not need to be symmetrical. For example, when the answer to "is the customer under 50" is a "yes", the branch stopped, resulted in a shorter branch compared to the rest of the tree;
3. You may use the same feature more than once. In this example, the question "has the customer played any instrument before" has appeared twice;
4. You can use both categorical and numerical features. In this example, age is numerical, whereas all other features are categorical;
5. It is accustomed to split to two branches at each node but no more. If you want three branches, you can do it at the next node: two branches first, then one of the next nodes splits into another 2 branches. Personally, I like keeping it that way. One exception may be at the end node, as there is no further split.

There are other elements of a decision tree algorithm that you can not observe directly from this example but are very important. We will examine these in more details below.

## Split Criterion

At each node, the split must be based on some criterion. The commonly used criteria are **Gini impurity** and **Entropy** (or **Log-loss**). According to the scikit-learn documentation, let

$$p_{mk}=\frac{1}{n_m}\sum_{y\in Q_m}{I(y=k)}$$

denote the proportion of class $k$ observations in node $m$, where $Q_m$ is the data at node $m$, $n_m$ is the sample size at node $m$, and $I(\cdot)$ returns 1 when $y=k$ and 0 otherwise. Then, the **Gini impurity** is given by:

$$H(Q_m)=\sum_{k}{p_{mk}(1-p_{mk})}$$

whereas **Entropy** is given by:

$$H(Q_m)=-\sum_{k}{p_{mk}\log{(p_{mk})}}$$

At each node $m$, a `candidate` is defined by the combination of feature and threshold. For example, in the above example, for the question "Is the customer under 30," the feature is age and the threshold is 30. Let $\theta$ denote a candidate, which splits $Q_m$ into two partitions: $Q_m^{\text{left}}$ and $Q_m^{\text{right}}$. Then the quality of a split with $\theta$ is computed as the weighted average of the criterion function $H(Q_m)$:

$$G(Q_m, \theta) = \frac{n_m^{\text{left}}}{n_m}H(Q_m^{\text{left}}(\theta)) + \frac{n_m^{\text{right}}}{n_m}H(Q_m^{\text{right}}(\theta))$$

The objective of the decision tree algorithm is to find the candidate that minimizes the quality at each $m$:

$$\theta^* = \argmin_{\theta}{G(Q_m, \theta)}$$

It is straightforward to see that, from either the **Gini impurity** or the **Entropy** criterion function, the unconstrained minimum of $G(Q_m, \theta)$ is achieved at $p_{mk}=0$ or $p_{mk}=1$, i.e., when the result of the split consists of a single class.

Before we move on, here is a quick remark: while there exists a global optimum for building a decision tree, where the quality function is minimized for the *whole* tree, the computation of such algorithm is too complex. As a result, practical decision tree algorithms resolve to using *local* optima at each node as described above.

## Pruning

If achieving a "pure" branch, where there remains observations from a single class after a split, minimizes the quality function $G(Q_m, \theta)$, then why did we not achieve that "pure" state in the illustrative music store example? There are two main reasons. First, we may not have enough features. Imagine you have two individuals in your data set, one bought a small instrument and the other bought a large instrument. These two individuals are almost identical: the only difference is in their eye colors. If "eye color" is not one of the features captured in your data set, you will have no way to distinguish these two individuals. On the other hand, imagine we know *everything* about each and every individual, then it is almost guaranteed that you can find a "perfect" tree, such that there is a single class of individuals at each end node. Such "perfect" tree may not be unique. At the extreme, imagine a tree such that each end node represents a single individual.

The second reason is something we have already covered: the Bias-Variance Tradeoff. Because the ultimate goal is to predict, fitting a "perfect" tree can result in too high of a variance. Continued with the previous example, your ability to build a perfect tree would totally depend on whether you have "eye color" as a feature in your data set. That means that your algorithm is too sensitive to one particular feature - if this feature does not exist, your algorithm would fail to build a "perfect" tree (assuming that was the goal). Or, if this feature is somehow absent or incorrect in the data set you are predicting on, your algorithm would have a breakdown.

This is why a decision tree needs to be pruned. This is often done by specifying two hyperparameters in the decision tree algorithm: the maximum depth of the tree (`max_depth`) and the minimum number of samples required to split (`min_samples_split`). Without going into the technical details, we can intuitively understand that both of these restrictions can prevent us from splitting the tree to the extreme case such that each end node represents an individual. In other words, they restrict the growth of a tree.

The caveat of a single decision tree algorithm is obvious: it can easily suffer from either high bias or high variance, especially the latter. This is why ensemble methods such as **bagging** and **boosting** were invented. In practice, a single decision tree is rarely used anymore, except as a demonstrative example.

## Bagging and Random Forest

`Bagging` is one of two ensemble methods based on the decision algorithm. `Bagging` is short for *boostrap aggregation*, which explains what bagging algorithms do: select random subsets from the training data set, fit the decision tree algorithm on each sample, and aggregate to get the final result. There are several variations of `Bagging` algorithms depending on how random samples are drawn:
1. When random subsets were drawn with replacement (bootstrap), the algorithm is known as `Bagging` (Breiman, 1996)
2. When random subsets were drawn without replacement, the algorithm is known as `Pasting` (Breiman, 1999)
3. When random subsets are drawn based on features rather than individuals, the algorithm is known as `Subspaces` (Ho, 1998)
4. When random subsets are drawn based on both features and individuals, the algorithm is known as `Random Patches` (Louppe and Geurts, 2012)
5. When random subsets were drawn with replacement (bootstrap) *and* at each split, a random subset of features is chosen, the algorithm is known as `Random Forest` (Breiman, 2001)

In scikit-learn, the first four algorithms can be implemented in `BaggingClassifier` whereas `Random Forest` is implemented in `RandomForestClassifier`.

In bagging algorithms, the "aggregation" of results during prediction is usually taken by votes. For example, suppose you have fit your data with `Random Forest` algorithm with 1,000 trees, and now you want to know whether a new customer is going to buy a small or a large instrument. When the algorithm considers the first split, it will look at all 1,000 trees and see which candidate was used the most. Suppose "Is the customer under 30" appeared in 800 of the trees, then the algorithm would split according to `age=30`. And so, at east split, the algorithm would take a tally from the 1,000 individual trees and act accordingly, just like how one would look at a flow-chart to determine their actions.

While a `Bagging` algorithm helps to reduce bias, the main benefit of bootstrapping is to reduce variance. The `Random Forest` algorithm, for example, is able to reduce variance in two ways: First, bootstrapping random samples is equivalent to consider many different scenarios. Not only does this mean that the algorithm is less reliant on a particular scenario (the whole training data set), it also makes it possible that one or some of the random scenarios may be similar to the "future," i.e., the environment that the algorithm needs to make prediction on. Second, by considering a random set of features at each split, the algorithm is less reliant on certain features, and is hence resilient to "future" cases where certain features may be missing or have errors.

## Boosting and AdaBoost

While the main benefit of `Bagging` is in reducing variance, the main benefit of `Boosting` is to reduce bias, while maintaining a reasonably low variance. Boosting is able to maintain a low variance because, like Bagging, it also fits many trees. Unlike Bagging, which builds the trees in parallel, Boosting builds them sequentially.

The basic idean of boosting is to have incremental (small/"weak") improvements from the previous model, which is why it is built sequentially. This idea can be applied to all types of algorithms. In the context of decision tree, a boosting algorithm can be demonstrated by the following pseudocode:

```
Step 1: Build a simple decision tree (weak learner)
Step 2: Loop:
            Minimize weighted error
```

Currently, there are three main types of tree-based boosting algorithms: `AdaBoost`, `Gradient Boosting`, and `XGBoost`. The different algorithms are different in how to *boost*, i.e., how to implement Step 2.

`AdaBoost` was introduced by Freund and Schapire (1995). It is short for *Ada*tive *Boost*ing. `AdaBoost` implement boosting by changing the weights of observations. That is, by making some observations/individuals more important than the other. In a training data set with $N$ individuals, the algorithm begins by weighting each individual the same: $1/N$. Then it fits a simple decision tree model and makes predictions. Inevitably, it makes better decision for some individuals than the other. The algorithm then increases the weight for individuals that it did not make correct/good predictions on in the first model. Effectively, this asks the next decision tree algorithm to focus more on these individuals that it has failed to understand in the first tree. And this process continues until a stopping rule is reached. Such stopping rule can be something like "**stop** when 98% of the cases are correctly predicted".

It is straightforward to see that a boosting algorithm lowers bias. But was it often able to main a low *variance* too? It was able to do because a boosting algorithm effectively builds different trees at each iteration. When making predictions, it takes a weighted average of the models. Some mathematical details may be helpful.

Let $w_{ij}$ denote the weight of individual $i$ in stage/iteration $j$. In the beginning of the algorithm, we have $w_{i1}=1/N$ for all $i$ where $N$ is the the total number of individuals. After the first weak tree is built, we can calculate the error/misclassification rate of stage $j$ as

$$e_j = \frac{\sum_{N}{w_{ij}\times I_{ij}(\text{incorrect})}}{\sum_{N}{w_{ij}}}$$

where $I_{ij}(\text{incorrect})$ equals 1 if the prediction for individual $i$ is incorrect in stage $j$ and 0 otherwise. We can then calculate the *stage value* of model $j$:

$$v_j = \frac{1}{2}\log\left(\frac{1-e_j}{e_j}\right)$$

The stage value is used both in updating $w_{ij+1}$, i.e., the weight of individual $i$ in the next stage, and in acting as the weight of model $j$ when prediction is computed. To update the weight for the next stage/model, we have

$$w_{ij+1} = w_{ij} \times \exp{(v_j \times I_{ij}(\hat{y}_{ij}=y_i))}$$

where $\hat{y}_{ij}$ is the prediction for individual $i$ in stage $j$ and $y_i$ is the true label for individual $i$. For binary classification, it is conventional to expression $\hat{y}_{ij}$ and $y_i$ as 1 and -1, so that the above equation can be simplified into

$$w_{ij+1} = w_{ij} \times \exp{(v_j \times \hat{y}_{ij}\times y_i)}$$

At each stage $j(>1)$, the `AdaBoost` algorithm aims to minimize $e_j$.

To compute the overall/final prediction, let $\hat{y}_{ij}$ denote the prediction of model/stage $j$ for individual $j$, then the predicted value is calculated by:

$$\hat{y}_{i} = \sum_{J}{\hat{y}_{ij} \times v_j}$$

where $J$ is the total number of stages.

## Gradient Boosting and XGBoost

`Gradient Boosting` (Friedman, 2001) is another approach to boost. Instead of updating the weight after each stage/model, Gradient Boosting aims to minimize a loss function, using method such as gradient decent. The default loss function in scikit-learn, which is also the most common in practice, is the binomial deviance:

$$L_j = -2\sum_{N}{y_i\log{(\hat{p}_{ij})} + (1-y_i)\log{(1-\hat{p}_{ij})}}$$

where $N$ is the number of individuals, $y_i$ is the true label for individual $i$, and $\hat{p}_{ij}$ is the predicted probability that individual $i$ at stage $j$ having a label of $y$, and is given by the softmax function (logistic) when log-loss is specified:

$$\hat{p}_{ij} = \frac{\exp{(F_j(x_i))}}{1+\exp{(F_j(x_i))}}$$

where $F_j(x_i)$ is a numerical predicted value for individual $i$ by regressor $F_j(x)$. Here, $F_j(x)$ is the aggregated regressor in stage $j$, which is given by

$$F_j(x) = F_{j-1}(x) + h_j(x)$$

where $h_j(x)$ is the weak learner/regressor at stage $j$ that minimizes $L_j$. Substituting $F_M(x)$, the final regressor, into the above formula for $\hat{p}_{ij}$ gives the final prediction of the Gradient Boosting model.

Finally, using first-order Taylor approximation, it can be shown that minimizing $L_j$ is approximately equivalent to predicting the negative gradient of the samples, where the negative gradient for individual $i$ is given by

$$-g_i = -\left[\frac{\partial l_{ij-1}}{\partial F_{j-1}(x_i)}\right]$$

where $l_{ij-1}$ is the term inside the summation in $L_j$ (but lagged one stage):

$$l_{ij-1} = y_i\log{(\hat{p}_{ij-1})} + (1-y_i)\log{(1-\hat{p}_{ij-1})}$$

In other words, while the basic decision tree algorithm aims to predict the true classes, usually represented by 0s and 1s, `Gradient Boosting` aims to predict a numerical value that is the gradient. This means that, at each stage, Gradient Boosting is a regression problem rather than a classification problem. Predicting the gradient allows the algorithm to utilize many well developed methods for such task, for example, the Nelder-Mead method or simple grid search.

The discussion above focused on binary classification, which requires a single tree to be built in each stage. In multiclass classification, $K$ trees would be built for $K$ classes. For example, if `Gradient Boosting` is used to identify the 26 English alphabets, 26 trees are built and fitted in each stage.

`XGBoost` was introduced by Tianqi Chen in 2014. It is short for "e*X*treme *G*radient *Boost*ing". Instead of gradient decent, `XGBoost` implements [Newton's Method](https://en.wikipedia.org/wiki/Newton%27s_method), which is computationally much more demanding than gradient decent and requires a second-order Taylor approximation (instead of first-order as in `Gradient Boosting`). Due to this, in addition to **Gradients**, `XGBoost` also calculates the **Hessians**, which are a set of second-order derivatives (whereas gradients are the first-order derivatives).

`Python` library `xgboost` implements `XGBoost` and can be easily integrated with `scikit-learn`, which we use to implement all other algorithms covered in this chapter.

## Python Implementation with scikit-learn

## Confusion Matrix

## Comparison the Algorithms

## Summary
Mention causal tree.

## References

* https://scikit-learn.org/stable/modules/tree.html#tree
* https://xgboost.readthedocs.io/en/stable/tutorials/model.html
* https://machinelearningmastery.com/boosting-and-adaboost-for-machine-learning/
* https://stats.stackexchange.com/questions/157870/scikit-binomial-deviance-loss-function
* https://www.ccs.neu.edu/home/vip/teach/MLcourse/4_boosting/slides/