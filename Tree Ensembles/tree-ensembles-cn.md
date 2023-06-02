<!-- omit in toc -->
离散选择，分类，和基于树模型的集成算法
================================

**作者：** *罗子俊*

## 引言

如果你是一个很成功的电商，现在考虑在全国几个大城市开实体店。你的电商经验告诉你，不同地区对产品的需求是不一样的。你希望能够有办法通过数据来了解不同的市场，因为对市场的了解将会影响到你的货存决策。顾客对你产品的购买，是一个“离散选择”问题：他们决定“买”还是“不买”。这个决策跟考虑“我今天晚上要锻炼多长时间”是不一样的。

在经济学和社会科学里，比较流行的是“自上而下”（top-down）的方法：首先我们需要了解数据是怎么来的，然后我们有一些假设。Logit和Probit是两个常用的离散选择模型。如果干扰项是逻辑分布，那么我们就用logit或者说逻辑回归。如果干扰项是正态分布，那么就是probit。其他各种不同的情形，也大多有相应的模型。这些方法的好处是可以做假设检验，并且有助于对机制的了解。这些都是经济学和社会科学所关心的。

与社会科学不同，机器学习更关心预测，所以机器学习的算法一般是“自下而上（bottom-up）的。我们可以说，经济学的离散选择模型更注重偏差（bias），但是机器学习的方法在考虑偏差和方差（variance）时更全面。

## 偏差-方差间权衡（The Bias Variance Tradeoff）

我们先说*方差（Variance）*。一个模型如果方差比较高，那么就意味着它对训练数据集（training data）非常敏感，从而能够捕抓到训练数据当中的很多细节。可是，这样的模型是很难被一般化的。一方面，实际用于预测的数据未必跟训练数据集一样详尽。另外一方面，在训练数据集中重要的变量，在以后得数据里，未必重要。

一个可以捕抓到数据中细节的模型，通常都有低偏差（bias）。我们的算法要从训练数据集中学习，如果它的偏差比较高，那么它能够准确预测的可能性也很低。所以，虽然我们很难见到偏差和方差都低的模型，我们常常把降低偏差放在更重要的位置。一个降低方差的办法，就是通过修改模型的参数，使得训练数据集中更多的细节能被模型捕抓到。但是这么做，就更让方差变高。这也就是为什么这二者之间存在权衡的原因。

我们来考虑一个例子。假设一个动物园希望训练一个机器学习模型来区分不同的企鹅的品种。为了这个任务，动物园的工作人员和数据科学家给动物园里的企鹅拍了很多照片，把这些照片用来训练和测试模型。他们发现，算法判断企鹅种类的准确率高达98%。

可是，当动物园的游客在其他水族馆里使用这个算法时，发现它根本没有办法准确判断企鹅种类。为什么呢？原来这个算法并没有根据企鹅的一些特征，譬如他们的头，脖子，和尾巴，来判断企鹅的种类。动物园的算法是通过企鹅身上的标记来判断的。在这个动物园里，不同的企鹅，是用不同颜色来标记的。但是其他的水族馆，用的是其他的标记方式。所以，这个算法，虽然它在训练时偏差很低，但是它的方差却很高，因为它完全是基于这一个动物园对企鹅所作的标记来进行预测的。

下一篇文章，我们会讨论简单的决策树算法。我们会发现，决策树算法很容易出现高方差，或者说过度拟合（over-fitting）。

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
Step 2: Loop until stopping rule has reached:
            Try to improve from model in the previous iteration
```

Currently, there are three main types of tree-based boosting algorithms: `AdaBoost`, `Gradient Boosting`, and `XGBoost`. The different algorithms are different in how to *boost*, i.e., how to implement Step 2.

`AdaBoost` was introduced by Freund and Schapire (1995). It is short for *Ada*tive *Boost*ing. `AdaBoost` implement boosting by changing the weights of observations. That is, by making some observations/individuals more important than the other. In a training data set with $N$ individuals, the algorithm begins by weighting each individual the same: $1/N$. Then it fits a simple decision tree model and makes predictions. Inevitably, it makes better decision for some individuals than the other. The algorithm then increases the weight for individuals that it did not make correct/good predictions on in the first model. Effectively, this asks the next decision tree algorithm to focus more on these individuals that it has failed to understand in the first tree. And this process continues until a stopping rule is reached. Such stopping rule can be something like "**stop** when 98% of the cases are correctly predicted".

It is straightforward to see that a boosting algorithm lowers bias. But was it often able to main a low *variance* too? It was able to do because a boosting algorithm effectively builds different trees at each iteration. When making predictions, it takes a weighted average of the models. Some mathematical details may be helpful.

Let $w_{ij}$ denote the weight of individual $i$ in stage/iteration $j$. In the beginning of the algorithm, we have $w_{i1}=1/N$ for all $i$ where $N$ is the the total number of individuals. After the first weak tree is built, we can calculate the error/misclassification rate of stage $j$ as

$$e_j = \frac{\sum_{N}{w_{ij}\times I_{ij}(\text{correct})}}{\sum_{N}{w_{ij}}}$$

where $I_{ij}(\text{correct})$ equals 1 if the prediction for individual $i$ is correct in stage $j$ and 0 otherwise. We can then calculate the *stage value* of model $j$:

$$v_j = \log\left(\frac{1-e_j}{e_j}\right)$$

The stage value is used both in updating $w_{ij+1}$, i.e., the weight of individual $i$ in the next stage, and in acting as the weight of model $j$ when prediction is computed. To update the weight for the next stage/model, we have

$$w_{ij+1} = w_{ij} \times \exp{(v_j \times I_{ij}(\text{correct}))}$$

To compute the prediction, let $\hat{y}_{ij}$ denote the predict of model/stage $j$ for individual $j$, then the predicted value is calculated by:

$$\hat{y}_{i} = \sum_{J}{\hat{y}_{ij} \times v_j}$$

where $J$ is the total number of stages.

## Gradient Boosting and XGBoost

`Gradient Boosting` (Friedman, 2001) is another approach to boost. Instead of updating the weight after each stage/model, Gradient Boosting aims to minimize a loss function, using method such as gradient decent. The default loss function in scikit-learn, which is also the most common in practice, is the binomial deviance:

$$LL = -2\sum_{N}{y_i\log{(\hat{p}_{ij})} + (1-y_i)\log{(1-\hat{p}_{ij})}}$$

where $N$ is the number of individuals, $y_i$ is the true label for individual $i$, and $\hat{p}_{ij}$ is the predicted probability that individual $i$ at stage $j$ having a label of $y$.

`XGBoost` was introduced by Tianqi Chen in 2014. It is short for "e*X*treme *G*radient *Boost*ing"

## Python Implementation with scikit-learn

## Confusion Matrix

## Comparison the Algorithms

## Summary
Mention causal tree.

## References

* https://scikit-learn.org/stable/modules/tree.html#tree
* https://xgboost.readthedocs.io/en/stable/tutorials/model.html
* https://www.nvidia.com/en-us/glossary/data-science/xgboost/
* https://machinelearningmastery.com/boosting-and-adaboost-for-machine-learning/
* https://stats.stackexchange.com/questions/157870/scikit-binomial-deviance-loss-function
* https://www.ccs.neu.edu/home/vip/teach/MLcourse/4_boosting/slides/