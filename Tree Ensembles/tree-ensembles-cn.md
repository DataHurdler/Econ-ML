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

## 决策树

我们首先来讨论最基本的模型：决策树。因为我们在Python代码中会使用 [scikit-learn](https://scikit-learn.org/stable/)，所以我们的讨论会尽量保持与scikit-learn的使用说明一致。需要注意的是，这里对决策树的讨论只限于其中重要的或者与文章后面介绍的算法相关的内容。在网络上你可以找到很多对决策树更深入的介绍，譬如[An Introduction to Statistical Learning](https://www.statlearning.com/) 和 [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/)这两本书中的相关内容。

理解决策树最简单的方法是把它想象成一个流程图，特别是用于作诊断或判断的流程图。我们把很多数据提供给电脑，然后电脑通过决策树算法来构造一个流程图来解释数据。回到前面的例子，如果我们有很多关于顾客的数据，那么我们就有可能通过决策树算法来构造一个决策树，来判断一个顾客是会蓝色还是黄色的产品：

* 顾客有30岁吗?
  * 有：顾客是女性吗?
    * 是：顾客是回头客吗?
      * 是：这个顾客有90%概率会买黄色
      * 不是：这个顾客有15%概率会买黄色
    * 不是：顾客已婚吗?
      * 已婚：这个顾客有5%概率会买黄色
      * 未婚：这个顾客有92%概率会买黄色
  * 没有：顾客有50岁吗?
    * 有：这个顾客有10%概率会买黄色
    * 没有：顾客是回头客吗？
      * 是：这个顾客有100%概率会买黄色
      * 不是：这个顾客有20%概率会买黄色

从这个例子当中，我们可以看到决策树算法的几个基本特征：
1. 决策树算法的结果可以很容易地被展示为树状图；
2. 树状图并不需要是对称的。譬如，当顾客我们知道顾客有50岁时，那一个分支就直接结束了，导致这个分支相对短一些；
3. 同一个变量可以被使用多次。在这个例子当中，“顾客是回头客吗?”出现了两次；
4. 数值变量和分类变量都能用。在这里例子当中，年龄是数值变量，其他的都是分类变量；
5. 通常，每个节点只拆分成两个分支。有一个分类变量有三个分类，那么可以先分成两组，然后下一个分支再分两组。

决策树算法中还有一些重要的方面是这个例子中无法展现的。以下两节内容，我们就来了解一下

## 拆分标准

决策树算法的每一次拆分，都需要根据一定的标准来进行。这些标准当中最常用的两个是基尼系数（Gini impurity）和信息熵（Entropy）。我们用

$$p_{mk}=\frac{1}{n_m}\sum_{y\in Q_m}{I(y=k)}$$

来代表类型 $k$ 在节点 $m$ 所占比重，其中 $Q_m$ 是节点 $m$ 的所有数据，$n_m$ 是在节点 $m$ 的样本数。如果 $y=k$，那么$ I(\cdot)$ 取值为1，否则取值为0。那么，基尼系数的计算如下：

$$H(Q_m)=\sum_{k}{p_{mk}(1-p_{mk})}$$

而信息熵的计算则是：

$$H(Q_m)=-\sum_{k}{p_{mk}\log{(p_{mk})}}$$

在每一个节点 $m$，一个“候选方法（candidate）”被定义为变量和阈值的组合。譬如说，“顾客有30岁吗”这个候选方法中，变量是年龄而阈值是30。如果我们 $\theta$ 来代表候选方法，而这个方法把 $Q_m$ 拆分为两组：$Q_m^{\text{left}}$ 和 $Q_m^{\text{right}}$，那么，这个拆分的质量（quality）可以通过标准函数（criterion function）的加权平均来计算：

$$G(Q_m, \theta) = \frac{n_m^{\text{left}}}{n_m}H(Q_m^{\text{left}}(\theta)) + \frac{n_m^{\text{right}}}{n_m}H(Q_m^{\text{right}}(\theta))$$

决策树算法的目标就是寻找在每一个节点可以最小化上述质量函数（quality function）的候选方法：

$$\theta^* = \argmin_{\theta}{G(Q_m, \theta)}$$

我们很容易发现，无论我们用的是基尼系数还是信息熵来作为标准函数，在没有限制条件下，$G(Q_m, \theta)$ 的最小值将在$p_{mk}=0$ 或 $p_{mk}=1$ 时实现。换句话说，当拆分后只有一个分类时，质量函数的值会最小。

值得一提的是，决策树算法中中存在一个全局最优（global optimum），但是找到这个全局最优所需的计算量太大了。在实践当中，决策树算法找的是最小化每个节点质量函数的局部最优（local optima）。

## 修剪

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