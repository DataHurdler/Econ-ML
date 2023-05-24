<!-- omit in toc -->
Discrete Choice, Classification, and Tree-Based Ensemble Algorithms
=============================================================

*Zijun Luo*

## Introduction

Suppose you are the owner of an e-commerce website that sells a musical instrument in 5 sizes: soprano, alto, tenor, bass, and contrabass. You are now considering to open up some physical stores. With physical stores, inventory is an important consideration. Based on your past experience shipping out your instruments, you feel like different communities can have different communities toward different sizes, and you wonder if you could make inventory decision accordingly.

If the stake for a musical store is too low, consider the classic discrete choice example: automobiles. Ultimately, we want to understand the question of "who chooses what?" This can inform many business decisions, especially inventory, since the aggregation of individuals can tell us the demand of a market. Discrete choice is different from choices in which you would need to make decisions about "how much," for example, do I exercise 15 minutes or 25 minutes tonight?

In economics and social sciences, the popular approaches are what I call a "top-down" approach: it starts with a understanding of the data-generating process and some assumptions. If the error term is believed to follow a logistic distribution, then use logit. If the error term is believed to follow a normal distribution, then use probit. If there is nested structure, then use nested-logit or nested-probit. And so on.

This is fine, since the focus of economics and social sciences is hypothesis testing and understanding the mechanism. On the other hand, the machine learning approach is more bottom-up and pragmatic: Can you make a good prediction. In a way, we can say that the economics approach of discrete choice cares more about "bias" whereas the machine learning approach considers the bias-variance tradeoff more holistically.

## The Bias-Variance Tradeoff

## Decision Tree

## Bagging and Random Forest

## AdaBoost

## Gradient Boosting

## XGBoost

## Comparison the Algorithms

## Causal Tree

## Summary

## References