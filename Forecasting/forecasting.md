<!-- omit in toc -->
Time Series, Forecasting, and Deep Learning Algorithms
==========

*Zijun Luo*

## Introduction

This chapter is structurally different from other previous chapters. In the next section, we will first look at a `python` implementation, that is, the implementation of *traditional* time-series/forecasting models with the library `statsmodels`. This serves both as a review for forecasting concepts and an introduction to yet another widely used python library.

We will then cover, briefly, how we can transform a forecasting problem into one of `machine learning`'s. Such transformation allows us to use regression and classification algorithms to tackle complicated forecasting tasks. We have already introduced **classification** algorithms in the last chapter. Machine learning **regression** algorithms, however, is deferred to the next chapter. However, readers with a quantitative should have no problem linking forecasting to regression models.

The main emphasis of this chapter is the use of `deep learning` models for forecasting tasks. For this, we will look at how different neural network models, including `Artificial Neural Networks` (ANN), `Convolutional Neural Networks` (CNN), and `Reccurent Neural Networks` (RNN) may be useful. We will implement some of these methods in Python using `PyTorch`. Finally, this chapter ends with the introduction to `Facebook`'s `Prophet` library, which is a widely-used library in the industry.

**Forecasting** should require no further introduction. At its simplest form, you have a time series data, which is data of the single value overtime, and you try to predict the "next" value into the future. In more complicated cases, you can have covariates/features, as long as these features do not result in the so-called "information leakage": at the time of your forecast, you should know and be certain about values of the features. For example, if you are doing weather forecast and your goal is to forecast whether it is going to rain tomorrow, then a time-series dataset would contain only information of whether it has rained or not for the past many days, whereas additional features such as temperature, dew point, and precipitation may be included. These additional weather variables should be from the day before your forecast, not the day of your forecast when you are training your model.

## Implementation in `statsmodels`

## Machine Learning Methods (Brief)

## Self-supervised Learning (?)

## Deep Learning Methods

## Long Short-Term Memory (LSTM)

## Facebook Prophet

## Summary