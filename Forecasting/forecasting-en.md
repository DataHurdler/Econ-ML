<!-- omit in toc -->
Time Series, Forecasting, and Deep Learning Algorithms
==========

*Zijun Luo*

## Introduction

This chapter is structurally different from other chapters. In the next section, we will first look at Python implementations. We will implement time-series/forecasting models that are not based on machine learning algorithms mainly using the Python library `statsmodels`. This serves both as a review for forecasting concepts and an introduction to the `statsmodels` library, another widely used python library for statistical/data analysis.

We will then cover, briefly, how we can transform a forecasting problem into one of `machine learning`'s. Such transformation allows us to use regression and classification algorithms to tackle complicated forecasting tasks. We have already introduced **classification** algorithms in the last chapter. Machine learning **regression** algorithms, however, is deferred to the next chapter. But readers with a quantitative background should have no problem linking forecasting to regression models.

The main emphasis of this chapter is the use of `deep learning` models for forecasting tasks. For this, we will look at how different neural network models, including `Artificial Neural Networks` (ANN), `Convolutional Neural Networks` (CNN), and `Reccurent Neural Networks` (RNN) may be useful. We will implement some of these methods in Python using `TensorFlow`. Finally, this chapter ends with the introduction to `Facebook`'s `Prophet` library, which is a widely-used library in the industry.

**Forecasting** should require no further introduction. At its simplest form, you have a time series data set, which is contains values of a single object/individual overtime, and you try to predict the "next" value into the future. In more complicated cases, you can have covariates/features, as long as these features are observable at the moment of forecasting and do not result in the **information leakage**"**. For example, if you are doing weather forecast and your goal is to forecast whether it is going to rain tomorrow, then a time-series dataset would contain only information of whether it has rained or not for the past many days, whereas additional features such as temperature, dew point, and precipitation may be included. These additional weather variables should be from the day before your forecast, not the day of your forecast when you are training your model. A class example of information leakage happens when forecasting with moving average (MA) values. For example, if you are doing a 3-day MA, then the value of today requires the use of the value from tomorrow, which is only possible in historic data but not with real data.

## Implementation in `statsmodels`

## Machine Learning Methods (Brief)

## Self-supervised Learning (?)

## Deep Learning Methods

## Long Short-Term Memory (LSTM)

## Facebook Prophet

## Summary