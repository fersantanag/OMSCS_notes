![image](https://github.com/fersantanag/OMSCS_notes/assets/59957385/0f464cc9-bc2a-4c7f-8353-65fd8cc42f9f)# Lesson 1

In this lesson we will learn about reading and plotting stock data with Python. Why use Python? Python is an optimal tool because we can quickly prototype algorithms with high computational speed. Some features of Python include:

1. Strong scientific libraries
2. Strongly maintained
3. Fast

## Data In CSV Files

**CSV (Comma Separated Values) files** contain data typically formatted in rows and columns. We typically denote the top row for headers and the rest of the rows as data under each header.

## Real Stock Data

A _closing_ price on a stock may differ greatly from _adjusted closing_ price (since adjusted closing price accounts for splits, dividends, etc.).

## Pandas Dataframe

**Pandas** is a Python data analysis library created by Wes McKinney at a hedge fund called AQR Capital Management out of the need for a high performance, flexible tool to perform quantitative analysis on financial data.

A **dataframe** consists of a two dimensional chart consisting of stocks, prices, times, etc. There may also be additional layers to the dataframe such that different stock attributes may be evaluated (e.g., adjusted close, close, volume, etc.).

## Section Quizzes

### Which Fields Should Be In A CSV File Quiz

_Which fields would you expect to see in a csv file of stock data_?

- Date and time
- Price of the stock


# Lesson 2

In this lesson we will cover how to properly load data into a pandas dataframe such that we can visualize what we want.

## Problems To Solve

What are some of the basic problems we want to solve with the pandas dataframe?

1. How to look at specific date ranges?
2. How to look at multiple stocks?
3. How to align equity and dates?
4. How to order dates?

## Building A Dataframe

The first step to building a dataframe is loading the range of dates we are interested in. We then load in equity and dates for SPY ETF (SPDR S&P 500 Exchange Traded Fund) since we know that SPY does not trade on weekends, we can use SPY for reference.

## Obtaining A Slice Of Data

To obtain a slice of data, we can specify the following in a dataframe:

```python
df[startDate:endDate, ['A', 'B']]
```

Which is not limited to just certain rows or columns. We may decide to slice entire rows and copy onto another dataframe if needed.

## Problems With Plotting

Although we can just use `plot()` to plot dataframes, how do we go about comparing stocks such that data is normalized?

## Section Quizzes

### NYSE Trading Days Quiz

_How many days were U.S. stocks traded at NYSE in 2014_? 252.

### Types Of `join()` Quiz

`df1.join(dfSPY, how='inner')`, where inner (from the docs, [pandas.DataFrame.join](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.join.html)):

> forms intersection of calling frame’s index (or column if on is specified) with other’s index, preserving the order of the calling’s one.

### How To Plot On Equal Footing Quiz

_What is the best way to normalize price data so that all prices start at 1.0 (normalize to one)_? B, `df = df / df.ix[0, :]`.


# Lesson 3

**NumPy** is a numerical library that acts as a wrapper around underlying C and Fortran code (consequently, the runtime is optimal). NumPy's claim to fame is having ability to quickly vectorize, index, and broadcast n-dimensional arrays. It is a primary reason why many researchers decide to use Python for financial research.

## Relationship To Pandas

We previously discussed that NumPy is a wrapper around C and Fortran. Pandas is a wrapper around NumPy. This could be observed as the data in the dataframe is really an n-dimensional (`ndarray`) array surrounded by stock symbols (first row) and dates (first column).

When we treat a dataframe as an `ndarray`, we get many more methods which will enable us to do more with given data.

## Notes On Notation

To utilize `ndarray` we may specify certain values with:

```python
ndarray[row index, col index]
```

You may be familiar with accessing ranges from previous lessons. We may access ranges of values by specifying the starting indices (row or column) and ending indices (row or column) separated by a `:` to denote a range:

```python
ndarray[starting row index:ending row index, starting col index:ending col index]
```

## Section Quizzes

### Replace A Slice Quiz

_Which statement does the job_? The line:

```python
 nd1[0:2, 0:2] = nd2[-2:, 2:4]
```

### Specify The Datatype Quiz

_Specify the datatype for the given code (provided)_. `dtype = int`.


# Lesson 4

In this lesson we will examine various kinds of statistics that can help us work with time series data and extract value.

## Global Statistics

**Global statistics** allow us to look at the overall attributes of our dataframe. Some commonly used global statistics include:

- Mean
- Median
- STD (standard deviation)
- Sum
- Prod (product of values over request axis)
- Mode

For more global statistics, see the [dataframe docs](https://pandas.pydata.org/pandas-docs/stable/reference/frame.html).

## Rolling statistics

**Rolling statistics** comes in handy when working with a window of data (smaller sample of the overall data). Some commonly used rolling statistics include:

- [Rolling mean](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html)
- [Rolling STD](https://pandas.pydata.org/docs/reference/api/pandas.core.window.rolling.Rolling.std.html)

For more examples, see the page on pandas [computational tools](https://pandas.pydata.org/pandas-docs/stable/user_guide/computation.html).

## Bollinger Bands

**Bollinger Bands** are:

> Envelopes plotted at a standard deviation level above and below a simple moving average of the price. Because the distance of the bands is based on standard deviation, they adjust to volatility swings in the underlying price.

Bollinger Bands uses two primary parameters:

- Period
- STD

## Daily Returns

**Daily returns** refers the increase or decrease of the price of a stock on a particular day. Daily return may be calculated by using the following formula:

```python
daily_returns[t] = (price[t] / price[t - 1]) - 1
```

Where `t` refers to the specific day.

## Cumulative Returns

**Cumulative returns** refers to the total returns up to current day (or latest date available). Cumulative returns may be calculated by using the following formula:

```python
cumulative_returns[t] = (price[t] / price[0]) - 1
```

Where `t` refers to the current day.

## Section Quizzes

### Which Statistic To Use Quiz

_Which statistic_? Rolling standard deviation.


# Lesson 5

Data is a very important part of finance and accounting and can often be incomplete. In this lesson we will look into incomplete data and remedies around this issue.

## Pristine Data

When it comes to financial data, most people believe e.g.:

- Data is perfectly recorded minute by minute
- There are no gaps or missing data points

However, in reality data is aggregated from many different sources. Not all stocks trade when looking at specific times.

## Why Data Goes Missing

Data may go missing for a variety of reasons. If a company using a particular stock symbol (e.g., JAVA) was acquired or sold to another company, the stock may stop trading (at least under that symbol).

Another reason may be that the company is thinly traded (companies that do not have a high market capitalization tend to trade in-and-out the market during periods of time).

## Why This Is Bad - What Can We Do

So how do we deal with data that is missing?

Recommended approach: we want to **fill forward** (or **fill backwards** depending on the gap and last traded price of interest) such that the last traded price before the gap is extended until it fills the gap (until the next non-empty traded price).

**Not** recommended: interpolation since it will modify the original data and attempt to predict prices (we do not want to create predictions with this activity).

## Section Quizzes

### Pandas `fillna()` Quiz

_How would you call fillna() to fill forward missing values_?

According to the [pandas docs](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html): `df.fillna(method='ffill')`.


# Lesson 6

Daily returns from a stock is important but having a way to visually compare daily returns for different stocks is also key. In this lesson we look at how to use histograms and scatter plots to visualize stock data.

## Histogram Of Daily Returns

When we plot our recorded daily returns, we can examine characteristics such as:

- Mean
- STD
- **Kurtosis**: measures how different the distribution is from a Gaussian normal distribution by examining the tails of the distribution

A _positive_ kurtosis indicates a fat tail (tails are outside of the Gaussian normal distribution) and a _negative_ kurtosis indicates skinny tail (tails are inside of the Gaussian normal distribution)

## Scatter Plots

Another way to visualize data is through scatter plots. We can plot the daily return value for each specific day on the scatter plot for two or more separate stocks and determine the overall trend.

## Fitting A Line To Data Points

When using scatter plots, it is common to use line of best fit for the data. The line of best fit may yield the following information:

- Slope ($\beta$): allows us to gauge how reactive a stock is to the market or another stock (`stock A % / stock B %`)
- Y-intercept ($\alpha$): allows us to gauge how well a stock is doing on average when compared to S&P 500 or another stock

## Slope Does Not Equal Correlation

Slope should **not** be confused with **correlation** (how well the line actually fits the data). A correlation close to one indicates high correlation whereas a correlation of close to zero indicates low correlation.

## Real World Use Of Kurtosis

In many cases, returns are assumed to be normally distributed (Gaussian distribution) in financial research. However, in reality kurtosis occurs and assuming that returns follow a normal distribution may cause disasters. One such incident is when investment banks created bonds based on mortgages (assuming normal distribution of returns and that the returns were independent) which unfortunately led to the [2008 financial crisis](https://www.investopedia.com/articles/economics/09/financial-crisis-review.asp).

## Section Quizzes

### What Would It Look Like Quiz

_Suppose we measured the daily return each day for the S&P 500, what would the histogram plot look like_?

C, a bell curve

### Compare Two Histograms Quiz

_Given a histogram of SPY and XYZ stock (imaginary stock), which of the following is true_?

XYZ has lower return and higher volatility, since the mean of XYZ is lower than the mean of SPY (lower return) and the spread of the data is greater indicating a higher STD (higher volatility).

### Correlation Vs. Slope Quiz

_Given the two scatter plots of SPY and XYZ (imaginary stock), which of the following is true_?

ABC has higher beta and higher correlation, since the slope is steeper (higher beta) and the line fits the data well (high correlation).


# Lesson 7

Previously, we have been mainly focused on statistics for stocks. In this lesson we will examine statistics for portfolios and how to evaluate performance.

We consider a portfolio an allocation of funds to a set of stocks and will follow a buy-and-hold strategy (for this lesson) where we invest in a certain set of stocks with a certain allocation and observe returns over time.

## Daily Portfolio Values

There are several steps we want to perform to determine our portfolio value:

1. Normalize the stock prices in our dataframe by dividing the prices after the first row with the first row
2. Multiply the normalized stock prices with the allocated amount for each stock
3. Multiply the allocated prices with the start value for each stock to determine the stock _position value_ for each day
4. Get the sum of each row (each day) from the stock position value dataframe to determine the _daily portfolio value_

## Portfolio Statistics

After we determine the portfolio value, we can examine several key portfolio metrics:

- Cumulative returns
- Average returns
- STD daily returns (risk)
- Sharpe ratio

## Sharpe Ratio

**Sharpe ratio** is a metric which adjusts returns according to risks (risk adjusted return). In general, the Sharpe ratio advocates (all else equal):

- Lower risk is better
- Higher return is better

The Sharpe ratio also considers risk free rate or return (e.g., returns from depositing money into a bank account).

## Computing Sharpe Ratio

Given that the Sharpe ratio is defined as $S = \frac{R_p - R_f}{\sigma_p}$, where $R_p$ is the portfolio return, $R_f$ is the risk free rate of return, and $\sigma_p$ is the STD of portfolio return. The ratio may be computed in Python as follows:

```python
s = mean(daily_returns - daily_rf_returns)/std(daily_returns - daily_rf_returns)
```

Where `daily_rf_returns` in the denominator may be dropped in most cases as it is a constant. A traditional shortcut is to use `daily_rf_returns = (1.0 + 0.1)^(1/252) - 1`.

## But Wait - There Is More

The Sharpe ratio can vary widely depending on how frequently you sample. The Sharpe ratio is an annual measure so if sampling more frequently we need to adjust as follows:

- Use `s_annualized = k * s` where `k = (samples per year)^(1/2)`
- If sampling daily use `k = (252)^(1/2)` (252 trading days a year)
- If sampling weekly use `k = (52)^(1/2)`
- If sampling monthly use `k = (12)^(1/2)`
![image](https://github.com/fersantanag/OMSCS_notes/assets/59957385/12fb190a-c43f-4311-b304-4e1ace29e929)
![image](https://github.com/fersantanag/OMSCS_notes/assets/59957385/e58e3a56-fed4-4095-abbd-29e704ed3b9e)

## Section Quizzes

### Which Portfolio Is Better Quiz

1. _Which portfolio (provided) is better? ABC or XYZ_? ABC, since the volatility is the same but the return is higher
2. _Which portfolio (provided) is better? ABC or XYZ_? XYZ, since the volatility is lower
3. _Which portfolio (provided) is better? ABC or XYZ_? XYZ, based on the Sharpe ratio

### Form Of The Sharpe Ratio Quiz

_Given the following portfolio metrics (provided), which formula is best_? The Sharpe ratio which is $\frac{R_p - R_f}{\sigma_p}$, where $R_p$ is the portfolio return, $R_f$ is the risk free rate of return, and $\sigma_p$ is the STD of portfolio return.

### What Is The Sharpe Ratio Quiz

_What is the Sharpe ratio given the following (provided)_? 12.7, based on the given formula for Sharpe ratio from lecture.


# Lesson 8

In this lesson we will be examining optimizers which will allow us to theoretically optimize portfolios and other statistics.

## What Is An Optimizer

An **optimizer** works by finding minimum values of functions. An optimizer also allows us to build parameterized models based on data. We can also refine allocations to stocks in portfolios as mentioned previously.

In general, we use an optimizer by:

1. Providing a function to minimize
2. Providing an initial guess
3. Calling the optimizer

## Convex Problems

**Convex problems** are easiest for the optimizer to solve. These problems are characterized by having a single minima where a line segment drawn between any two points on the graph is always above the graph (does not intersect more than two points).
![image](https://github.com/fersantanag/OMSCS_notes/assets/59957385/2ae640e3-dd3e-4e82-8ee3-2cc8ab290b6c)

## Building A Parameterized Model

When building out a parameterized model, we should think about what we are minimizing. E.g., find a line of best fit for scatter plot which requires finding the coefficients that minimize the distance between all points on the plot.

## Section Quizzes

### How To Defeat A Minimizer Quiz

1. _Would this function (provided) be hard to solve and why_? Yes, because this function has a constant initial condition which does not provide a gradient until later
2. _Would this function (provided) be hard to solve and why_? Yes, because this function has multiple minima (local)
3. _Would this function (provided) be hard to solve and why_? No, because the minima would be at the intersection of the two lines
4. _Would this function (provided) be hard to solve and why_? Yes, because this function is not continuous and because of the constant initial condition

### What Is A Good Error Metric Quiz

_Which of these metrics (provided) would be good for minimizing_? Either `sum(abs(error))` or `sum(squared(error))`.


# Lesson 9

The final project of this course is to create a portfolio optimizer. In this lesson we will define what a portfolio optimizer is and how to go about creating one.

## What Is Portfolio Optimization

A portfolio optimizer works as follows:

- _Given_ a set of assets and a time period
- _Find_ an allocation of funds to assets that maximizes performance

What is performance? Recall that we can use metrics such as cumulative return, risk (STD returns), risk adjusted return (Sharpe ratio)

## Framing The Problem

One way to optimize our portfolio while minimizing risk is to optimize the Sharpe ratio of our portfolio. We may do approach optimization by:

1. Providing a function to minimize $f(x) = -SR  (where SR is Sharpe ratio)
2. Providing an initial guess for $x$
3. Calling the optimizer

Where $x$ are the allocations (a vector of allocations for each stock) we are looking for to optimize our portfolio $f(x)$.

## Ranges And Constraints

To make the optimizer more efficient we could set:

- **Ranges**: limits on values for $x$ (e.g., each allocation cannot be over 100%)
- **Constraints**: properties of $x$ which must be true (e.g., sum of all allocations cannot be over 100%)

## Section Quizzes

### Which Criteria Is Easiest To Solve For Quiz

_For which (given criteria) would it be easiest to solve_? Cumulative return since all we have to do is invest all money into one stock which would optimize our portfolio (no good for risk however).


# Lesson 10

In this lesson we will examine how financial institutions utilize ML (machine learning) techniques and tools to predict future prices for stocks or other assets.

## The ML problem

ML models are used for many things. In general, a ML model is one that takes in an input $x$ (typically a vector of observations) and outputs $y$ (predictions).

Not all financial models utilize ML (e.g., [Black Scholes Model](https://www.investopedia.com/terms/b/blackscholes.asp))

The ML process consists of:

1. Feeding data into a ML algorithm
2. Producing a ML model from the ML algorithm
3. Using the ML model to predict $y$ with observations $x$
![image](https://github.com/fersantanag/OMSCS_notes/assets/59957385/403668f0-8574-4f24-841e-0fe4ce7501ad)

## Supervised Regression Learning

**Supervised regression learning** is simply a numerical prediction that occurs after an ML model is developed with data ($x$ and $y$ examples). Some techniques include:

- Linear regression (parametric since it does not store previous results)
- K-nearest neighbors (instance based since it stores previous results)
- Decision trees
- Decision forests

## How It Works With Stock Data

To leverage ML we have to start with out historical data for $x$ observations (which may be multi-dimensional depending on how many things we want to look at) and map each of our $x$ to the corresponding $y$ outputs. This information will be stored in our database.

## Example At A FinTech Company

How does the ML process work at a typical FinTech company?

1. Select $x$ (predictive factors) and $y$ outputs
2. Set desired time period and how many stock parameters we want to look at (e.g., P/E ratio, alpha, beta, etc.)
3. Train our model using a ML algorithm (in this case one of the supervised regression learning algorithms)
4. Plug a new $x$ into our trained ML model to produce a new $y$ prediction

## Backtesting

How can we be confident that our models can predict future stock prices? One way to evaluate our model is through **backtesting** which utilizes a portion of historical data to predict prices in the future (the _historical future_ which is the time period that _already_ occurred) then compare that to the _actual_ prices which occurred within the same time period.

## Problems With Regression

There are several problems with regression however:

- Predictions are noisy and uncertain (the more data the better)
- Challenging to estimate prediction confidence
- Unknown holding time and allocation

Some of these challenges may be solved with reinforcement learning (e.g., policy learning where a system learns whether to buy or sell a stock).

## Section Quizzes

### What Is X And Y Quiz

1. _Which of the following (provided) could be X_?
   - Price momentum
   - Bollinger value
   - Current price
2. _Which of the following (provided) could be Y_?
   - Future price
   - Future return


# Lesson 11

In this lesson we will learn more about supervised regression learning which utilizes regression algorithms to produce numerical outputs based on a set of numerical inputs.

## Parametric Regression

**Parametric regression** utilizes a line of best fit (e.g., linear, polynomials, etc.) to estimate our data and then finds the parameters (coefficients of the equation) to describe our overall model.

## K-nearest Neighbor

The **kNN (k-nearest neighbor)** approach utilizes data (instance based approach) to find the mean value of the nearest $k$ data points to produce a $y$ output.

A similar approach is **kernel regression** which utilizes weights on each kNN based on distance (evaluating contributions based on distance). The kNN approach only assigns equal weights to all kNN.

## Parametric Vs. Non-parametric

Comparing parametric vs. non parametric approach:

- Parametric:
  - Pros: we do not have to store the original data (space efficient - faster queries)
  - Cons: cannot update the model as more data is gathered (slower training)
- Non-parametric:
  - Pros: new data can be added easily (no parameters needed - faster training)
  - Cons: have to store all data points (huge data set - slow queries)
![image](https://github.com/fersantanag/OMSCS_notes/assets/59957385/2f00b140-e8ec-4156-9f80-732747e3712a)
![image](https://github.com/fersantanag/OMSCS_notes/assets/59957385/4eed5178-3255-4485-a01c-aacd9bd9458d)
![image](https://github.com/fersantanag/OMSCS_notes/assets/59957385/0f98ec01-5a67-4b05-9efd-ea65cb6b1558)

## Training And Testing

We should split up training and testing sets (out-of-sample testing) in order to evaluate that the trained model does in fact predict prices with some level of confidence and uncertainty.

Since we are examining stock prices over time in this class we should split up training and testing sets based on different time periods. Typically, training data is based on _older_ time periods while testing data is based on more _recent_ time periods.

## Learning APIs

The learning APIs may be structured for parametric and non-parametric approaches as follows:

- Linear regression:

  ```python
  learner = linear_regression_learner()
  learner.train(x_train, y_train)
  y = learner.query(x_test)
  ```

- kNN:

  ```python
  learner = knn_learner(k = <int>)
  learner.train(x_train, y_train)
  y = learner.query(x_test)
  ```

Essentially, the `learner` should first be trained with training sets (`x_train`, `y_train`), then be fed an input of `x_test` which would yield an output of `y`.

## Example For Linear Regression

The pseudo-code for linear regression may look something like this:

```python
class linear_regression_learner:
    def __init__():
        pass
    def train(x, y):
        self.m, self.b = <linear regression equation of choice>(x, y)
    def query(x):
        y = self.m * x + self.b
        return y
```

The API for kNN will be nearly identical.

## Section Quizzes

### How To Predict Quiz

_What should we do with these points (provided)_? Find the mean of their $y$ values such that an observed $y$ may be determined.

### Parametric Vs. Non-parametric Quiz

> Determine whether we should use a parametric or non-parametric regression approach for the following scenarios.

1. Parametric, since it is more obvious how $x$ and $y$ may be related (biased)
2. Non-parametric, since it is not obvious what happens when $x$ increases (unbiased)


# Lesson 12

Previously we learned about parametric (linear regression) and non-parametric (kNN) methods. In this lesson we will learn more about how to assess the pros and cons of each algorithm.

## Metric 1: RMS Error

How do we quantify how well our model fits the data? We may use the **RMSE (root mean squared error)** is the standard deviation of the residuals which is mathematically defined as follows:

$RMSE = \sqrt{\frac{\sum(y_(test) - y_(predict))^2}{N}}\\$

The RMSE tells us how far (on average) the model (prediction) is away from the actual (test) data. It is important to note that the _in-sample_ (training set) RMSE is different than the _out-of-sample_ (testing set) RMSE since the two sets of data are separate in practice.
![image](https://github.com/fersantanag/OMSCS_notes/assets/59957385/78212340-8264-4aec-9c23-d9a34727f579)

## Cross Validation

Often researchers do not have enough data to analyze their algorithm. This can be remedied by **cross validation** whereby data is sliced into multiple chunks of training and testing. In this way, one set of data could be used for multiple trials by switching chunks of training and testing data for each trial.

## Roll Forward Cross Validation

An important note is that cross validation is not as effective in financial research if a trial includes the testing set before the training set (no peeking into the future). Consequently, a best practice is to always have our training set **before** our testing set and _rolling forward_ each pair of train and testing data sets until we run out of data.

## Metric 2: Correlation

Another metric we may use to evaluate our model is through **correlation**. This may be done by generating a set of $y_(predict)$ using the test set and then plotting the $y_(test)$ outputs on the same plot.

Correlation may be evaluated using the NumPy method `np.corrcoef()` where the coefficient of correlation may be in a range of -1 to 1 or 0 (no correlation).

Note that the correlation coefficient is **not** related to the slope of the correlation line.
![image](https://github.com/fersantanag/OMSCS_notes/assets/59957385/4b0ba25f-822c-4d8e-839e-bf0eca576744)
![image](https://github.com/fersantanag/OMSCS_notes/assets/59957385/ac5e5680-d1a9-467e-9c62-c071385a9d49)

## Overfitting

**Overfitting** occurs when the in-sample (training set) error is low but the out-of-sample (testing set) error is high. Given a parametric model, if the in-sample error decreases while the out-of-sample error increases, then our model is likely overfitting.
![image](https://github.com/fersantanag/OMSCS_notes/assets/59957385/9465cce6-dcbb-4ed1-b780-7060d876369d)
![image](https://github.com/fersantanag/OMSCS_notes/assets/59957385/dc7695c2-4388-47b6-9d78-68a0933ac92d)

## Section Quizzes

### What Happens As K Varies Quiz

_Select the correct corresponding chart for each given $k$ value_.

1. The second chart since $k = 1$ averages over exactly one point
2. The first chart since $k = 3$ averages over three point
3. The second chart since $k = N$ averages over all points
4. False, as we increase k, we are _less_ likely to overfit (since we have more neighbors)

### What Happens As D Varies Quiz

_Select the correct corresponding chart for each given $d$ value_.

1. The second chart since $d = 1$ is linear
2. The first chart since $d = 2$ is quadratic
3. The second chart since $d = 3$ is polynomial
4. True, as we increase k, we are _more_ likely to overfit

### Which Is Worse Quiz

_Given RMSE of training and test set, which would you expect to be larger_? The out-of-sample (test set) error since it is data not previously seen.

### Correlation RMS Error Quiz

_In general as RMS error increases correlation.._. decreases since the trend line (prediction) does not match the data as closely.

### Overfitting Quiz

_Given the kNN plots, which depicts the model overfitting_? B, since the in-sample error decreases while the out-of-sample error increases (plot is independent on k so overfitting is observed at the beginning of the plot where $k = 1$).

### A Few Other Considerations Quiz

> Given the following additional considerations, select the best answer.

1. _Which is better on space saving for the model_? Linear regression
2. _Which is better on compute time to train_? kNN
3. _Which is better on compute time to query_? Linear regression
4. _Which is better on ease to add new data_? kNN
![image](https://github.com/fersantanag/OMSCS_notes/assets/59957385/05fcdd64-63ef-4f81-a602-bca83af88020)


# Lesson 13

In 1988 Michael Kearns and Leslie Valiant asked:

> Can a set of weak learners be combined to created a single strong learner?

This question was answered in 2009 following a Netflix 2006 challenge: create a ML algorithm which can perform 10% better than their current algorithm for predicting movies the user may like to watch. The winning algorithm was comprised of an _ensemble_ of ML algorithms which were able to solve the challenge. In this lesson we will examine ensemble learners.

## Ensemble Learners

**Ensemble learners** are just many different types of learners combined together to produce an output where the output is the average of all learner outputs. What are some benefits of using ensemble learners?

- Lower error
- Less overfitting

These are possible since each learner inherently has their own biases. We may reduce these biases and mitigate overfitting by averaging the outputs of each learner.
![image](https://github.com/fersantanag/OMSCS_notes/assets/59957385/5591226d-238a-4ab0-9c5d-9c68682664c1)

## Bootstrap Aggregating (Bagging)

**Bootstrap aggregating (bagging)** may be accomplished by using the **same** learner and training each learner on a different set of the data (taking the data and creating new random _bags_ of data through _bagging_). After training, we average the outputs of each learner. This technique is able reduce bias and overfitting even with the **same** learner due to training on separate data sets and averaging results.

Note: We still want to separate the training _bags_ of data with the testing _bags_ of data.
![image](https://github.com/fersantanag/OMSCS_notes/assets/59957385/a767628d-b7fd-46be-b278-c634c4fcb576)
![image](https://github.com/fersantanag/OMSCS_notes/assets/59957385/19f11866-e806-4553-8d55-3efc5ec93488)

## Boosting

**Boosting** or AdaBoost is a technique which takes the bootstrap bagging technique one step further by adding the following steps:

- Random _bag_ of data is created from the training set
- Learner is trained on the training set and tested using the **training** set again
- Training data outside of an RMSE limit will be weighted such that the next learner has a higher chance of being trained with a _bag_ of data consisting of the outlier data points

This process is repeated for all subsequent learners until the last learner and the result is that the average of all learners are _boosted_ in such a way that the output fits the test data more closely.
![image](https://github.com/fersantanag/OMSCS_notes/assets/59957385/83b77691-2641-443f-af5a-36f09640f3cd)
![image](https://github.com/fersantanag/OMSCS_notes/assets/59957385/b177b8fd-775c-42c9-b45a-1ca06a837d69)

## Ensemble Learners: Bagging And Boosting Summary

In summary, ensemble learners are wrappers around existing methods which may benefit financial researchers as it is able to create models which reduces error and overfitting.

## Section Quizzes

### How To Build An Ensemble Quiz

_Given the following, what is the best way of going about building an ensemble_? D, we can train several parameterized polynomials of differing degrees then train several kNN models using different subsets of data thereby reducing bias.

### Which Is Most Likely To Overfit Quiz

_Given the following, which model is most likely to overfit_? Single 1NN model trained on all the data.

### Overfitting Quiz

_Given the following,, which is more likely to overfit as m increases_? AdaBoost since it assigns more specific data to subsequent learners in the ensemble.


# Lesson 14

In this lesson we will learn about computational investing, portfolio management, and what portfolio managers do. Typically, portfolio managers will be responsible for some type of fund and how well the fund performs.

## Types Of Funds

There are various types of funds a portfolio manager might handle:

- **ETF (exchange traded fund)** are funds which:
  - Have ability to buy and sell just like stocks
  - May consist of a basket of stocks
  - Are very transparent and liquid
- **Mutual Fund** are funds which:
  - Have ability to buy and sell at the end of day
  - Only disclose holdings quarterly
  - Are less transparent and liquid
- **Hedge Fund** are funds which:
  - Only have the ability to buy and sell by agreement
  - Are not required to disclose holdings
  - Are not transparent and may be difficult to liquidate
![image](https://github.com/fersantanag/OMSCS_notes/assets/59957385/625b62f6-b3a3-40f7-866f-08c11d687930)

## Liquidity And Capitalization

**Liquidity** refers to the ease at which one could buy or sell shares or holdings. **Capitalization** refers to how much the company is worth according to the number of shares and the price of each share.

Note that the price of a stock does not necessarily reflect the value of a company but the price at which the stock is trading at.

## Incentives For Fund Managers

How are portfolio managers compensated? One way to calculate compensation is by examining **AUM (assets under management)** which refers to the amount of money managed by the fund. The compensation also varies for fund type:

- ETFs: compensation is through expense ratios which are some percentage of AUM (typically very low for ETFs)
- Mutual funds: compensation is also through expense ratios which are some percentage of AUM (however, higher than ETFs)
- Hedge funds: compensation is typically based on around 2% of the AUM and 20% of profits made (_two and twenty_ rule)

However, recently the compensation for majority of hedge funds are lower than _two and twenty_.

## How Funds Attract Investors

What type of investor does a hedge fund attract?

- Wealthy individuals
- Institutions
- Funds of funds

Why would these types of investors invest in a hedge fund?

- Proven track record
- Simulation and a good story
- Good portfolio fit

## Hedge Fund Goals And Metrics

There are a several goals that a hedge fund might have:

- Beat a benchmark (e.g., outperform the S&P 500)
- Obtain absolute return (through _long_ or _short_)

_Long_ is when a hedge fund makes positive bets on stocks they think are going to go up. _Short_ is when a hedge fund makes negative bets on stocks they think are going to go down.

How would an investor measure how well a hedge fund is doing? They may do so through examining:

- Cumulative returns (gains/losses over some period of time)
- Volatility (standard deviation)
- Risk/reward ratio (Sharpe ratio)

![image](https://github.com/fersantanag/OMSCS_notes/assets/59957385/20b5307d-6801-4af7-957d-32d754732de9)

## Section Quizzes

### What Type Of Fund Is It Quiz

> Given several funds, select whether the fund is an ETF, mutual fund, or hedge fund.

1. VTINX is a mutual fund
2. DSUM is an ETF
3. FAGIX is a mutual fund
4. Bridgewater Pure Alpha is a hedge fund
5. SPLV is an ETF
![image](https://github.com/fersantanag/OMSCS_notes/assets/59957385/edae3d73-bdfc-4470-a467-6fa9e14ad1d9)

### Incentives Quiz

> Given the following incentive structures, select the motivations.

1. _Which motivates AUM accumulation_? Both expense ratio and _two and twenty_
2. _Which motivates profits_? _Two and twenty_ rule
3. _Which motivates risk taking_? _Two and twenty_ rule since you have to take more risks to make those gains
![image](https://github.com/fersantanag/OMSCS_notes/assets/59957385/a15a898d-b3a8-4e09-b1be-7e5c02126799)


# Lesson 15

In this lesson we will learn about how the market works (e.g., what actually happens behind the scenes when you buy or sell a stock).

## What Is In An Order

To understand what happens when we buy or sell a stock (essentially placing n order), we should know what composes an order:

- Order type (buy or sell)
- Symbol
- Number of shares
- Limit or market order
- Price

## The Order Book

Each exchange keeps an order book for each stock that is traded. The order book enables the exchange to track metrics such as:

- Order type
- Price
- Size

This allows the exchange to track and order which stocks people are selling and buying such that orders can be fulfilled in a sequential manner. Hence, not everyone gets all their desired shares at the desired price. In some cases orders may not be fulfilled if certain specified order conditions are not met (e.g., limit orders where orders may not fulfilled if the price does not meet the limit sell or buy price).

## How Orders Get To The Exchange

A simple example of how orders get to an exchange is through a broker where the broker places the order at the best price on one of the exchanges.

However, it is common now to see 80-90% of all trades going through a **dark pool** which acts as an intermediary between brokerages and exchanges. Dark pools examine vast amounts of orders across various exchanges to predict where prices will go and pay brokers the privilege of peeking into people's orders before orders make it to the exchange. By having brokers interact with dark pools, both parties save money on exchange fees.

## How Hedge Funds Exploit Market Mechanics

Hedge funds can exploit the market in a variety of ways:

- Order book exploit: a hedge fund may have a co-located server which observes the order book microseconds faster than anyone else - this essentially allows them to buy and sell at the best prices before anyone else
- Geographic arbitrage: differences in prices may occur due to market inefficiencies which allows hedge funds to buy in at a lower price in one region of the world and sell at a higher price in another region

## Additional Order Types

Exchanges have the following order types:

- Buy and sell
- Market limit

Additional order types are offered by the broker:

- Stop loss - sell shares when it drops to a certain price
- Stop gain - sell shares when it reaches a certain price
- Trailing stop - combination of stop loss and an adjustable margin (stop loss limit drags along with price rise)
- Sell short - take a negative position on a stock (sell shares when prices go down)

Note that when _shorting_, we are selling stocks we do not own.

## Mechanics Of Short Selling: Exit

**Short selling** or _shorting_ is a way someone could make or lose money (generally large amounts) since shares are borrowed from another investor through a brokerage.

Positive example:

- Person A borrows $10000 worth of shares (100 shares at $100 each) from an investor through a broker to short
- The stock trading at $100 _drops_ to $90 - a $10 difference so Person A pockets 100 shares multiplied by $10 which equals $1000

Negative example:

- Person B borrows $10000 worth of shares (100 shares at $100 each) from an investor through a broker to short
- The stock trading at $100 _rises_ to $150 - a $50 difference so Person B is out 100 shares multiplied by $50 which equals $5000

## Section Quizzes

### Up Or Down Quiz

_Given the order book with a specific set of orders, is the price of the stock going to go up or down_? The price will go down since the size of the sell is much higher.

### Short Selling Quiz

_What would be the net return if you shorted IBM in this given situation_? Since we have 100 shares and it drops $10, $10 multiplied by 100 shares would give us $1000.


# Lesson 16

In this lesson we will examine how to evaluate a company and its worth.

## Why Company Value Matters

Company value matters if we are trying to determine the best time to sell stock. We could estimate the value of a company through their _true value_ by using the following metrics:

- **Intrinsic value**: the value of a company based on future dividends
- **Book value**: the value of all assets the company owns
- **Market cap**: the value of the stock on the market and shares outstanding

## The Value Of A Future Dollar

To calculate the value of a dollar we may use the following formula:

`PV = FV / (1 + IR) ** i`

Where `PV` is the present value, `FV` is the future value (payment at regular intervals), `IR` is the interest rate, and `i` is the time (usually years) into the future.

## Intrinsic Value

Generally, _interest_ and _discount_ rate are terms that refer to the same quantity, but are used to distinguish two slightly different use cases:

- **Interest rate**: is used with a given PV, to figure out what the FV would be
- **Discount rate**: is used when we have a known or desired FV, and want to compute the corresponding PV

Generally the discount rate has an _direct_ relationship with risk of a particular company (i.e., discount rate is lower if the company is less risky but higher if the company is more risky).

To determine the intrinsic value (`IV`) of a company we may use:

$IV = \sum_{i = 1}^{\infty} \frac{FV}{(1 + IR)^i} = FV/DR$

Where `DR` is the discount rate.

## Book Value

How do we calculate the book value of a company? We take the total assets (e.g., real estate) minus intangible assets (e.g., brand) and liabilities (e.g., loans). Note that we usually do **not** count intangible assets.

## Market Capitalization

Market cap is calculated by `market_cap = n_shares * price`.

## Why Information Affects Stock Price

News about a company could increase or decrease the intrinsic value of a company. Information may affect a company in the following ways:

- **Company-specific**: news that affects a specific company
- **Sector-specific**: news that affects a specific sector
- **Market-specific**: news that affects a specific market

## What Is A Company Worth Summary

Knowing what a company is worth will allow us to determine proper action. A hedge fund might evaluate the intrinsic value and market cap of a company to determine a buy or sell (_buy_ when intrinsic value is _high_ but prices are _low_ and _sell_ when intrinsic value is _low_ but prices are _high_).

Generally, the market cap of a particular company would not go below book value. Otherwise a predatory company would buy it and sell all assets at book value to pocket the difference.

## Section Quizzes

### What Is A Company Worth Quiz

_Given that a company can generate $1 every year, what is a company worth_? Between $10 and $25 depending on interest rates.

### The Balch Bond Quiz

_Rank the below options (given) from 1 to 3 where 1 is the most preferred and 3 the least. When comparing these different options, assume that they cost you the same today. Say, you have 80 cents to invest, and these are the 3 options you can get for that money_.

1. $1 right now
2. US government bond $1 in 1 year
3. Tucker Balch bond

### Intrinsic Value Quiz

_What's the intrinsic value given the following (provided)_? $50

### Compute Company Value Quiz

_Calculate the following for a company_:

1. _What is the book value given the following_? $80M
2. _What is the intrinsic value given the following_? $20M
3. _What is the market capitalization given the following_? $75M

### Would You Buy This Stock Quiz

_Given the following company values (provided), would you buy this stock_? Yes, since we can sell and profit $5M right away.


# Lesson 17

In this lesson we will examine the **CAPM (Capital Assets Pricing Model)** (developed by Sharpe, Markowitz, Merton, and Miller) which explains how the market impacts individual stock prices and provides a mathematical framework for hedge fund investing. CAPM led to the belief of index funds and the idea of _"you can't beat the market"_.

## Definition Of A Portfolio

A **portfolio** is a weighted set of assets where the sum of all weights must equal 1 (i.e., 100%). The returns of a portfolio for a particular point in time may be calculated by:

$r_p(t) = \sum_{i} w_i r_i(t)$

Where $r_p(t)$ is the portfolio return at time $t$, $w_i$ is the weight of the asset, and $r_i(t)$ is the return of the asset on a particular day.

## The Market Portfolio

The market portfolio is an index that broadly covers a wide variety of stocks (e.g., S&P 500). Most stocks in an index is set according to their weighted cap:

$w_i = \frac{m_cap_i}{\sum_{j} m_cap_j}$

Where $m_cap_i$ is the market cap for a particular stock and $m_cap_j$ is the market cap of any other stock in the index.

## The CAPM Equation

The CAPM equation is a linear equation (e.g., below for a particular stock) could be modeled as follows:

$r_i(t) = \beta_i r_m(t) + \alpha_i(t)$

Where $\beta_i r_m(t)$ represents market effects on returns (e.g., if the market goes up X percent then the stock will go up Y percent) and $\alpha_i(t)$ represents residual effects (with an expected value of zero although not always zero in practice). The extent to which the market affects a particular stock is represented by $\beta_i$.

Both $\beta_i$ (represented as the slope of the daily return line) and $\alpha_i(t)$ (represented as the y-intercept of the daily return line) are derived from daily returns for a particular stock when compared to an index.
![image](https://github.com/fersantanag/OMSCS_notes/assets/59957385/0174d880-8d88-49da-bfe6-ab332f30b7b5)

## CAPM Vs. Active Management

Since the invention of CAPM, there have been two schools of though on the CAPM model regarding investing:

- **Passive investing**: the idea that you should buy an hold since $\alpha_i(t)$ from the CAPM model is random and the expected value is nearly zero in most cases
- **Active investing**: the idea that you should actively pick stocks as $\alpha_i(t)$ is predictable and the expected value is **not** necessarily random
![image](https://github.com/fersantanag/OMSCS_notes/assets/59957385/aac3139c-e345-4147-8804-3006d4aae9ab)

## CAPM For Portfolios

CAPM for portfolios leverages the same equation but for a variety of stocks:

$r_p(t) = \sum_{i} w_i (\beta_i r_m(t) + \alpha_i(t))$

Recall that passive investing CAPM would be different than an active investing CAPM:

- Passive CAPM: $r_p(t) = \beta_p r_m(t) + \alpha_p(t)$
- Active CAPM: $r_p(t) = \beta_p r_m(t) +  \sum_{i} w_i \alpha_i(t)$
  
![image](https://github.com/fersantanag/OMSCS_notes/assets/59957385/fefde531-c2d3-4310-8eac-884ed8831911)

## Implications Of CAPM

The implications of CAPM are:

- Expected value of $\alpha$ is almost always zero
- The only way to beat the market is by choosing $\beta$
- Choose high $\beta$ in up markets
- Choose low $\beta$ in down markets

However, EMH (Efficient Markets Hypothesis) claims that you cannot predict the market with $\beta$.
![image](https://github.com/fersantanag/OMSCS_notes/assets/59957385/c1582382-0cc5-4cf2-b548-d7a46d5b3934)

## Arbitrage Pricing Theory

**Arbitrage Pricing Theory** suggests that instead of treating $\beta$ as representing a particular stock with respect to the entire market, there are instead multiple $\beta$ for representing a particular stock with respect to various sectors of the market. Therefore, the return becomes:

$r_i(t) = \sum_{i} (\beta_i r_m(t)) + \alpha_i(t)$

Where $\beta_i r_m(t)$ represents a particular stock in a particular sector.
![image](https://github.com/fersantanag/OMSCS_notes/assets/59957385/f897ae37-0991-4fee-a634-d073e11b84b5)

## Section Quizzes

### What Is The Portfolio Return As A Percentage Quiz

_Given the following weights and returns (provided), what is the portfolio return as a percentage_? `(0.75)(0.01) + (-0.25)(0.02) = 1.25`

### Compare Alpha And Beta Quiz

_Given the following scenarios for $\beta$ and $\alpha$, select the best answer_.

1. _Which has a higher $\beta$_? ABC, since the slope is steeper
2. _Which has a higher $\alpha$_? ABC, since the y-intercept is greater
![image](https://github.com/fersantanag/OMSCS_notes/assets/59957385/70df989f-c38c-43c2-adcf-558510a9ace9)

### Implications Of CAPM Quiz

_Given the following market scenarios (provided), select the best answer_.

1. _In upward markets, do you want a larger or smaller $\beta_p$_? Larger, since you can take advantage of the high prices
2. _In downward markets, do you want a larger or smaller $\beta_p$_? Smaller, since you do not want to risk crashing


# Lesson 18

Hedge funds try to find stocks which will yield a higher market relative return or lower market relative loss. In this lesson we examine how a hedge fund might leverage CAPM to determine these stocks.

## Two Stock CAPM Math

For [long-short](https://www.investopedia.com/terms/l/long-shortequity.asp) strategies, it is possible to lose out on a market bet even with perfect $\alpha$ and $\beta$ if we do not have the proper _allocations_. How do we find the proper allocations for each stock such that our risk is zero (zero $\beta$)? We may leverage CAPM (shown below) to determine the set of allocations (weights) which would minimize risk

$r_p(t) = \sum_{i} w_i (\beta_i r_m(t) + \alpha_i(t))$

Once the proper weights are found which minimizes risk, we can compute the portfolio return. Note though that the beta values are not guaranteed in the future.

## CAPM For Hedge Funds Summary

Hedge funds may utilize CAPM as follows:

- Given:
  - $\alpha_i$ (forecast for a particular stock)
  - $\beta_i$ (risk for a particular stock)
- Solution: find $beta_p = 0$ such that market risk is minimized (find appropriate weights for each stock $w_i$)

Utilizing CAPM allowed for long-short strategies to emerge.

## Section Quizzes

### Two Stock Scenario Quiz

_Given information on the two stocks compute the following_:

1. $r_a$ % = 11
2. $r_a$ = $5.5
3. $r_b$ % = -19%
4. $r_b$ = -$9.5
5. total % return = -4%
6. total return = -$4

### Allocations Remove Market Risk Quiz

_Given the following scenarios for two stocks, determine the following such that market risk is minimized_:

1. $w_a$ = 0.67
2. $w_b$ = -0.33


