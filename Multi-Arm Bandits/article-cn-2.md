随机对照试验的贝叶斯方法（2）
=======================

**作者：** *罗子俊*

在随机对照试验的贝叶斯方法（1）当中讨论了Epsilon Greedy这个算法，并且给出了详尽的Python代码。现在，我们就来看看这个代码的细节。

首先，我们导入`numpy`和`random`这两个库。我们会用到`numpy`当中的`argmax`以及`random`当中的`randrange`：

```python
import numpy as np
import random
```

然后，我们设定三个全局参数：

```python
# set the number of bandits
N_bandits = 5
# set the number of trials/visitors
N = 100000
# set the number of trials to try all bandits
N_start = 50
```

在实际应用中，`N_bandits`的值将取决于你使用中所测试的版本数，而访客数`N`则是未知的。

在这个代码中，我们将会构造一个名为的`BayesianAB`的类（class）。这篇文章里的所有算法，都会放在这个类的下面。我们首先初始化以下数值：

```python
class BayesianAB:
  def __init__(
      self,
      number_of_bandits: int = 2,
  ):
    self.prob_true = [0] * number_of_bandits # only in demonstration
    self.prob_win = [0] * number_of_bandits
    self.history = []
    self.count = [0] * number_of_bandits
    self.a = [1] * number_of_bandits
    self.b = [1] * number_of_bandits
```

`BayesianAB` 默认有两个老虎机。我们预分配六个列表来储存算法所用到的值：
* `prob_true` 用来储存每个老虎机胜率的真实值。在实际应用当中，这些真实值是未知的；
* `prob_win` 用来储存每个老虎机胜率的计算值。这个值会在每一轮当中更新；
* `history` 用来储存每一轮所产生的 `prob_win` 值。这些值既可以用于计算下一轮的均值，也可以用来在最后评估算法的效果；
* `count` 用来储存每个老虎机所被使用的次数；
* `a` and `b` 是汤普森采样（`Thompson Sampling`）方法所需要用到的数值。这个算法会在后面讨论。

下面几行代码构建老虎机胜率的真实值：

```python
    # set the last bandit to have a win rate of 0.75 and the rest lower
    # only in demonstration
    self.prob_true[-1] = 0.75
    for i in range(0, number_of_bandits-1):
      self.prob_true[i] = round(0.75 - random.uniform(0.05, 0.65), 2)
```

其中，最后一个老虎机有最高的胜率，在这里设置为.75。剩下的老虎机胜率在.1和.7之间。我们也可以直接指定几个胜率，但是我更喜欢我在这里用的方法，因为这样我就可以允许胜率值的数量随着`N_bandits`的值而改变。

接下来，我们定义两个函数。文章里的好几个算法都会用到这两个函数：

```python
  # Returns a random value of 0 or 1
  # only in demonstration
  def pull(
      self,
      i,
  ) -> bool:
    return random.random() < self.prob_true[i]

  # Updates the mean
  def update(
      self,
      i,
      k,
  ):
    outcome = self.pull(i)
    # may use a constant discount rate to discount past
    self.prob_win[i] = (self.prob_win[i] * k + outcome) / (k+1)
    self.history.append(self.prob_win.copy())
    self.count[i] += 1
```

`pull()`函数会比较`random.random()`所产生的随机数以及老虎机$i$的胜率。如果随机数小于胜率，那么就返回1，反之，则返回0。这个函数在实际应用当中是多余的。在实际应用中，每当网站有一个新的访客时，系统就会触发`BayesianAB`或者这个类里面的某一个方法。在这个访客离开网站前，你就会知道他/她有没有买东西。如果买了，就是1，没有买，就是0。

`update()`函数会更新预期值，并且把一些其他数值储存到`history`当中去。最后，它还会给老虎机$i$的计数器加1。

下面是`BayesianAB`中真正执行`贪婪算法`的代码：

```python
  def epsilon_greedy(
      self,
      epsilon: float, # decay epsilon?
  ) -> list:

    self.history.append(self.prob_win.copy())

    for k in range(0, N_start):
        i = random.randrange(0, len(self.prob_win))
        self.update(i, k)

    for k in range(N_start, N):
      # find index of the largest value in prob_win
      i = np.argmax(self.prob_win)

      if random.random() < epsilon:
        j = random.randrange(0, len(self.prob_win))
        # If the randomly picked bandit is the same as one from argmax, pick a different one
        while j == i:
          j = random.randrange(0, len(self.prob_win))
        else:
          i = j

      self.update(i, k)

    return self.history
```

上面的代码与伪代码的逻辑基本一致。首先，在第一个迭代循环中，头50个（`N_start`）访客会被随机分配到5个不同的版本。借着，`update()`函数会根据随机结果更新版本 $i$ 的均值。从第51个访客开始，第二个迭代循环将会被执行。这个循环会执行以下步骤：
1. 找到预期回报率最高的版本 $i$；
2. 检查从`random.random()`中得到的随机值是否小于`epsilon`。如果随机值小于`epsilon`，那么将随机选取一个版本 $j$；`epsilon`的值将在代码运行的时候被指定；
3. 如果随机选选取的版本 $j$ 与预期回报率最高的版本 $i$ 是同一个版本，那么就随机再选取另外一个版本，直到 $j$ 与 $i$ 不相等为止；
4. 通过`update()`函数来更新被选中版本的均值。

`epsilon_greedy`会返回实验的完整历史，其中包括每个版本被选中的次数以及它们的预期值。

我们可以通过下面的代码来执行`epsilon_greedy`。我们还会打印一些基本的结果：

```python
eg = BayesianAB(N_bandits)
print(f'The true win rates: {eg.prob_true}')
eg_history = eg.epsilon_greedy(epsilon=0.5)
print(f'The observed win rates: {eg.prob_win}')
print(f'Number of times each bandit was played: {eg.count}')
```

