error的来源主要是“bias”和“variances”



## Bias

假设随机变量x的均值$\mu$，方差是$\sigma$。

随机抽取N points：{$x^1,x^2,\cdots,x^N$}，则
$$
m=\frac 1N\sum_nx^n\ne\mu
$$
但是
$$
E[m]=E[\frac1N\sum_n x^n]=\frac 1N \sum_n E[x^n] = \mu
$$
因为m的期望值就等于$\mu$，因此说m是**unbiased**。



## Variance

又有方差公式如下
$$
Var[m]=\frac {\sigma^2}N
$$
如何估计方差? 用$s^2$
$$
s^2=\frac 1N \sum_n(x^n-m)^2
$$
计算$s^2$的期望值
$$
E[s^2]=\frac {N-1}N \sigma^2
$$
只有N很大的时候$s^2$才和$\sigma^2$没多大差别，因此$s^2$是**有bias**的。（瞄准时都没瞄准）

> bias就是你瞄准靶子时，有没有瞄准，而variance则是射出去的子弹因为其它因素而导致的随机误差。



把每次抽样并计算出来的function $f^*$看作是一个抽样，则简单的Model一般其Variance就较小。

简单的Model有较大的bias，但有较小的variance；复杂的Model有较小的bias，但有较大的variance；

Error来自于bias很大，则是underfitting；error来自于variance很大，则是overfitting；



**判断Model是bias大还是variance大**：

如果Model无法fit训练数据，则是bias大；

如果Model能fit训练数据，但在testing data上面误差很大，则是variance大；



如果bias大，需要redesign模型；

如果variance大，需要**增加data**(甚至可以自己制作假data：翻转图片、男声转女声、手动加噪音) or 加正则式regularization。



## N-fold Cross Validation

例如3-fold，就把Training Set均分成3份，进行3轮训练。第i轮把第i份数据看作是Vailidation Set其余看作是Training Set，这样3轮训练就得到1个Function的3个Model，并且得到它们在Validation Set的Error，计算3个Error的均值，选择均值最小的一个Function，作为最后的Model，并在整个Training Set上再去训练一遍，再拿到真正的Testing Set上去测试。
