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



> ML Lecture2 18分34秒

