## Gradient Descent

目标是找到使L最小的$\theta$
$$
\theta^* = \mathop {argmin}_{\theta} L(\theta)
$$
设$\theta=\{\theta_1, \theta_2\}$

初始化$\theta_0=\left[ \begin{matrix} \theta_1^0 \\ \theta_2^0 \end{matrix} \right]$

则
$$
\left[ \begin{matrix} \theta_1^1 \\ \theta_2^1 \end{matrix} \right]=\left[ \begin{matrix} \theta_1^0 \\ \theta_2^0 \end{matrix} \right]-\eta \left[ \frac {\frac {\partial L(\theta_1^0)}{\partial \theta_1}}{\frac {\partial L(\theta_2^0)}{\partial \theta_2}} \right]
$$
即
$$
\theta_1 = \theta_0 - \eta \nabla L(\theta^0)
$$

通常$\eta$随着update越来越小。



## Adagrad

$$
w^{t+1} \leftarrow w^t - \frac {\eta^t}{\sigma^t} g^t
$$

其中（g是gradient的意思）
$$
\eta^t = \frac {\eta}{\sqrt{t+1}} \\
g^t = \frac {\partial L}{\partial w}
$$
而最重要的
$$
\begin{split}
&\sigma^t是前面出现过的所有微分的平方均值root\ mean\ square \\
&即\sigma^t = \sqrt{\frac 1{t+1} \sum^t_{i=0} (g^i)^2}
\end{split}
$$
例如
$$
\begin{split}
w^1 \leftarrow  w^0 - \frac {\eta^0}{\sigma^0} g^0 \qquad &\sigma^0 = \sqrt{(g^0)^2} \\
w^2 \leftarrow  w^1 - \frac {\eta^1}{\sigma^1} g^1 \qquad &\sigma^1 = \sqrt{\frac 12[(g^0)^2+(g^1)^2]} \\
\cdots
\end{split}
$$
其中$\eta^t$和$\sigma^t$中的t+1可以抵消，则最终Adagrad公式为
$$
w^{t+1} \leftarrow w^t-\frac {\eta}{\sqrt{\sum^t_{i=0} (g^i)^2}} g^t
$$

> Adagrad会越来越慢，老师说Adam是现在比较好的
>
> Adagrad中，当$g^t$越大时，后续update的分母就越大，即此次更新的步伐越大，则后续步伐越小。
>
> Adagrad中这种思想就是靠"反差"，当前gradietn除以之前的gradient的平方均值。



#### Adagrad的思想

![image-20210711160119336](.\images\image00.png)

上图是L分别随$w_1$和$w_2$的变化曲线。

可以看到两个曲线中的共同点是：离最佳点的距离越远，则gradient越大(越陡)。

但是从两个曲线的对比来看，可以发现：取两个曲线距离最佳点相同距离的点来看，二次微分越小的曲线其gradient越小。

例如，在w2中可能取一点其gradient很大(因为曲线很陡)，但是不代表w2就应该变化得很大，相反由于w2曲线很陡，w2要更新的步伐反而应该较小点。

这种思想就是the best step is
$$
\frac {|一次微分|}{二次微分}
$$
而在实际训练中，取二次微分的耗时很大，因此就可以用**平方的均值再开方的值来代替二次微分**。

一般来说二次微分越大，图像变化的越快，均值平方开方的值越大。



## Stochastic Gradient Descent

传统的梯度下降的Loss Function是当前参数$\theta$在所有样本的误差的和
$$
L=\sum_n(\hat y^n-(b+\sum w_i x_i^n))^2
$$
而随机(stochastic)梯度下降法，则每次只是随机抽一个样本的误差作为Loss
$$
L^n=(\hat y^n-(b+\sum w_i x_i^n))^2
$$
在更新参数的时候只考虑对那一个example的Loss的gradient，然后就立马update参数。

优点：非常非常快



## Feature Scaling特征缩放

如果不同属性分布的range不同，那么建议做Scaling。

![image-20210711193854772](.\images\image01.png)

如果不scaling，则会在训练后期更新困难。

如何对一个属性scaling：

- 计算均值m，计算标准差（方差的平方根）$\sigma$，则

$$
x\leftarrow \frac {x- m} {\sigma}
$$



## Math Theory

Taylor展开式：如果h(x)在某一点$x_0$可以无限微分，则h(x)可以写成
$$
\begin{equation}\begin{split}
h(x)&=\sum_{k=0}^{\infin} \frac {h^{(k)}(x_0)}{k!}(x-x_0)^k\\
&=h(x_0)+h'(x_0)(x-x_0)+\frac {h''(x_0)}{2!}(x-x_0)^2+\dots
\end{split}\end{equation}
$$
当x很接近x0时
$$
h(x) \approx h(x_0)+h'(x_0)(x-x_0)
$$
Taylor展开式的多变量情况：
$$
h(x,y)\approx h(x_0,y_0)+\frac {\partial h(x_0,y_0)}{\partial x}(x-x_0)+\frac {\partial h(x_0,y_0)}{\partial y}(y-y_0)
$$
回到Loss Function，假设当前参数$\theta$只有a和b，则
$$
L(\theta)\approx L(a,b)+\frac {\partial L(a,b)}{\partial \theta_1}(\theta_1-a)+\frac {\partial L(a,b)}{\partial \theta_2}(\theta_2-b)
$$
则在$(\theta_1-a)^2+(\theta_2-b)^2 \le d^2$的范围内，很显然当前L增大最快的方向是$\vec v$
$$
\left[ \frac {\partial L(a,b)}{\partial \theta_1}, \frac {\partial L(a,b)}{\partial \theta_2} \right]
$$
因此在范围内，只要取和$\vec v$相反方向的那个点就是使L减小最快的点。**这就是梯度下降的原理**。即
$$
To\ minimize\ L(\theta)\\
\left[ \begin{matrix}
\triangle \theta_1 \\
\triangle \theta_2
\end{matrix} \right]
=-\eta
\left[ \begin{matrix}
u \\
v
\end{matrix} \right]
$$
也就是说要让Gradient Descent有效，必须要求Loss Function是满足泰勒展开式的。

为什么只用泰勒展开式的一阶展开？因为快。

为了保证泰勒展开式逼近真实的函数，则范围$d$不能取得太大，也就是在机器学习的时候学习率$\eta$不能太大。

