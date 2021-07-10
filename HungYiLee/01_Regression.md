## Linear Model:

$$
y=b+\sum w_i x_i
$$

> 用$\hat y$表示真实值
>
> 用$(x^n_i,\hat y)$表示第n个数据的第i个属性，而y则是标签



## Loss Function $L$:

$$
\begin{equation}\begin{split}
L(f)&=L(w,b)\\
&=\sum^{10}_{n=1}(\hat y^n-(b+w·x^n_{i}))^2
\end{split}\end{equation}
$$



## Pike the best Function:

$$
f^*=\mathop{argmin}_{f}L(f)\\
w^*,b^*=\mathop{argmin}_{w,b}L(w,b)
$$

> PS：线性代数可以直接解出这个最好的$f$



## Gradient Descent:

现假设**只有一个**参数$w$
$$
w^*=\mathop{argmin}_{w}L(w)
$$

1. 随机初始化$w^0$
2. 计算在$w=w^0$的情况下，$\frac {dl}{dw}$的值，即L(w)与w组成的曲线在$w^0$点的斜率
3. $w^1 \leftarrow w^0-\eta \frac {dl}{dw}|_{w=w^0}$
4. 计算在$w=w^1$的情况下，$\frac {dl}{dw}$的值，$\cdots$

> 最终会达到local optimal​
>
> 在linear regression，没有local optimal的现象的出现，因为这里的Loss function L是convex的（凸函数），没有局部最佳的点，只有全局最佳。



现假设有两个参数$w,b$，和上面情况其实类似
$$
w^*,b^*=\mathop{argmin}_{w,b}L(w,b)
$$

1. 随机初始化$w^0, b^0$
2. 计算$\frac{\partial L}{\partial w}|_{w=w^0,b=b^0},\frac{\partial L}{\partial b}|_{w=w^0,b=b^0}$
3. $w^1\leftarrow w^0-\eta\frac{\partial L}{\partial w}|_{w=w^0,b=b^0}\\b^1\leftarrow b^0-\eta\frac{\partial L}{\partial b}|_{w=w^0,b=b^0}$
4. $\cdots$

> $$
> \nabla L=
> \left[ \begin{matrix}
> \frac{\partial L}{\partial w}\\\frac{\partial L}{\partial b}
> \end{matrix} \right]_{gradient}
> $$



二次式Function来更好的拟合
$$
y=b+w_1·x_{cp}+w_2·(x_{cp})^2
$$




将多种类别的Linear Function合并为一个大的Linear Function
$$
\begin{split}
if\ x_s=P \quad &y=b_1+w_1·x_{cp}\\
if\ x_s=W \quad &y=b_2+w_2·x_{cp}
\end{split}
$$
合并为
$$
\begin{equation}\begin{split}
y&=b_1·\delta(x_s=P)+w_1·\delta(x_s=P)x_{cp}\\
&=b_2·\delta(x_s=W)+w_2·\delta(x_s=W)x_{cp}
\end{split}\end{equation}
$$

> 其中
> $$
> \delta(x_s=P)=\begin{cases}
> 1&if\ x_s=P\\
> 0&otherwise
> \end{cases}
> $$

选取哪些参数到function中一般由具有domain知识的专家来进行，当然也可以都加进model中，由model来学习。



## Regularization

在Loss Function中把一些Knowledge放进去（范式作为惩罚项），例如
$$
L=\sum_n(\hat y^n-(b+\sum w_ix_i))^2+\lambda\sum(w_i)^2
$$
此时就是$w_i$越小越好，即function越平滑越好。此时output对输入的变化就很不敏感。

$\lambda$的增大，$L$会先减小后增大。太平滑的话，会在testing case上又表现得糟糕，例如极端情况下L变成一条直线。



在做Regularization时，不考虑上述式子中的$b$这一项，即bias不参与Regularization。

因为预期是找一个平滑的function，而bias与平滑程度没关系。

