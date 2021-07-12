> 回顾：
>
> deep learning的三步：
>
> 1. define a set of function
> 2. goodness of function
> 3. pick the best function
>
> $\theta$：网络中的所有参数的集合

神经网络就是个函数$f$,接受一个向量输出一个向量。

一般来说，input layer就是个值，没有计算功能，其它layer才有类似$w\cdot x+b$。



神经网络就是可以看作成：
$$
\sigma(W^L\cdots \sigma(W^2\cdot \sigma(W^1 \cdot x+b^1)+b^2)+\cdots+b^L)
$$
把output layer之前的各层都可以看作是在做feature extraction。PS：最后一层一般是softmax层。



## 如何设计有多少层中间层，以及中间层的神经元数目?

凭直觉...

or 遗传演算法之类的...



> 回顾：
> $$
> \nabla L=
> \left[ 
> \begin{matrix}
> \frac{\partial L}{\partial w_1} \\
> \frac{\partial L}{\partial w_2} \\
> \dots
> \end{matrix}
> \right]
> $$

