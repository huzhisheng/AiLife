> 不讲数学证明，什么时候会一定收敛？

- SGD，stochastic gradient descent，随机梯度下降法
- SGDM，stochastic gradient descent with momentum，带动量的随机梯度下降法
- Adagrad
- RMSProp
- Adam

>一些注释
>
>$\theta_t$：神经网络的参数
>
>$\nabla L(\theta_t)$：Loss function的gradient
>
>$m_{t+1}$：从step 0到step t积累的动量
>
>$x_t$：神经网络的输入
>
>$y_t$：神经网络的输出
>
>$\hat y_t$：真实数据的Label



On-line：每个time step只输入一个数据

Off-line：每个time step会把所有的数据都输入到Model后算出一个完整的$L(\theta)$再更新

> 后面的介绍暂时都假设是Off-line的，即每个时刻$L(\theta)$只与$\theta$有关。



## SGD

SGD就是：
$$
\theta^1=\theta^0-\eta \nabla L(\theta^0) \\
\theta^2=\theta^1-\eta \nabla L(\theta^1) \\
\dots \\
stop\ until\ \nabla L(\theta^t) \approx 0
$$

## SGDM

SGDM就是：公式中的v代表动量
$$
\begin{split}
&v^0=0 \\
&v^1=\lambda v^0-\eta \nabla L(\theta^0) \\
&\theta^1 = \theta^0+v^1 \\
&v^2=\lambda v^1-\eta \nabla L(\theta^1) \\
&\theta^2 = \theta^1+v^2 \\
&\dots
\end{split}
$$
SGDM的好处就是避免训练过程在一些假的局部最优点上停下。



## Adagrad

Adagrad就是：
$$
\theta_t=\theta_{t-1}-\frac {\eta}{\sqrt{\sum_{i=0}^{t-1}(g_i)^2}}g_{t-1}
$$
如果过去Gradient很大，代表在一个比较陡峭的方向，因此用小一点的学习率；相反如果过去gradient很小，代表在一个比较平缓的方向，因此用大一点的学习率。



## RMSProp

RMSProp和Adagrad的唯一差别就是学习率的分母不一样：
$$
\begin{split}
&\theta_t=\theta_{t-1}-\frac {\eta}{\sqrt{v_t}}g_{t-1} \\
&v_1=g_0^2 \\
&v_t=\alpha v_{t-1}+(1-\alpha)(g_{t-1})^2
\end{split}
$$
也就是时间越近的gradient的影响越大。且之前的Adagrad有一个问题就是分母会无限地增大，最后就必然越来越小。



## Adam

Adam就是将SGDM中的Momentum和RMSProp中的$\vec v$合并起来：（因为RMSProp还是会有在假的局部最优点停下来的问题）
$$
\begin{split}
&\theta_t = \theta_{t-1}-\frac {\eta}{\sqrt{\hat v_t}+\epsilon} \hat m_t \\
&\hat m_t=\frac {m_t}{1-0.9^t} \\
&\hat v_t=\frac {v_t}{1-0.999^t} \\
&\epsilon = 10^{-8}
\end{split}
$$



> BERT、Transformer、Tacotron是用Adam训练出来的
>
> Yolo是用SGDM

SGDM和Adam已经是两个最厉害的优化器了。一般来说：

- Adam会快速提高准确率，但是在validation上的表现会有较大的落差
- SGDM一般来说训练得较慢，但是非常的稳定，converge比较好，且在validation上的表现也不错

这是因为可能Adam找到的minimum是非常陡峭的峡谷中的minimum，而SGDM则是找到的是比较平坦的平原上的minimum，这样当training和validation的数据有所偏差时，SGDM的表现不会差太多。



## SWATS

SWATS就是Adam和SGDM的结合，其想法就是一开始用Adam，准确率到达一定阈值后就转而使用SGDM进行进一步的收敛。



## AMSGrad

作者提出之前Adam会导致某一步的gradient即使再大，其更新的步长也是会受限制的，即bounded。即假设前面1000步的gradient都是1，在第1001步的时候gradient突变为100000，则这一步的更新步长也最多就是$10\sqrt{10} \eta$，但前面的1000步已经更新了差不多$1000\eta$，因此其实Adam会导致真正有用的信息受限制。

因此作者提出了AMSGrad，目的是移除小gradient的影响：
$$
\begin{split}
&\theta_t=\theta_{t-1}-\frac {\eta}{\sqrt{\hat v_t}+\epsilon}m_t \\
&\hat v_t=max(\hat v_{t-1}, v_t)
\end{split}
$$
相当于$\hat v_t$只取过去的$v_t$中的最大值了。



## AdaBound

前面的AMSGrad只考虑了当gradient很大的情况。

AdaBound也是针对Adam的一个修改版，其思想就是强制给Adam中的学习率通过Clip函数设定一个学习率的上界和下界：
$$
\begin{split}
&\theta_t=\theta_{t-1}-Clip(\frac {\eta}{\sqrt{\hat v_t}+\epsilon})m_t \\
&Clip(x)=Clip(x, 0.1-\frac {0.1}{(1-\beta_2)t+1}, 0.1+\frac {0.1}{(1-\beta_2)t})
\end{split}
$$


## Cyclical LR

这个方法是针对SGDM的改进，就是说学习率会周期性的从小到大，当大到设定的阈值时又减小。

## SGDR

这个方法和Cyclical LR类似，只不过学习率的变化曲线呈三角波函数的样子。

## One-Cycle LR

这个方法和Cyclical LR类似，只不过学习率的变化是只有一轮的先增大后减小，呈一个三角形。

> 上面三种方法都是通过人工设定Learning Rate的方法希望让SGDM收敛的比较快。



## Adam with warm-up

warm-up的意思就是说，刚开始训练的时候，由于随机初始化，因此Adam的gradient和学习率都有点混乱，直到训练几轮后gradient和学习率才能进入“正轨”（这里我没完全弄明白）

而做了warm-up后，则在训练的初始几轮，Adam的gradient和学习率就能比较正确。

#### 什么是warm-up

warm-up就是一种学习率优化方法（最早出现在ResNet论文中）。在模型训练之初选用较小的学习率，训练一段时间之后（如：10epoches或10000steps）使用预设的学习率进行训练；

#### 为什么用warm-up

1. 因为模型的weights是随机初始化的，可以理解为训练之初模型对数据的“理解程度”为0（即：没有任何先验知识），在第一个epoches中，每个batch的数据对模型来说都是新的，模型会根据输入的数据进行快速调参，此时如果采用较大的学习率的话，有很大的可能使模型对于数据“过拟合”（“学偏”），后续需要更多的轮次才能“拉回来”；
2. 当模型训练一段时间之后（如：10epoches或10000steps），模型对数据具有一定的先验知识，此时使用较大的学习率模型就不容易学“偏”，可以使用较大的学习率加速模型收敛；
3. 当模型使用较大的学习率训练一段时间之后，模型的分布相对比较稳定，此时不宜从数据中再学到新特点，如果仍使用较大的学习率会破坏模型的稳定性，而使用小学习率更容易获取local optima。



## RAdam

RAdam就是Adam用了warm-up技术的改进版，具体来说就是
$$
\begin{split}
&前面几步采用 \\
&\theta_t=\theta_{t-1}-\eta\hat m_t \quad 相当于就是SGDM算法 \\
&后面几步采用 \\
&\theta_t=\theta_{t-1}-\frac {\eta r_t}{\sqrt{\hat v_t}+\epsilon} \hat m_t
\end{split}
$$
其中$r_t$是一个会随着gradient方差变大而变小的变量，刚开始训练时gradient的方差较大，因此$r_t$会较小，这就相当于warm-up，而当后面gradient稳定时$r_t$就会较大，这时就相当于是Adam算法。

因此RAdam其实是和SWATS完全相反的算法，先用SGDM后用Adam。



## Optimizer Wrapper

Optimizer Wrapper就是一些可以通用于所有optimizer的方法。

#### Lookahead

这个方法就是说，先从a往前走虚假的k步走到了b，然后再倒回到a处，再从a往$\vec {ab}$方向上走一定距离，这个才是真正的一步。

#### Nesterov accelerated gradient (NAG)

就是SGDM算法中的动量m改成来自未来的动量即可。（相当于根据未来的动量来进行此刻的更新）

之前的SGDM可以改写为：
$$
\begin{split}
&\theta_t=\theta_{t-1}-m_t \\
&m_t=\lambda m_{t-1}+\eta \nabla L(\theta_{t-1}) \\
&变为 \\
&\theta_t=\theta_{t-1}-\lambda m_{t-1}+\eta \nabla L(\theta_{t-1}) \\
&m_t=\lambda m_{t-1}+\eta \nabla L(\theta_{t-1}) 
\end{split}
$$
而NAG中就是$\lambda m_{t-1}$变为了$\lambda m_{t}$：
$$
\begin{split}
&\theta_t=\theta_{t-1}-\lambda m_{t}+\eta \nabla L(\theta_{t-1}) \\
&m_t=\lambda m_{t-1}+\eta \nabla L(\theta_{t-1}) 
\end{split}
$$

#### Nadam

类似NAG对SGDM，Nadam也是将Adam中一个$m_{t-1}$换成$m_{t}$即可。



> [TA 补充课] Optimization for Deep Learning (2/2)  33min

