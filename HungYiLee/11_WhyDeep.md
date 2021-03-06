> 近似参数数量的情况下，deep的神经网络明显好于只有一层的神经网络

## 为什么Deep更好

- 模块化的思想

  类似写程序时将相似功能集中到一个模块(or 一个函数)的思想。模块化可以减少程序的复杂度。

  Deep的神经网络就相当于模块化使得网络更加简洁。例如：

  > 对{长发男, 长发女, 短发男, 短发女}这四种类别进行分类，则只有一层的神经网络必须一下子学会分辨这四种，那么对应每一种则很可能数据就不足够训练得充分。
  >
  > 而如果有两层神经网络，则第一层神经网络可以就设置两个神经元，分别代表“男or女”分类器和"长发or短发"分类器（相当于做一次feature extraction），然后第二层神经元就可以利用前面一层提取的特征进行更好的训练。

- Deep相当于就是减少了参数，不容易over fitting，或者说只需要较少的data。例如实际上两层的逻辑电路也可以构造出一台电脑，但实际上这是不可行的，因为过于复杂。而神经网络也是类似道理。

## 模组化 - Speech

#### 人类语言的结构

- Phonene音素：

  假设人说"How do you think?"，则音素是这个样子
  $$
  hh\ w\ aa\ t\quad d\ uw\quad y\ uw\quad th\ ih\ ng\ k
  $$

- Tri-phone：就是将一个音素与其前面和后面的音素组合起来
  $$
  \dots,t-d+uw,\quad d-uw+y,\quad uw-y+uw,\quad y-uw+th, \dots
  $$

- State：（为每个音素建立模型）一个音素可以拆成几个state

$$
t-d+uw\ \rightarrow\ \lbrace t-d+uw1,t-d+uw2,t-d+uw3 \rbrace
$$

#### 语音识别大致步骤

1. 识别State
   1. 分帧加窗，提取每一帧的acoustic feature(声学特征)，这样就得到Acoustic feature sequence
   2. 识别每一帧声学特征得到State
2. 识别Phonene
3. 将Phonene转成文字
4. 处理同音字问题

#### 传统语音识别

采用HMM-GMM模型。

GMM(Gaussion Mixture Model)中将每个state看作是一个高斯分布，每帧数据假设是$x$，则就去看哪个state的$P(x|state)$的几率最大，选几率最大的那个state就作为这一帧的state。

> HMM-GMM中每一帧的识别state都是独立进行的，然而实际上我们应该考虑其上下文环境再决定这一帧是什么state



5个母音：a, e, i, o, u其实之间的口型和舌头形状是会部分相同的，例如i(yi)和u(you)其实只改变了口型而舌头位置不变，因此我们没必要为每个音素都建立模型，而是它们之间的部分参数是可以共享的。



#### DNN语音识别

Deep Network就是说前面的层实际上只是在探测“舌头的位置”，“嘴巴的口型”之类的基础特征，而后面的层再根据这种类似的底层特征再做进一步的识别。

> 上述结论是学者将某一层hidden layer的输出结果进行降维后分析得出的结论。



> DNN与剪窗花的比喻

DNN前几层的Feature Transformation其实也类似剪窗花前将纸对折，对折后减去线性的一刀，那么展开后对应到原来的空间中可能就是非线性的。

李老师说根据这个“剪窗花”理论还可以知道，deep network和只有一层的神经网络相比，当data不足即training不充分时，只有一层的神经网络比deep network难看得多。虽然它们俩在充分训练后会差不多。



> DNN的另一好处，端到端

传统的语音识别，wav信号要经过"DFT", "滤波器", "log", "DCT"后得到MFCC，MFCC再去送给GMM模型进行识别。

而DNN则可以不经过如此多人工设计的算法层，而是经过多层神经网络，它会自动学习到这些功能。Google的一篇论文中曾尝试过这个方向，输入就只是个.wav信号，连傅里叶都不计算了，但最后最好的结果也只是和之前的方法打平并未明显地超越。



> DNN的另一好处，适合复杂的任务

例如萨摩耶和北京熊，两种动物非常相似，但是output却不同；而不同的汽车外形很不一样，但output却都是"Car"。DNN就更适合这种需要仔细辨别的任务，通过多层Layer总能学习到一些非常明显的特征进行标签辨识。

