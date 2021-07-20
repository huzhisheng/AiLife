> What I cannot create, I do not understand. 要会创造一个事物，你才算真正地理解它。

## PixelRNN

> To create an image, generating a pixel each time.

- 看前1个pixel，产生下一个pixel；
- 看前2个pixel，产生下一个pixel；
- 看钱3个pixel，产生下一个pixel；
- $\cdots$

#### WaveNet

waveNet就是PixelRNN差不多的思想，根据直接输入进来的wav(采样得到的声音的amplitude，一点处理也没有的原始数据)不断生成下一帧的wav数据。

#### VAE简单介绍

> VAE将在"18.md"中介绍
>
> 假设要生成三维的latent code
>
> VAE生成的latent code每一维看起来都有特定的含义

![image-20210718211359957](./images/image45.png)

> 上图中的Minimize下面的公式前面少了个负号

VAE用来作诗句：

- 其实就和Magenta中的两个音乐产生的中间的latent code拿去产生音乐，结果和两个音乐都很相似是一样的做法。
