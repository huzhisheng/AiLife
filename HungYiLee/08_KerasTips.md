> TensorFlow相比于Keras更Flexible，因此也就更需要学习成本；
>
> Keras相当于是一个方便用户使用的TensorFlow，一个TensorFlow的接口；

简单识别Minst的Keras1.x示例：

```python
model = Sequential()
model.add(Dense(input_dim=28*28, output_dim=500))
model.add(Activation('sigmoid'))
model.add(Dense(output_dim=500))
model.add(Activation('sigmoid'))
model.add(Dense(output_dim=10))
model.add(Activation('softmax'))

model.compile(loss = 'categorical crossentropy', optimizer='adam', metrics=['accuracy'])

#x_train的shape=(n, 28*28), y_train的shape是(n, 10)
model.fit(x_train, y_train, batch_size=100, nb_epoch=20)

score = model.evaluate(x_test, y_test)
print('Total loss on Testing Set:', score[0])
print('Accuracy of Testing Set:', score[1])

result = model.predict(x_test)
```



#### Batch

- batch肯定是要随机分；
- 每一轮epoch训练，是随机选batch进行训练，不放回抽取；
- 每个batch在训练时，Loss是每个样本的loss之和，然后再对大Loss求微分再进行更新；（一个batch更新一次）
- 如果Batch-size=1，则就是随机梯度下降SGD；
- 实际上batch_size越大，一个epoch的时间越短；（因为batch_size越大减少了update的次数，且batch_size越大更能发挥GPU的并行运算）；但batch_size过于大的时候也会导致因为存储(显存 or 内存)不足而导致训练时间增大，且batch_size过大则越容易进入局部最优点。



简单的Keras2.0示例（只有add函数那里有变化）

```python
model = Sequential()
model.add(Dense(input_dim=28*28, uints=500, activation='relu'))
model.add(Dense(units=500, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss = 'categorical crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=100, nb_epoch=20)

score = model.evaluate(x_test, y_test)
print('Total loss on Testing Set:', score[0])
print('Accuracy of Testing Set:', score[1])

result = model.predict(x_test)
```

