## DSGD MF in scala and spark

Implemantation of [Large-scale matrix factorization with distributed stochastic gradient descent](https://dl.acm.org/citation.cfm?id=2020426) in spark


### Conclusion
DSGD 对 rating 矩阵进行分块，如果氛围 N x M  块，对 rating 矩阵完成一次分解，
则需要 spark shuffle N * M 次。

在实际业务场景中，user 数 >> item 数，则可只对
user RDD 和对应的 rating RDD 分为 N 份，在 N 个 spark executor 上。item RDD 分为 N 份在 N 个 executor 上做循环，循环一次，则完成了一次对
rating 矩阵的分解。由于 item RDD 相对比较小，所以 shuffle 循环的耗时不大。


实测时，在数据量巨大，spark executor num 较大时，总体耗时很大，后续可考虑用 GPU 节点作为 executor 进行提速
