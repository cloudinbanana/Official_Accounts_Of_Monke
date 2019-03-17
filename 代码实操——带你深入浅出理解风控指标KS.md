### 代码实操——带你深入浅出理解风控指标KS
笔者在工作中计算单变量的ks值时，发现几个分布不同的变量好y计算的ks值相同，凭借统计直觉，发现一定存在问题，笔者从数据和计算ks代码两个方向进行排除。最后定位到计算使用stats.ks_2samp()函数计算ks值时，如果变量存在缺失值，计算得到ks值有误，下面笔者就来好好梳理一下ks值的前世今生。

#### ks检验介绍
笔者刚入门机器学习开始做的例子就是金融场景下风控模型。那时评价模型的好坏就用传统的机器学习评价标准，比如说准确率、精确率和AUC，对风控模型的ks指标还一无所知，倒是作为统计科班出身的童鞋，第一次见到ks想到的就是数理统计中的Kolmogorov-Smirnov检验(柯尔莫哥洛夫-斯米尔洛夫)。后来实习过程中，发现老板们在看风控模型结果最关注的结果就是ks指标，才开始对ks指标逐渐重视起来。在衡量模型效果时，对评分卡或者机器学习模型给出的违约概率和y值计算ks值，给出模型效果来确定模型的好坏（一般0.3左右可以使用，0.4以上模型效果较高，太高了超过0.7，可能模型有问题，这时候需要去debug一下是不是出现了特征信息泄露以及一些其他的问题）。笔者在实际应用中，发现市面上关于ks系统介绍的文章比较少，本文就系统的介绍一下ks的前世今生，以及在在风控模型下的多种实现。

##### KS检验-统计角度
首先，我们了解下统计上KS检验的概念。这里维基百科[Kolmogorov–Smirnov test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)已经解释的很详细了。**如果让我一句话解释KS检验的话：**我会说KS检验是比较一个频率分布f(x)与理论分布g(x)或者两个观测值分布的检验方法。提到检验，我们第一步就是明确我们检验的原假设和备择假设。其原假设$H0$: **两个数据分布一致或者数据符合理论分布**，定义$D=max| f(x)- g(x)|$，当实际观测值$D>D(n,α)$则拒绝$H0$，否则则接受$H0$假设。$D(n,α)$可以查表得到[Critical Values for the Two-sample Kolmogorov-Smirnov test (2-sided) ](https://www.webdepot.umontreal.ca/Usagers/angers/MonDepotPublic/STT3500H10/Critical_KS.pdf)。KS检验与t-检验之类的其他方法不同是KS检验不需要知道数据的分布情况，算是一种非参数检验方法。代价就是数据分布指定的情况下KS效果不如指定的检验好。在样本量比较小的时候，KS检验一般在分析两组数据之间是否不同时相当常用。具体的ks检验的Case可以看文章[KS-检验（Kolmogorov-Smirnov test） -- 检验数据是否符合某种分布](https://www.cnblogs.com/arkenstone/p/5496761.html)

##### KS检验-风控角度
从统计角度，我们知道KS是分析两组数据分布是否相同的检验指标。在金融领域中，我们的y值和预测得到的违约概率刚好是两个分布未知的两个分布。好的信用风控模型一般从准确性、稳定性和可解释性来评估模型。一般来说。好人样本的分布同坏人样本的分布应该是有很大不同的，KS正好是有效性指标中的区分能力指标：**KS用于模型风险区分能力进行评估，KS指标衡量的是好坏样本累计分布之间的差值**。好坏样本累计差异越大，KS指标越大，那么模型的风险区分能力越强。

KS的计算步骤如下：
- 计算每个评分区间的好坏账户数（计算的是特征的KS的话，是每个特征对应的好坏账户数）。
- 计算每个评分区间的累计好账户数占总好账户数比率(good%)和累计坏账户数占总坏账户数比率(bad%)。
- 计算每个评分区间累计坏账户占比与累计好账户占比差的绝对值（累计good%-累计bad%），然后对这些绝对值取最大值即得此评分卡的KS值。

#### KS检验-python实现
上面介绍了KS的统计原理以及实现方法，下面我们从三个不同的角度去实现KS的计算
##### ks_2samp实现
我们直接调用stats.ks_2samp()函数来计算。链接[scipy.stats.ks_2samp¶](https://github.com/scipy/scipy/blob/v0.19.1/scipy/stats/stats.py#L4682-L4760)为ks_2samp()实现源码，笔者按照源码实现了下，方便查看其中的cdf计算结果。

```
def ks_calc_2samp(data,score_col,class_col):
    '''
    功能: 计算KS值，输出对应分割点和累计分布函数曲线图
    输入值:
    data: 二维数组或dataframe，包括模型得分和真实的标签
    score_col: 一维数组或series，代表模型得分（一般为预测正类的概率）
    class_col: 一维数组或series，代表真实的标签（{0,1}或{-1,1}）
    输出值:
    'ks': KS值，'cdf_df': 好坏人累积概率分布以及其差值gap
    '''
    Bad = data.ix[data[class_col[0]]==1,score_col[0]]
    Good = data.ix[data[class_col[0]]==0, score_col[0]]
    data1 = Bad.values
    data2 = Good.values
    n1 = data1.shape[0]
    n2 = data2.shape[0]
    data1 = np.sort(data1)
    data2 = np.sort(data2)
    data_all = np.concatenate([data1,data2])
    cdf1 = np.searchsorted(data1,data_all,side='right')/(1.0*n1)
    cdf2 = (np.searchsorted(data2,data_all,side='right'))/(1.0*n2)
    ks = np.max(np.absolute(cdf1-cdf2))
    cdf1_df = pd.DataFrame(cdf1)
    cdf2_df = pd.DataFrame(cdf2)
    cdf_df = pd.concat([cdf1_df,cdf2_df],axis = 1)
    cdf_df.columns = ['cdf_Bad','cdf_Good']
    cdf_df['gap'] = cdf_df['cdf_Bad']-cdf_df['cdf_Good']
    return ks,cdf_df
```

##### crosstab实现
我们知道计算ks的核心就是好坏人的累积概率分布，我们采用pandas.crosstab函数来计算累积概率分布。
```
def ks_calc_cross(data,score_col,class_col):
    '''
    功能: 计算KS值，输出对应分割点和累计分布函数曲线图
    输入值:
    data: 二维数组或dataframe，包括模型得分和真实的标签
    score_col: 一维数组或series，代表模型得分（一般为预测正类的概率）
    class_col: 一维数组或series，代表真实的标签（{0,1}或{-1,1}）
    输出值:
    'ks': KS值，'crossdens': 好坏人累积概率分布以及其差值gap
    '''
    ks_dict = {}
    crossfreq = pd.crosstab(data[score_col[0]],data[class_col[0]])
    crossdens = crossfreq.cumsum(axis=0) / crossfreq.sum()
    crossdens['gap'] = abs(crossdens[0] - crossdens[1])
    ks = crossdens[crossdens['gap'] == crossdens['gap'].max()]
    return ks,crossdens
```

##### roc_curve实现
我们同时发现在sklearn库中的roc_curve函数计算roc和auc时，计算过程中已经得到好坏人的累积概率分布，同时我们利用sklearn.metrics.roc_curve来计算ks值

```
from sklearn.metrics import roc_curve,auc
def ks_calc_auc(data,score_col,class_col):
    '''
    功能: 计算KS值，输出对应分割点和累计分布函数曲线图
    输入值:
    data: 二维数组或dataframe，包括模型得分和真实的标签
    score_col: 一维数组或series，代表模型得分（一般为预测正类的概率）
    class_col: 一维数组或series，代表真实的标签（{0,1}或{-1,1}）
    输出值:
    'ks': KS值
    '''
    fpr,tpr,threshold = roc_curve((1-data[class_col[0]]).ravel(),data[score_col[0]].ravel())
    ks = max(tpr-fpr)
    return ks
```

#### 案例测试
##### 模拟数据data_test_1(数据中不含有NAN)
```
data_test_1 = {'y30':[1,1,1,1,1,1,0,0,0,0,0,0],'a':[1,2,4,2,2,6,5,3,0,5,4,18]}
data_test_1 = pd.DataFrame(data_test_4)
```
计算结果：

```
ks_2samp,cdf_2samp = ks_calc_2samp(data_test_1, ['a'], ['y30'])
ks_2samp
cdf_2samp

Out[7]: 0.5
Out[8]: 
     cdf_Bad  cdf_Good       gap
0   0.166667  0.166667  0.000000
1   0.666667  0.166667  0.500000
2   0.666667  0.166667  0.500000
3   0.666667  0.166667  0.500000
4   0.833333  0.500000  0.333333
5   1.000000  0.833333  0.166667
6   0.000000  0.166667 -0.166667
7   0.666667  0.333333  0.333333
8   0.833333  0.500000  0.333333
9   0.833333  0.833333  0.000000
10  0.833333  0.833333  0.000000
11  1.000000  1.000000  0.000000

ks_cross,cdf_cross = ks_calc_cross(data_test_1, ['a'], ['y30'])
ks_cross
cdf_cross

Out[10]: 0.5
Out[11]: 
y30         0         1       gap
a                                
0    0.166667  0.000000  0.166667
1    0.166667  0.166667  0.000000
2    0.166667  0.666667  0.500000
3    0.333333  0.666667  0.333333
4    0.500000  0.833333  0.333333
5    0.833333  0.833333  0.000000
6    0.833333  1.000000  0.166667
18   1.000000  1.000000  0.000000


ks_auc = ks_calc_auc(data_test_1, ['a'], ['y30'])
ks_auc

Out[12]: 0.5
```
三种方法计算得到的ks值均相同，且ks_calc_cross和ks_calc_2samp计算得到的cdf相同

##### 模拟数据data_test_2(数据中含有NAN)
```
data_test_2 = {'y30':[1,1,1,1,1,1,0,0,0,0,0,0,0],'a':[1,2,0,2,2,7,4,5,4,0,4,18,np.nan]}
data_test_2 = pd.DataFrame(data_test_2)
```
计算结果：

```
ks_2samp,cdf_2samp = ks_calc_2samp(data_test_2, ['a'], ['y30'])
ks_2samp
Out[15]: 0.69047619047619047

cdf_2samp
Out[16]: 
     cdf_Bad  cdf_Good       gap
0   0.166667  0.142857  0.023810
1   0.333333  0.142857  0.190476
2   0.833333  0.142857  0.690476
3   0.833333  0.142857  0.690476
4   0.833333  0.142857  0.690476
5   1.000000  0.714286  0.285714
6   0.166667  0.142857  0.023810
7   0.833333  0.571429  0.261905
8   0.833333  0.571429  0.261905
9   0.833333  0.571429  0.261905
10  0.833333  0.714286  0.119048
11  1.000000  0.857143  0.142857
12  1.000000  1.000000  0.000000


ks_cross,cdf_cross = ks_calc_cross(data_test_2, ['a'], ['y30'])
ks_cross
Out[18]: 
y30         0         1       gap
a                                
2.0  0.166667  0.833333  0.666667

cdf_cross
Out[19]: 
y30          0         1       gap
a                                 
0.0   0.166667  0.166667  0.000000
1.0   0.166667  0.333333  0.166667
2.0   0.166667  0.833333  0.666667
4.0   0.666667  0.833333  0.166667
5.0   0.833333  0.833333  0.000000
7.0   0.833333  1.000000  0.166667
18.0  1.000000  1.000000  0.000000

ks_auc = ks_calc_auc(data_test_2, ['a'], ['y30'])
ks_auc

Traceback (most recent call last):
ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
```
三种方法计算得到的ks值均不相同。
- 其中ks_calc_2samp计算得到的ks因为**searchsorted()函数**（有兴趣的同学可以自己模拟数据看下这个函数），会将Nan值默认排序为最大值，从而改变了数据的原始累积分布概率，导致计算得到的ks和真实的ks有误差。
- 其中ks_calc_cross计算时忽略了NAN，计算得到了数据正确的概率分布，计算的ks与我们手算的ks相同
- ks_calc_auc函数由于内置函数无法处理NAN值，直接报错了，所以如果需要ks_calc_auc计算ks值时，需要提前去除NAN值。
#### 总结
在实际情况下，我们一般计算违约概率的ks值，这时是不存在NAN值的。所以以上三种方法计算ks值均可。但是当我们计算单变量的ks值时，有时数据质量不好，存在NAN值时，继续采用ks_calc_auc和ks_calc_2samp就会存在问题。

解决办法有两个 1. 提前去除数据中的NAN值 2. 直接采用ks_calc_cross计算。
