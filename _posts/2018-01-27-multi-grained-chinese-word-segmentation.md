---
layout: post
title: 中文多粒度分词实验记录
tags:
- MWS
categories: MWS
description: 记录中文多粒度分词实验结果
---
### 2018/1/26
- CKY解码部分多进程，设置进程数为10

 | |训练|测试|总时间
---|---|---|---
单进程 | 3m11s|2m41s|5m52s
多进程 | 1m46s|1m15s|3m01s

### 2018/1/23
- 优化给定span(left,right)获得oracle label index的部分

 | |训练|测试|总时间
---|---|---|---
优化前 | 5m18s|2m39s|7m57s
优化后 | 3m11s|2m41s|5m52s



### 2018/1/18
- 计算label score部分为大矩阵运算，同时计算一个句子所有span的label score
- lstm部分将一个batch的所有句子合并大矩阵输入到lstm中计算
- 在cky过程中用numpy计算，只保存label,split point,累计score。cky解码结束后再构建图，生成树。
- 训练和测试部分都使用大矩阵。
- 设置batchsize=100,epoch=10

 | |训练|测试|总时间
---|---|---|---
gpu大矩阵 | 5m18s|2m39s|7m57s
cpu大矩阵 | 7m02s|3m41s|10m43s
cpu大矩阵+mkl | 9m01s|4m05s|13m06s
cpu小矩阵 | 7m51s|4m11s|12m02s
cpu小矩阵+mkl | 5m18s|2m49s|8m07s




### 2018/1/10
##### 修改代码，只执行一次cky。设置batchsize100，迭代10次，比较gpu大矩阵和cpu小矩阵的速度：
1. 计算label score时间：cpu小矩阵177s，gpu大矩阵24s。
2. 计算loss时间：cpu小矩阵0.0074s，gpu大矩阵3.95s。
3. cky解码时间：cpu小矩阵139.5s（cpu小矩阵的cky代码中都是用的expression，实际上也包含了一部分构建图的时间，我把其中npvalue计算label score的时间扣除了，没有算在这里面），gpu大矩阵246.4s。导致gpu比cpu慢的原因主要在这一块。
4. 构建图的时间：gpu大矩阵119.5s。cpu小矩阵没有单独统计这一块的时间（cky解码包含了一部分构建图的时间）。
5. 反向传播和更新总时间：cpu小矩阵131.3 s，gpu大矩阵 26.1s。




##### 设置batchsize100，迭代10次，比较gpu大矩阵和cpu小矩阵的速度：
1. 计算label score时间：cpu小矩阵177s，gpu大矩阵23.4。
2. 计算loss时间：cpu小矩阵0.0074s，gpu大矩阵6.2s。
3. cky解码时间：cpu小矩阵139.5s（cpu小矩阵的cky代码中都是用的expression，实际上也包含了一部分构建图的时间，我把其中npvalue计算label score的时间扣除了，没有算在这里面），gpu大矩阵452.8s。导致gpu比cpu慢的原因主要在这一块。
4. 构建图的时间：gpu大矩阵114.6s。cpu小矩阵没有单独统计这一块的时间（cky解码包含了一部分构建图的时间）。
5. 反向传播和更新总时间：cpu小矩阵131.3 s，gpu大矩阵 27.9s。

总体来看，计算label score，反向传播和和更新的时间gpu比cpu快很多。计算loss时间gpu慢（这个原因不太明白）。gpu速度慢的主要地方在cky解码部分，猜想是cpu在这一部分只是构建图，而gpu大矩阵代码在这一部分要进行numpy运算，同时还要保存解码路径、错误label的个数等信息。



### 2017/12/20
#### 加速实验记录
- lstm层使用大矩阵和不使用大矩阵比较
- batchsize=50

| | 前向传播|反向传播
|---|---|---
lstm使用大矩阵 | 65.74s|9.62s
lstm不使用大矩阵 | 59.49s|12.32s



---

- 训练集和测试集都为100句句子，每句句子长度都是50个字。
- 表格中为迭代一次的时间。batchaize=10

#### lstm层同时对一个batch中的所有句子计算（计算label score的部分用下面的方法1构建大矩阵）

|| 训练时间|测试时间|总时间
---|---|---|---
|gpu大矩阵 | 2m10s|42s|==2m53s==









#### 构建大矩阵
##### 方法1.计算span的label score时，将一个句子中所有span representation拼成大矩阵，同时计算所有span的label score

|| 训练时间|测试时间|总时间
---|---|---|---
|gpu大矩阵 | 2m4s|40s|==2ms44s==
|gpu小矩阵 | 1m24s|51s|2m15s
|cpu大矩阵 | 2m10s|39s|2m49s
|cpu小矩阵 | 58s|26s|1m24s


##### 方法2.计算span的label score时，将batch中所有句子中所有span representation拼成大矩阵，同时计算所有span的label score

|| 训练时间|测试时间|总时间
---|---|---|---
|gpu大矩阵 | 3m54s|1m1s|4m55s
|gpu小矩阵 | 1m24s|51s|2m15s

                
##### 观察实验结果，发现将batch中所有句子中所有span representation拼成大矩阵计算，比将一个句子中所有span representation拼成大矩阵，循环执行batchsize次速度要慢。
然后针对该现象在gpu上进行实验：
设计一组实验：
- 1：(1,2500)*(2500,4)+(1,4) 执行100000次
- 2：(10,2500)*(2500,4)+(10,4) 执行10000次
- 3：(100,2500)*(2500,4)+(100,4) 执行1000次
- 4：(1000,2500)*(2500,4)+(1000,4) 执行100次
- 5：(10000,2500)*(2500,4)+(10000,4) 执行10次
- 6：(100000,2500)*(2500,4)+(100000,4) 执行1次

实验 | 时间
---|---
1 | 26.231s
2 | 2.603s
3 | 0.277s
4 | 0.080s
5 | 0.050s
6 | 0.259s

再设计一组实验：


实验 | 时间
---|---
(1000,2500) 10000次| 8.023s
(10000,2500) 1000次 | 3.519s
(100000,2500) 100次 | 4.068s



能用实验解释方法1和方法2的上述现象，但是原理还不是很清楚。







### 2017/12/1

#### 之前emnlp17工作实验结果

| | P| R |F(dev)| P | R |F(test)
|---|---|---|---|---|---|---
|parsing(with bichar) | 96.55|96.40|96.48|97.00|95.16|**96.07**
|sequence labeling(with bichar) | 96.86|96.26|**96.59**|97.01|94.96|95.97
|SWS aggregation | 90.43|97.44|93.80|92.11|96.59|94.30
## 基于图的中文多粒度分词工作
#### baseline
借鉴ACL2017的论文[A Minimal Span-Based Neural Constituency Parser(Stern et al.)](http://xueshu.baidu.com/s?wd=paperuri%3A%2899342233f3ac1e772cec638ad7be1a5e%29&filter=sc_long_sign&tn=SE_xueshusource_2kduw22v&sc_vurl=http%3A%2F%2Farxiv.org%2Fabs%2F1705.03919&ie=utf-8&sc_us=1631692147359249170)把多粒度分词看做基于图的短语结构分析任务。
- dir: gpu/minimal-span-parser-MWS && gpu/minimal-span-parser-MWS-tag
- code: gpu/minimal-span-parser-MWS/src
- one-iter-max :15000
- data：gpu/data-mws/multi-4.9



| | P| R |F(dev)| P | R |F(test)
|---|---|---|---|---|---|---
|no wordcluster(no bichar) | 96.57|94.20|95.37|97.04|93.20|95.08
|with wordcluster(no bichar) | 96.60|94.36|95.47|97.08|93.41|95.21
|no wordcluster(with bichar) | 97.25|96.51|**96.88**|97.35|95.21|**96.27**





---

---

# 多粒度最新实验结果整理（基于转移、EMNLP17工作）
##### lstm-tagger模型(bichar)

|accuracy|dev: P|R|F|test: P| R|F|
|---|---|---|---|---|---|---
|94.43| 96.86=168710/174178|96.26=168710/175268 |96.59 |97.01=42997/44323|94.96=42997/45279|95.97

##### lstm-tagger模型(without bichar)

|accuracy|dev: P|R|F|test: P| R|F
---|---|---|---|---|---|---
93.21| 95.88=166394/173538|94.94=166394/175268 |95.41|96.56=42643/44162|94.18=42643/45279|95.35

##### lstm-tagger模型(3个单粒度模型合并)

|accuracy|dev: P|R|F|test: P| R|F
---|---|---|---|---|---|---
95.60(ctb)/95.43(pku)/96.63(msr)| 90.43=170778/188857|97.44=170778/175268|93.80|92.11=43734/47478|96.59=43734/45279|94.30

# 用coupled model自动转化的数据的PRF值

- 在1500句测试集上的评价结果：

P | R |F
---|---|---
99.64=44431/44590 | 98.13=44431/45279|98.88

- 多粒度词语分布统计：


|| # word(all the words)|单粒度词数|2粒度词数|3粒度词数|# word(most course-grained)|单粒度词语数|2粒度词语数|3粒度词语数
---|---|---|---|---|---|---|---|---
人工标注前 | 44590|74.50%=33218/44590|23.95%=10680/44590|1.55%=692/44590|36740|90.41%=33216/36740|9.45%=3472/36740|0.14%=52/36740
人工标注后 | 45279|71.58%=32412/45279|26.81%=12140/45279|1.61%=727/45279|36415|89.01%=32412/36415|10.85%=3952/36415|0.14%=51/36415


测试结果上：

|| # word(all the words)|单粒度词数|2粒度词数|3粒度词数|
---|---|---|---|---|
span-parser(bichar) | 44408|74.9%=33256/44408|23.5%=10455/44408|1.6%=697/44408
span-parser(without bichar) | 44434|74.2%=32965/44434|24.1%=10726/44434|1.7%=743/44434
lstm-tagger(bichar) | 44323|75.82%=33604/44323|22.73%=10075/44323|1.45%=644/44323
lstm-tagger(without bichar) | 44162|75.70%=33430/44162|22.80%=10068/44162|1.50%=664/44162
aggregation |47478|64.61%=30676/47478|31.38%=14900/47478|4.01%=1902/47478


# 单粒度最新实验结果整理
##### lstm-tagger模型（随机初始化）

||accuracy|P| R|F
---|---|---|---|---
ctb |95.60|94.34=47471/50317 |94.34=47471/50319|94.34
pku |95.43|96.01=113389/118106 |95.51=113389/118714|95.76
msr |96.63|96.52=102879/106588 |96.26=102879/106873|96.39

##### lstm-tagger模型（pretrain）

||accuracy|P| R|F
---|---|---|---|---
ctb|95.56| 94.17=47554/50498|94.51=47554/50319|94.34
pku |95.42| 95.86=113007/117891|95.19=113007/118714|95.52
msr |95.99| 95.93=102060/106395|95.50=102060/106873|95.71

##### lstm-tagger模型（pretrain & adagradient）

||accuracy|P| R|F
---|---|---|---|---
ctb|95.33| 94.06=47352/50341|94.10=47352/50319|94.08
pku |95.09| 95.67=112773/117871|95.00=112773/118714|95.33
msr |95.56|95.61=101454/106114 |94.93=101454/106873|95.27


##### lstm-tagger模型（pretrain & 部分更新embedding ）

||accuracy|P| R|F
---|---|---|---|---
ctb|95.54| 94.46=47405/50184|94.21=47405/50319|94.34
pku |95.55| 96.06=113598/118252|95.69=113598/118714|95.88
msr |96.54| 96.58=103011/106663|96.39=103011/106873|96.48




# 标注一致性
- cgong&ljguo:1090/1121=97.23
- jwsun&xzjiang:1131/1151=98.26
- zhli&cgong:1149/1170=98.21
- yzhu&yzhang:489/493=99.19

||姓名 |语料| 句子数|修改/处|diff/（处）|标对/（处）|一致性
---|---|---|---|---|---|---|---|
第一组|cgong |ctb| 49|8|20|7 |1149/1170=98.21
|     |zhli|  |   |18|  |13 |
第二组|ljguo |ctb| 45|11|18|8 |1090/1121=97.23
|     |cgong |   |   |15|  |10|
第三组|jwsun |msr| 41|11|15|9 |1131/1151=98.26
|     |xzjiang|  |   |10|  |6 |
第四组|yzhang |pku| 14|8|6|3 |489/493=99.19
|     |yzhu|  |   |10|  |3 |
|总共 |    |  |149|  |  |  |3859/3935=98.07
# 标注情况统计

姓名 | | 标注数据| 行数|修改 |标注时间（包括检查时间）  |
---|---|---|---|---|---|---
ljguo | 第一次|CTB|3000: 1001-4000|59|7h（1h）
| | 第二次|CTB|1372:11701-13072|26|2.5h(30min)
bzhang | 第一次|CTB|3000: 4001-7000|48|7h（0）
| | 第二次|CTB|1400:10301-11700|54|2h20min(20min)
jychao | 第一次|CTB|2000: 7001-9000|57|7h（30min）
| | 第二次|CTB|1300:9001-10300|40|2h(0.5h)
yzhang | 第一次|PKU|3000:1001-4000|53|7h（30min）
| | 第二次|PKU&MSR|1305:PKU中11301-11782&MSR中12601-13424|21|1h40min(10min)
xzjiang | 第一次|PKU|3000: 4001-7000|40|5h（0）
| | 第二次|MSR|1000:1-1000|10|2h30min(30min)
yzhu | 第一次|PKU|3000: 7001-10000|50|7h（2h）
| | 第二次|PKU|1782:10001-11782|38|2h30min(30min)
qrxia | 第一次|MSR|3000: 1001-4000|71|5h（1h）
| | 第二次|MSR|1300:10001-11300|15|2h(30min)
jwsun | 第一次|MSR|3000: 4001-7000|41|9h（3h）
| | 第二次|MSR|1000:1-1000|11|1h8min(0)
zwfan | 第一次|MSR|3000: 7001-10000|94|10h（2h）
| | 第二次|MSR|1300:11301-12600|31|2h30min(30min)


---


##### lstm tagger（pretrain）
（多粒度）

||original | pretrain
---|---|---
accuracy|94.46(60)| 93.75(55)


- accuracy为开发集上的结果
- PRF值均为测试集上的结果

(单粒度)

||accuracy|P| R|F
---|---|---|---|---
ctb|95.56| 94.17=47554/50498|94.51=47554/50319|94.34
pku |95.42| 95.86=113007/117891|95.19=113007/118714|95.52
msr |95.99| 95.93=102060/106395|95.50=102060/106873|95.71











# 数据整理
### coupled模型
###### msr-coupled-with-ctb5big (done)
- dir:176/home/jwsun/WS/exp/multi-grain-exp/msr-coupled-ctb5big
- code:176/home/jwsun/WS/src/src-r6-ws-offline-filter/
- tag-online-filter-lambda=0.995
- tag-online-filter-maxnum=16
- inst-num-from-train-1-one-iter=5000
- inst-num-from-train-2-one-iter=5000

|msr                | ctb5big
|-------------------|------------------
|==95.66（374/395）== | ==90.63(162/395)==

###### msr-coupled-with-pku126(done)
- dir:176/home/jwsun/WS/exp/multi-grain-exp/msr-coupled-pku126
- code:176/home/jwsun/WS/src/src-r6-ws-offline-filter/
- tag-online-filter-lambda=0.995
- tag-online-filter-maxnum=16
- inst-num-from-train-1-one-iter=5000
- inst-num-from-train-2-one-iter=5000

|msr                | pku126               
|-------------------|------------------
|==96.76（194/215）== | ==92.22(187/215)==    



### 准备offline filter data
172/disk4t/cgong/WS/data/multi-grain-data
1.  ctb模型：172/disk4t/cgong/WS/model/ctb5big-bs-5k-one-iter, Accuracy=90.69
2. pku模型：172/disk4t/cgong/WS/model/pku126-bs-5k-one-iter, Accuracy=92.11
3. msr模型：172/disk4t/cgong/WS/model/msr-bs, Accuracy=95.38
- msr:
> tag A:  msr正确答案（BIES@x）【172/home/cgong/WS/data/multi-grain-data/msr/msr-seg】  

> tag B: 在ctb5big/pku126模型上分析msr的结果）【172/home/cgong/WS/exp/multi-grain-data-prepare-exp/ctb5big-take_msr_as_test】【172/home/cgong/WS/exp/multi-grain-data-prepare-exp/pku126-take_msr_as_test】
- pku126/ctb5big: 
> tag A: 在msr上模型分析pku126/ctb5big的结果 【172/home/cgong/WS/exp/multi-grain-data-prepare-exp/msr-take_ctb5bigtrain_as_test】【172/home/cgong/WS/exp/multi-grain-data-prepare-exp/msr-take_pku126train_as_test】

> tag B: pku126/ctb5big正确答案【172/home/cgong/WS/data/multi-grain-data/pku126/pku126-seg】【172/home/cgong/WS/data/multi-grain-data/ctb5big/ctb5big-seg】




# 单粒度实验结果整理(随机初始化)
- accuracy为开发集上的结果
- PRF值均为测试集上的结果
##### CRF模型
（旧数据）


||accuracy|P| R|F
---|---|---|---|---
ctb |95.55| 93.99=47284/50310|93.97=47284/50319|93.98
pku |97.04| 97.74=115138/117800|96.99=115138/118714|97.36 ==（用大规模数据训练）==
msr |95.24| 95.20=101631/106757|95.10=101631/106873|95.15
##### lstm-tagger模型


(旧数据)

||accuracy|P| R|F
---|---|---|---|---
ctb|95.29| 94.20=47399/50316|94.20=47399/50319|94.20
pku |95.49| 95.97=113499/118265|95.61=113499/118714|95.79
msr |96.76| 96.35=102835/106733|96.22=102835/106873|96.28

（新数据）


||accuracy|P| R|F
---|---|---|---|---
ctb |95.60|94.34=47471/50317 |94.34=47471/50319|94.34
pku |95.43|96.01=113389/118106 |95.51=113389/118714|95.76
msr |96.59| 96.71=103091/106633|96.46=103091/106873|96.58

（张老师数据）

||accuracy|P| R|F
---|---|---|---|---
ctb |95.77|94.88=76988/81139 |94.37=76988/81579|94.62
pku |95.90| 94.88=97752/103028|93.66=97752/104372|94.26
msr |96.63|96.52=102879/106588 |96.26=102879/106873|96.39

# 最新实验结果整理(20170324)
##### CRF序列标注（一个模型）

|dev|   |   |test|  |   |
|---|---|---|---|---|---|
| P     | R     | F    |P    |R   |F
|96.08 = 69040/71854|95.22 = 69040/72505|95.65|97.31=29184/29990|94.80=29184/30786|96.04

##### CRF序列标注（三个模型合并）==模型有误==

|dev|   |   |test|  |   |
|---|---|---|---|---|---|
| P     | R     | F    |P    |R   |F
|90.40 = 70181/77635|96.79 = 70181/72505|93.49|92.39=29703/32149|96.48=29703/30786|94.39


##### lstm tagger【no word-cluster】

|dev|   |   |test|  |   |
|---|---|---|---|---|---|
| P     | R     | F    |P    |R   |F
|95.93 = 69752/72711|96.20 = 69752/72505|96.07|96.54=29405/30460|95.51=29405/30786|96.02




# 标注问题
### 标注不一致的原因：
- 不同的人对规范理解不一样，或者对语言理解不一样，造成同一个字符串在无歧义的情况下，切分方式不一样
- 由于规范无法覆盖或者明确规定所有的语言结构，导致“类似结构”的字符串切分方式不一样，如“中共中央”“越共中央”
- 还有，同一个人在不同阶段对规范或语言的感觉不一样

### 标注不一致的含义（表现形式）：
- 同一个字符串，无歧义的情况下，同一个语料中标注不统一
- 同类型（结构）的字符串，如（代词+量词），同一个语料中处理方式不统一【最难判断和检测】

### CTB中标注问题
-  重量级：切分不一致
- "一日"一词标注不一致
- “再就业”一词标注不一致
- “野牛”在ctb中标注不一致
- ctb将"联合报"作为一个词，而"联合/公报""中国/时报"等划分为两个词
- 最大、最小、最多、最少等标注不一致
- 还有、还是等标注不一致
- 'XX于'等词语例如致力于、收益于、归功于、热衷于等标注不一致，需要具体查看compare
- 下一代 下一场 下一步 切分不一致
- 主教练 切分不一致
- 副XX 切分不一致（如副总统 副经理）
- 变得 切分不一致
- 只有、只是 切分不一致
- 小企业 切分不一致、跨XX，如跨年度、跨世纪、跨国界、跨行业切分不一致

### MSR中标注问题
- 这/那+量词：“这项”拆（例外）
- 野牛/草 应该是“野牛草”切分错误
- "一日"一词标注不一致
- “过多”一词标注不一致
- 'XX于'等词语例如致力于、收益于、归功于、热衷于等标注不一致，需要具体查看compare
- ‘核XX’等名词如：核武器、核电站、核技术、核竞赛、核工业标注存在不一致，大部分合，核竞赛、核技术拆。
- 下一代 下一场 下一步 切分不一致
- 副XX 切分不一致（如副总统 副经理）
- 明显错误：“着重指出”
- 下岗 下岗证 下岗待业：切分不一致
- 只有、只是 切分不一致
- XX者 切分不一致：投资者、吸烟/者、创作/者
- 小企业 切分不一致、跨XX，如跨年度、跨世纪、跨国界、跨行业切分不一致
- 对成语标注不一致：愚公移山 精卫/填/海 夸父逐日

### PKU中标注问题
- 部长级：切分不一致
- 机构组织:“南斯拉夫联盟”拆分，“南联盟”合为一个词，不一致
- 越共/中央、民进/中央、致公党/中央、党中央、中共中央标注不一致（猜想pku采用最大匹配初步分词，然后人来修改，词表中有的词会分出来，否则会切开）
- "一日"一词标注不一致
- 波黑塞族共和国、南斯拉夫联盟共和国、塞族共和国、黑山共和国拆，中华人民共和国、塞尔维亚共和国、乌拉圭东岸共和国合。
- pku中将"再就业"划分为“再/就业”，“再就业率”则作为一个整体划分。（感觉又不是不一致）
- 过多：不一致
- 最大、最小、最多、最少等：不一致
- 还有、还是等标注不一致
- 'XX于'等词语例如致力于、收益于、归功于、热衷于等标注不一致，需要具体查看compare
- 下一代 下一场 下一步 切分不一致
- 重奖 切分不一致
- 偷猎/大象者 切分错误
- 只有、只是 切分不一致
- 小企业 切分不一致、跨XX，如跨年度、跨世纪、跨国界、跨行业切分不一致


# 标注规范整理
成语：先在compare中查找有没有一样的例子，有就按照例子标，没有就遵从模型预测的结果

|  ||CTB|PKU |MSR|
---|---|---|---|---
|时间词 | 2000年1月1日|2000年/1月/1日|同CTB|合
| | 2000年1月1日3点10分 | 2000年/1月/1日/3点/10分 | 同CTB | 合
| |今年下半年、一九九八年下半年、1995年下半年|拆：今年/下半年 |拆：今年/下半年 |合
| |一九九九年一月一日 | 拆| 拆| 合 
| |上午十一时、下午8时、下午两点|拆： 上午/十一时|拆：下午/两点|合
| |几年、近几年|拆“近/几/年”|拆“近/几/年”|合
| | [[[８０]p [年代]p]c [中期]pc]m | | |
数词+量词（个、种、项） | 一个、一位| 拆：一/个 | 合：一个 | 合
|| 三个、三位、10个、四台、首台、一件 |  拆 | 拆 | 合
||33亿元、数十亿元|拆：33亿/元|拆：33亿/元|合
||一年、八十二年、三百年|拆:八十二/年|拆:三百/年|合
|| 十多个 | 十多/个 | 十/多/个 | 合 
|| 整个 | 拆 | 合 | 合
|| 半个 | 拆 | 拆 | 合
||第一个、第一次、第二次| 拆：第一/个 | 拆：第一/个 | 合
最+形容词|最大、最小、最多、最少|拆（不一致）|拆（不一致）|合
这/那+量词 | 这种、这项、这个 | 拆 | 合 | 合 （例外：“这项”全是拆）
|单音节代词“本”“每”“各”“诸”后接名词|后接单音节名词：“各地”"各处""各国"等|拆（不一致）|合（不一致）|合（不一致）
||后接2个音节名词：“各单位”"各地区"|拆：各/单位、各/地区|拆：各/单位、各/地区|合
专有名词后接单音节名词|中国人、东方人、希腊人、欧美人、非洲人|合|拆（不一致）|拆
XXX共和国|中华人民共和国、塞尔维亚共和国、吉尔吉斯共和国、苏联共和国|拆为：塞尔维亚/共和国、吉尔吉斯/共和国（除/中华/人民/共和国）| 拆（存在不一致）|合
 XX联盟|南斯拉夫联盟、南联盟、欧洲联盟、阿拉伯联盟 |拆 | 拆（除了南联盟） | 合
|XX中央| 党中央、中共中央、民进中央、越共中央 | 拆| 拆（除党中央和中共中央） | 合
|XX队|申花足球队、成都五牛队、国际米兰队、鲁能泰山队|未出现|拆“申花/足球队”、“国际/米兰队”|合
||跳水队、羽毛球队、乒乓球队、合唱队|合|合|分
|XX奖|文学奖|拆:文学/奖|合|拆:文学/奖
|XX所|储蓄所、经济所、研究所|拆：储蓄/所|合|拆：储蓄/所
|XX建设|廉政建设、基本建设|拆:廉政/建设|拆：廉政/建设|合
名词+级/界 | 国宝/级 厅/级 世界/级 音乐/界 医学/界 宗教/界 | 合 | 合 | 拆
动词+出 | 拔出 看出 走出 胜出 超出 | 合 | 拆（不一致） | 拆 （不一致）
动词+住|保住、抱住、围住、缠住、守住、粘住、捂住|合|合（除了守住、抱住、粘住）|拆（除了粘住）
XX率| 通胀率 再就业率 录取率 发生率 | 合 | 合 | 拆
    || 市场占有率 | 市场/占有率 | 合 | 市场/占有率
    || 通货膨胀率 | 合 | 合 | 通货膨胀/率
    || 通胀率 | 合 | 合 | 通/胀/率 （可以看出msr不一致）
XX于|负于、用于、处于、有利于、定于、[[囿]pm [于]pm]c|拆（不一致）|拆（不一致）|合（不一致）
XX者|演唱者、钓鱼者、围观者、登山者|合|合|拆（有不一致：投资者）
XX商|广告商、建筑商|合|合|分：广告/商
XX员|销售员、协调员、解说员、专管员、发行员|合|合|拆
XX家|翻译家、改革家|合|合|拆：翻译/家
XX度|满意度、关联度|合|合|拆:满意/度
|每X|每年、每月、每周、每天|拆|合|合
|副XX|副主任、副主席、副部长、副县长|合|拆|拆
|全X|全省、全市、全村、全国|拆|合|合
|XX赛|足球赛、篮球赛、乒乓球赛、羽毛球赛|合|合|拆
XX业|石油业、金融业、通信业、加工业|合|合|拆:石油/业
X工业|轻工业、重工业、核工业|合|拆：轻/工业（注：加工业与其结构不同）|合
|XX性|自觉性、持续性、丰富性、及时性|合|合|拆
跨XX|跨年度、跨世纪、跨国界、跨行业|合（有不一致）|拆（有不一致）|合（不一致，跨学科 跨区 跨国界 拆）
XX污染 | 环境污染 水污染 空气污染）| 拆 （除 水污染）| 拆（除 水污染） 有不一致）| 合 
|XX法|管理法、国防法、组织法、保护法|合|合|拆
|XX化|商业化、电子化、集中化|合|合|拆：商业/化
|XX地|缓缓地、轻轻地|拆：缓缓/地|合|拆：缓缓/地
|XX籍|中国籍、巴西籍、四川籍、英籍|合|合|拆：中国/籍
|XX型|特大型、知识型、开放型|合|合|拆：特大/型
|养X|养猪、养鱼、养鸡|拆|合|拆
X一点|差一点、好一点、淡一点|合|拆：差/一点|合
|令人XX| 令人满意 令人担忧 | 令/人/XX （有不一致） | 合 | 令人/XX 
|核XX|核武器、核电站、核技术、核竞赛、核工业|拆|合|合（存在不一致，查看compare）
XX杯|冠军杯、解放者杯|合|合|拆
XX企业|乡镇企业、合资企业、私营企业、大型企业|拆|合|合
XX集团|海尔集团、联想集团|拆:海尔/集团|拆:海尔/集团|合
XX公园|森林公园、温泉公园|拆：森林/公园|拆：森林/公园|合
XX设计|建筑设计、工程设计|拆：建筑/设计|拆：建筑/设计|合
XX部门|政府部门、人事部门、司法部门|拆|合|拆
标点符号|——— 多个短杠（全角，暂时无法用键盘输入）|合|合|拆'—/—/—'
| | －－－（全角，暂时无法用键盘输入）|拆"－－/－－"|合|拆"－/－/－/－"
|其他高频具体例子| 一步一脚印 | 合 | 未出现 |　未出现
|| 一步一个脚印 | 未出现 | 合 |　合
||访华|拆|合|拆
||还有、还是|拆（不一致）|合（不一致）|合
||只有、只是|拆（标注不一致）|拆（不一致）|合（不一致）
|| 再就业 | 合（不一致） | 再/就业 | 合
||变得|拆（不一致）|拆|合
||特别是|合|拆|合
||小企业|拆|合（不一致）|拆
||某种|拆|合|合

1. [[[退]m [堤]m]c [工程]mc]p
2. 
# 分词规范相关
##### 中文分词十年回顾 2007
- 词表驱动。在相关上下文中未见歧义的情况下，==词表应当作为一个完整的切分单位，不许随意切碎或组合。==
- 把人名、地名、机构名等命名实体和日期、时间等数字表达式的定义纳入分词规范。
- 分词规范的制订与分词语料的标注、审定过程交互进行。

##### 分词规范亟需补充的三方面内容 2007 (msr)
- 命名实体的标注
  -  对于具有上下位关系的地名==按单位分别标注==，如中国/河南省/郑州市/中原路/93号。
  -  ==城市公共设施及地标性建筑物标注为地名（作为一个词语标注）==，如哈尔滨火车站、龙潭湖公园、北京图书馆。
- 表义字串的标注（指文本中的数字表达式，如日期、时间、地址、网址、外文字符、产品型号等）
  -  ==汉字的度量单位与数字串需切分标注==，如670~800/美元。
  -  ==外文字符的度量单位与数字串应整体标注==，如83℃。
  -  ==时间表达式、外文字符、产品型号、地址、网址整体标注==，如S-1240。
- 分词歧义字串及其消解（根据上下文语义切分词）

##### 关于分词规范的探讨 1997
- 分词单位下界
   -  动宾、动补、偏正结构中可扩展的一律切开
   -  二字以上词语的前加成分、后加成分同词干一律切开
   -  儿化音的“儿”同前面的词一律切开
   -  二字以上地名的通名与专名一律切开
   -  “月”“星期”“礼拜”同数字切开
   -  重叠的动词一律切开
- 分词单位上界
   -  简单动宾、动补、形宾、形补等，时间短语，处所短语，数量名短语，地名上下级全称，人名全称，机构名全称，商品名全称，术语全称等。
- 上下界之间的分词单位及其内部结构
- 基本词表（收集内部不可切分之词）


##### 《“信息处理用现代汉语分词规范”》的若干问题探讨 1989
- 信息处理所说的词和语言学中的词存在明显区别。信息处理倾向于尽可能把==表示一个完整概念的词语当做一个词，而不管它是词还是语（词组）==。引入术语“分词单位”。
- 分词规范参考汉语拼音正词法。如北京饭店（Beijing Fandian）、平分秋色（pingfen-qiuse）【但是感觉汉语拼音正词法现在并不常用】
- 分词规范要符合民族语用心理习惯。如一/月、星期/六、元月、星期日
 

# 最新数据整理
#### ctb5big/msr/pku126转化前数据：
- dir:172/disk4t/cgong/WS/data/multi-grain-data/ctb5big/ctb5big-seg
- dir:172/disk4t/cgong/WS/data/multi-grain-data/msr/msr-seg
- dir:172/disk4t/cgong/WS/data/multi-grain-data/pku126/pku126-seg
#### ctb5big/msr/pku126转化后数据：
- dir:172/home/cgong/WS/data/all-three-data
#### 微博数据：
- 序列标注测试数据:gpu/data/xzjiang/weibo-data/sequence-test
- span-parser测试数据:gpu/data/xzjiang/weibo-data/span-parser-test
# 评价PRF值程序目录
##### 微博
- dir:gpu/data/xzjiang/evaluation/script/eval.sh
##### msr & ctb5big & pku126
- dir:175/home/jwsun/span_parser/evaluate-prf
# 多粒度数据的分布
##### 微博

去除频率=1

单粒度 | 2粒度 | 3粒度 | 4粒度 | 5粒度
---|---|---|---|---|
8230/11065|2638/11065|175/11065|19/11065|3/11065


##### msr & ctb5big & pku126

||单粒度 | 2粒度|3粒度
---|---|---|---
|msr-test|9723/13696 | 3877/13696|96/13696
|ctb-test|7314/9500 | 2150/9500|36/9500
|pku-test|11740/16419|4581/16419|98/16419




# 最新实验结果整理

##### CRF序列标注（一个模型）
- train-dir:172/home/cgong/WS/exp/multi-grain-exp/baseline-alltrain-alldev
- test-dir:
- [ ] ++1500dev++:172/home/cgong/WS/exp/multi-grain-exp/test-baseline-alldata-merge-1500dev 
- [ ] ++weibo++:172/home/cgong/WS/exp/multi-grain-exp/test-baseline-alldata-merge-weibo
- [ ] ++msr & pku126 & ctb5big测试集++:172/home/cgong/WS/exp/multi-grain-exp/test-baseline-alldata-merge-msr_pku_ctb_test
- [ ] ++ctb&pku&msr开发集++:172/home/cgong/WS/exp/multi-grain-exp/test-baseline-alldata-merge-ctb_pku_msr_dev/
- [ ] ++人工标注测试集++:172/home/cgong/WS/exp/multi-grain-exp/test-baseline-alldata-merge-mannual
- train1:172/home/cgong/WS/data/all-three-data/msr-train
- train2:172/home/cgong/WS/data/all-three-data/ctb-train
- train3:172/home/cgong/WS/data/all-three-data/pku-train
- dev:172/home/cgong/WS/data/all-three-data/dev-data
- src:172/home/cgong/WS/src/src-ws-pos-r4-support-B_I_I_new


||P | R | F | 相同 | 交叉（句子） | 答案
---|---|---|---|---|---|---
1500dev|96.44|95.66|96.05|-|-|-
微博（去除词率=1）|92.43| 54.33 | 68.44 | 15894 | 1301 | 29254
msrtest|96.83 = 117947/121804|95.41 = 117947/123617|96.12|-|-|-
ctbtest|95.89 = 51843/54065|95.18 = 51843/54471|95.53|-|-|-
pkutest|96.91 = 128559/132655|96.20 = 128559/133631|96.56|-|-|-
ctb&pku&ctb开发集|96.08 = 69040/71854|95.22 = 69040/72505|95.65|-|-|-



##### CRF序列标注（三个模型合并）==模型有误==
1. ctb5big模型：
- train-dir:172/home/cgong/WS/exp/multi-grain-exp/baseline-ctb5big
- test-dir:
- [ ] ++1500dev++:172/home/cgong/WS/exp/multi-grain-exp/test-baseline-ctb5big-1500dev
- [ ] ++weibo++:172/home/cgong/WS/exp/multi-grain-exp/test-baseline-ctb5big-weibo
- [ ] ++msr & pku126 & ctb5big测试集++:172/home/cgong/WS/exp/multi-grain-exp/test-baseline-ctb5big-msr_pku_ctb_test
- [ ] ++ctb&pku&msr开发集++:/home/cgong/WS/exp/multi-grain-exp/test-baseline-ctb5big-ctb_pku_msr_dev
- [ ] ++人工标注测试集++:/home/cgong/WS/exp/multi-grain-exp/test-baseline-ctb5big-mannual
- dev:172/home/cgong/WS/data/all-three-data/ctb-dev
- src:172/home/cgong/WS/src/src-ws-pos-r4-support-B_I_I_new

2. pku126模型：
- train-dir:172/home/cgong/WS/exp/multi-grain-exp/baseline-pku126
- test-dir:
- [ ] ++1500dev++:172/home/cgong/WS/exp/multi-grain-exp/test-baseline-pku126-1500dev
- [ ] ++weibo++:172/home/cgong/WS/exp/multi-grain-exp/test-baseline-pku126-weibo
- [ ] ++msr & pku126 & ctb5big测试集++:172/home/cgong/WS/exp/multi-grain-exp/test-baseline-pku126-msr_pku_ctb_test
- [ ] ++ctb&pku&msr开发集++:/home/cgong/WS/exp/multi-grain-exp/test-baseline-pku126-ctb_pku_msr_dev
- [ ] ++人工标注测试集++:/home/cgong/WS/exp/multi-grain-exp/test-baseline-pku126-mannual
- train1:172/home/cgong/WS/data/all-three-data/pku-train
- dev:172/home/cgong/WS/data/all-three-data/pku-dev
- src:172/home/cgong/WS/src/src-ws-pos-r4-support-B_I_I_new

3. msr模型：
- train-dir:172/home/cgong/WS/exp/multi-grain-exp/baseline-msr
- test-dir:
- [ ] ++1500dev++:172/home/cgong/WS/exp/multi-grain-exp/test-baseline-msr-1500dev
- [ ] ++weibo++:172/home/cgong/WS/exp/multi-grain-exp/test-baseline-msr-weibo
- [ ] ++msr & pku126 & ctb5big测试集++:172/home/cgong/WS/exp/multi-grain-exp/test-baseline-msr-msr_pku_ctb_test
- [ ] ++ctb&pku&msr开发集++:/home/cgong/WS/exp/multi-grain-exp/test-baseline-msr-ctb_pku_msr_dev
- [ ] ++人工标注测试集++:/home/cgong/WS/exp/multi-grain-exp/test-baseline-msr-mannual
- train1:172/home/cgong/WS/data/all-three-data/msr-train
- dev:172/home/cgong/WS/data/all-three-data/msr-dev
- src:172/home/cgong/WS/src/src-ws-pos-r4-support-B_I_I_new


||P | R | F | 相同 | 交叉 | 答案
---|---|---|---|---|---|---
1500dev|91.54|97.04|94.21|-|-|-
微博（去除词率=1）|85.31|69.77|76.76|20411|3515|29254
msrtest|92.12 = 119650/129886|96.79 = 119650/123617|94.40|-|-|-
ctbtest|89.53 = 52644/58800|96.65 = 52644/54471|92.95|-|-|-
pkutest|91.07 = 130973/143812|98.01 = 130973/133631|94.41|-|-|-
ctb&pku&ctb开发集|90.40 = 70181/77635|96.79 = 70181/72505|93.49|-|-|-




##### lstm tagger【add word-cluster】
- train-dir:175/home/jwsun/span_parser/v5-bies-addtag
- test-dir:gpu/data/cgong/exp-mws/v6.1-LSTM-tagger-add-wordcluster
- [ ] ++人工标注测试集++:gpu/exp-mws/v6.1-LSTM-tagger-add-wordcluster
- [ ] ++ctb&pku&msr开发集++:gpu/home/data/cgong/exp-mws/v6-LSTM-tagger
- src:175/home/jwsun/span_parser/v5-bies/src
- src:175/home/jwsun/span_parser/v5-bies-addtag/src
- train1:175/home/jwsun/span_parser/all-three-data/msr-train-tag
- train2:175/home/jwsun/span_parser/all-three-data/ctb-train-tag
- train3:175/home/jwsun/span_parser/all-three-data/pku-train-tag
- dev:175/home/jwsun/span_parser/all-three-data/dev-data-tag


||P | R | F | 相同 | 交叉 | 答案
---|---|---|---|---|---|---
1500dev|96.54 = 38225/39594|95.93 = 38225/39846|96.24|-|-|-
微博（去除词率=1）|96.59|57.94|72.43|16950 |599|29254
msrtest|96.77 = 118653/122611|95.98 = 118653/123617|96.38|-|-|-
ctbtest|95.15 = 51732/54371|94.97 = 51732/54471|95.06|-|-|-
pkutest|96.40 = 128612/133409|96.24 = 128612/133631|96.32|-|-|-
ctb&pku&ctb开发集|96.02 = 69155/72022|95.38 = 69155/72505|95.70|-|-|-



##### lstm tagger【no word-cluster】
- train-dir:175/home/jwsun/span_parser/v5-bies
- test-dir:gpu/data/cgong/exp-mws/v6-LSTM-tagger
- [ ] ++人工标注测试集++:gpu/data/cgong/exp-mws/v6-LSTM-tagger
- [ ] ++ctb&pku&msr开发集++:gpu/home/data/cgong/exp-mws/v6-LSTM-tagger
- src:175/home/jwsun/span_parser/v5-bies/src
- train1:175/home/jwsun/span_parser/all-three-data/msr-train
- train2:175/home/jwsun/span_parser/all-three-data/ctb-train
- train3:175/home/jwsun/span_parser/all-three-data/pku-train
- dev:175/home/jwsun/span_parser/all-three-data/dev-data


||P | R | F | 相同 | 交叉 | 答案
---|---|---|---|---|---|---
1500dev|96.45|96.64|96.54|-|-|-
微博（去除词率=1）|97.41|59.89|74.18|17520|465|29254 
msrtest|96.79 = 119815/123790|96.92 = 119815/123617|96.86|-|-|-
ctbtest|94.59 = 52002/54976|95.46 = 52002/54471|95.03|-|-|-
pkutest|96.47 = 129543/134282|96.94 = 129543/133631|96.71|-|-|-
ctb&pku&ctb开发集|95.93 = 69752/72711|96.20 = 69752/72505|96.07|-|-|-




##### span-parser  【bi-char & no-regulation & add-word-cluster】
gpu:/data/xzjiang/span-parser/bi-drop-3file 

||P | R | F | 相同 | 交叉 | 答案
---|---|---|---|---|---|---
1500dev|95.63|96.50|96.61|-|-|-
微博（去除词率=1）|97.01 | 62.77 | 76.23 | 18365 | 566 | 29254
msrtest|96.88=119409/123254|96.60=119409/123617|96.74|-|-|-
ctbtest|95.27=52093/54679|95.63=52093/54471|95.45|-|-|-
pkutest|96.69=129500/133935|96.91=129500/133631|96.80|-|-|-


##### span-parser  【bi-char & no-regulation & no word-cluster】
       
  
gpu:/data/cgong/v7-no-tagger-bi-char-drop-3-files  
gpu:/data/cgong/gpu:/data/cgong/v7-no-tagger-bi-char-drop-3-files

||P | R | F | 相同 | 交叉 | 答案
---|---|---|---|---|---|---
1500dev|96.85|96.36|96.60|-|-|-
微博（去除词率=1）|96.88 | 59.86 | 73.99 |17510  |564  |29254 



##### span-parser  【uni-char & no-regulation】
  
gpu:/data/xzjiang/span-parser/uni-drop-3file

||P | R | F | 相同 | 交叉 | 答案
---|---|---|---|---|---|---
1500dev|96.25|95.48|95.86|-|-|-
微博（去除词率=1）|93.57 | 55.24 | 69.47 | 16160 | 1111 | 29254
msrtest|96.45=117999/122343|95.46=117999/123617|95.95|-|-|-
ctbtest|94.77=51594/54439|94.72=51594/54471|94.75|-|-|-
pkutest|96.55=128635/133231|96.26=128635/133631|96.41|-|-|-



### 2016/1/14
# 实验结果整理

CRF序列标注（一个模型）


||P | R | F | 相同 | 交叉 | 答案
---|---|---|---|---|---|---
微博（全部）|89.96|46.92|61.67|19935|2225|42486
微博（去除词率=1）|92.26| 54.00 | 68.13 | 15880 | 1332 | 29407
1500dev|96.44|95.66|96.05|-|-|-

CRF序列标注（三个个模型合并）==模型有误==


||P | R | F | 相同 | 交叉 | 答案
---|---|---|---|---|---|---
微博|82.63|61.49|70.51|26126|5492|42486
微博（去除词率=1）|85.16|69.39|76.47|20404|3556|29047
1500dev|91.54|97.04|94.21|-|-|-

lstm tagger


||P | R | F | 相同 | 交叉 | 答案
---|---|---|---|---|---|---
微博|95.03|52.13|67.33|22147|1158|42486
微博（去除词率=1）|97.24|59.54  |73.86 |17509  |497  |29407 
1500dev|96.45|96.64|96.54|-|-|-

span-parser  
bi-char  
no-regulation


||P | R | F | 相同 | 交叉 | 答案
---|---|---|---|---|---|---
微博（全部）|94.81|53.83|68.67|22869|1253|42486
微博（去除词率=1）|96.81 | 62.40 | 75.89 | 18350 | 605 | 29407
1500dev|95.63|96.50|96.61|-|-|-

span-parser  
uni-char  
no-regulation


||P | R | F | 相同 | 交叉 | 答案
---|---|---|---|---|---|---
微博（全部）|90.70|47.60|62.44|20224|2073|42486
微博（去除词率=1）|93.36 | 54.91 | 69.15 | 16148 | 1149 | 29407
1500dev|96.25|95.48|95.86|-|-|-

---
#### 加规则：
span-parser  
uni-char  
add-regulation

||P | R | F | 相同 | 交叉 | 答案
---|---|---|---|---|---|---
微博（全部）|91.28|43.36|58.79| 18420 |1760|42486
微博（去除词率=1）|93.20 | 51.97 | 66.73 | 15284 | 1116 | 29407
1500dev|96.10|94.89|95.49|-|-|-
      
span-parser  
bi-char  
add-regulation

||P | R | F | 相同 | 交叉 | 答案
---|---|---|---|---|---|---
微博（全部）|95.87|49.01|64.86| 20824|898|42486
微博（去除词率=1）|97.33 | 58.60 | 73.15 | 17232 | 472 | 29407
1500dev|97.00|96.16|96.58| -|-|-
---
### 2016/1/14
##### 神经网络序列标注

==94.44(44/65)==

---

### 2016/1/6
##### add bichar

||P| R|F
---|---|---|---|
|val|96.11%(35/46) | 96.15%(35/46)|96.13%(35/46)


##### add bichar+lenfeature+elefeature

||P| R|F
---|---|---|---|
|val|95.77%(38/45) | 96.21%(38/45)|95.99%(38/45)

### 2016/12/16
#### span parser
1. 不添加约束
- dir:xzjiang@m175 ~/new/no_exploration
- code:xzjiang@m175 ~/new/no_exploration/src
- 

|  |P| R|F
---|---|---|---|
val|96.35%(15/25) | 95.63%(15/25)|95.99%(15/25)
2. 添加规则，不修改loss值
- dir:xzjiang@m175 ~/lab/add_regulation_50type
- code:xzjiang@m175 ~/lab/add_regulation_50type/src


||P| R|F
---|---|---|---|
|val|95.82%(22/23) | 94.77%(22/23)|95.29%(22/23)

3. 添加规则，修改loss值

- dir:xzjiang@m175 ~/lab/add_regulation_modify_loss
- code:xzjiang@m175~/lab/add_regulation_modify_loss/src

||P| R|F
---|---|---|---|
|val|96.23%(27/30) | 95.63%(27/30)|95.93%(27/30)


---


### 2016/12/15
#### CRF分词模型（msr,ctb5big,pku126三个规范）
- tag:BIES_BIES_BIES
- dir:172 ~/WS/exp/multi-grain-exp/baseline-msr-no-merge
- code:172/disk4t/cgong/WS/src/src-ws-pos-r2/
- inst-num-from-train-1-one-iter=10000


==95.13/581==

msr: 
P | R|F|
---|---|---|
0.9733| 0.9581|0.9656

---

### 2016/11/24
#### 分词标注出现交叉时的规律总结：
1. 遇到数字时，CTB数据集分词错误的概率较大。（如“19时03分”分词结果为“19/时03/分”）
2. 当遇到“xxxx者”、"xxxx长"、"xxxx人"时，msr分词错误概率较大。（如“财政部长”分词结果为"财政部/长"）
3. 人名、地名、专有名词
4. 出现括号、书名号
5. 带“们”的词容易出现交叉，如“小朋友们”的分词结果有“小/朋友们”和“小朋友/们”
6. ABB或AAB式的叠词容易分词错误，如“一点点”分词结果为“一点/点”
7. 出现成语时容易分词错误
8. 出现英文单词时容易分词错误
9. 出现复合单位，如“亿千瓦时”、“千公里”时容易分词错误

#### 出现交叉且两种分词结果都正确的特例：
- 全国人大代表-> 全国/人大代表和全国人大/代表
- 不利于->不/利于和不利/于

### 2016/11/22
#### CRF分词模型(msr,ctb5big,pku126三个规范)
- tag:BIES_BIES_BIES（若标记相同则只记录一个，如：苏/B 州/E_I 大/B_I 学/E）
- dir:172 ~/WS/exp/multi-grain-exp/baseline-ctb5big
- code:172/disk4t/cgong/WS/src/src-ws-pos-r2/
- inst-num-from-train-1-one-iter=10000

ctb5big: ==93.382(332/363)==


### 2016/11/17
包含交叉分词的句子数统计（以句子为单位统计）

|  |ctb5big|pku126|msr
---|---|---|---
|train|197/16091（标完）|416/46815（标完）|1920/85918(标完21%)
|dev|9/803（标完）| 17/2000（标完）|5/1000（标完）
|test|22/1910（标完）| 39/5000（标完）|57/3985（标完）

包含交叉分词的句子数统计（以词语为单位统计）

|  |ctb5big|pku126|msr
---|---|---|---
|train|0.04%=212/480550|0.05%=525/1103582|0.1%=2804/2776796

---
### 2016/11/9
##### 1.msr->ctb5big

|          | train|
-----------|------|
msr粗粒度词语比例 |151997 / 195313 = 0.778|
ctb5big粗粒度词语比例|43316 / 195313 = 0.222|


##### 2.ctb5big->msr

|          | train|
-----------|------|
msr粗粒度词语比例 |24675 / 28872 = 0.855|
ctb5big粗粒度词语比例|4197 / 28872 = 0.145|

##### 3.msr->pku126

|          | train|
-----------|------|
msr粗粒度词语比例 |104922 / 142558 = 0.736
pku126粗粒度词语比例|37646 / 142558 = 0.264

##### 4.pku126->msr

|          | train
-----------|------
msr粗粒度词语比例 |45401 / 57549 = 0.789
pku126粗粒度词语比例|12148 / 57549 = 0.211

##### 5.pku126->ctb5big

|          | train|
-----------|------|
ctb5big粗粒度词语比例 |11046 / 45989 = 0.240
pku126粗粒度词语比例|34943 / 45989 = 0.760

##### 6.ctb5big->pku126

|          | train|
-----------|------|
ctb5big粗粒度词语比例 |1985 / 14578 = 0.136
pku126粗粒度词语比例|12593 / 14578 = 0.864
---

### 2016/10/29
#### msr/ctb5big/pku126相互转化
- 不同分词：指两数据集中分词结果不同（如"中国人"在msr中分词结果为"中国人"，在ctb5big中分词结果为"中国/人"）
- 交叉分词：如“中国人”在一个数据集中的分词结果为“中国/人”，在另一个数据集中的分词结果为“中/国人”，其中“中国”和“国人”被交叉分词。
##### 1.msr->ctb5big

|          | train|dev   |test
-----------|------|------|---
不同分词数 |195313|1146   |9057
平均每句不同分词数 |2.27|1.15 |2.27
交叉分词数 |4     |0 |0

##### 2.ctb5big->msr

|          | train|dev   |test
-----------|------|------|---
不同分词数 |28872|1442   |3637
平均每句不同分词数 |1.79|1.80 |1.90
交叉分词数 |0     |0 |0

##### 3.msr->pku126

|          | train|dev   |test
-----------|------|------|---
不同分词数 |142558| 551  |6148
平均每句不同分词数 |1.66|0.55 |1.55
交叉分词数 |7     | 0|1

##### 4.pku126->msr

|          | train|dev   |test
-----------|------|------|---
不同分词数 |57549|2502  |6927
平均每句不同分词数 |1.23|1.25|1.39
交叉分词数 |  0  | 0|0

##### 5.pku126->ctb5big

|          | train|dev   |test
-----------|------|------|---
不同分词数 |45989 |2014 |5040
平均每句不同分词数 |0.98|1.01|1.01
交叉分词数 |2   |0|0

##### 6.ctb5big->pku126

|          | train|dev   |test
-----------|------|------|---
不同分词数 |14578| 664 |1665
平均每句不同分词数 |0.91|0.82|0.87
交叉分词数 |5  |0|0
---
### 2016/10/29
#### msr-coupled-with-ctb5big & msr-coupled-with-pku126实验记录
##### 1. 准备offline filter data:
- msr:
> tag A:  msr正确答案（BIES@x）【172/home/cgong/WS/data/multi-grain-data/msr/msr-seg】  

> tag B: 在ctb5big/pku126模型上分析msr的结果）【172/home/cgong/WS/exp/multi-grain-data-prepare-exp/ctb5big-take_msr_as_test】【172/home/cgong/WS/exp/multi-grain-data-prepare-exp/pku126-take_msr_as_test】
- pku126/ctb5big: 
> tag A: 在msr上模型分析pku126/ctb5big的结果 【172/home/cgong/WS/exp/multi-grain-data-prepare-exp/msr-take_ctb5bigtrain_as_test】【172/home/cgong/WS/exp/multi-grain-data-prepare-exp/msr-take_pku126train_as_test】

> tag B: pku126/ctb5big正确答案【172/home/cgong/WS/data/multi-grain-data/pku126/pku126-seg】【172/home/cgong/WS/data/multi-grain-data/ctb5big/ctb5big-seg】

##### 2. 跑实验:
172/disk4t/cgong/WS/model/coupled-model
###### msr-coupled-with-ctb5big (done)
- dir:176/home/jwsun/WS/exp/multi-grain-exp/msr-coupled-ctb5big
- code:176/home/jwsun/WS/src/src-r6-ws-offline-filter/
- tag-online-filter-lambda=0.995
- tag-online-filter-maxnum=16
- inst-num-from-train-1-one-iter=5000
- inst-num-from-train-2-one-iter=5000

|msr                | ctb5big
|-------------------|------------------
|==95.66（374/395）== | ==90.63(162/395)==

###### msr-coupled-with-pku126(done)
- dir:176/home/jwsun/WS/exp/multi-grain-exp/msr-coupled-pku126
- code:176/home/jwsun/WS/src/src-r6-ws-offline-filter/
- tag-online-filter-lambda=0.995
- tag-online-filter-maxnum=16
- inst-num-from-train-1-one-iter=5000
- inst-num-from-train-2-one-iter=5000

|msr                | pku126               
|-------------------|------------------
|==96.76（194/215）== | ==92.22(187/215)==     
   

 



###### baseline

|msr              | ctb5big          |pku126
------------------|------------------|------------------
==95.33(919/970)==|==90.70(207/238)==|==92.11(214/245)==



---


### 2016/10/9
#### outlines
- 多粒度分词任务：将一个句子分别按粗细两种粒度进行分词。如：输入句子“我是中国人”，可分别获得粗粒度分词结果：“我/是/中国人”和细粒度分词结果：“我/是/中国/人”。
- dir:172/WS
- data:ctb5big+pku126
- 不同的分词个数：257135
#### baseline
##### 多粒度
- dir:/disk4t/cgong/model/seg-multigrain-ctb5big-pku126-baseline
- code:172/disk4t/cgong/WS/src/src-ws-pos-r2
- one-iter-max :10000
- inst-max-num-eval=1000
- tag:细粒度(BIES)^粗粒度(BIES)(一个标记)




==best:96.65(it=555)== DONE

==evaluate==
- testfile:172/home/cgong/WS/processconll/getDifferentGrains/ctbtest-grain.conll
- 总共200147个字

|  |P  | R |F
---|---|---|---
细粒度|0.978 | 0.971|0.974
粗粒度|0.976|0.972|0.974
joint|0.980|0.970|0.974
##### 单粒度（CTB5big）
- dir:172/disk4t/cgong/WS/model/ctb5-big-bs
- code:172/disk4t/cgong/WS/src/src-ws-pos-r2
- one-iter-max :10000
- inst-max-num-eval=1000
- tag:BIES

==best:95.55(it=273)==  DONE

 ##### 单粒度（pku126）
- dir:172/disk4t/cgong/WS/model/pku126-bs
- code:172/disk4t/cgong/WS/src/src-ws-pos-r2
- one-iter-max :10000
- inst-max-num-eval=1000
- tag:BIES

==best:97.04(it=516)== DONE  
#### coupled(不区分CTB和PKU)
- dir:172/disk4t/cgong/WS/exp/coupled-seg-grain-ctb5big-pku126
- code:172/WS/src-coupled-map/src-r5-same-feature-with-wstagger-src-r
- one-iter-max :10000
- inst-max-num-eval=1000
- tag:细粒度(BIES)^粗粒度(BIES)(两个标记)


joint            |细粒度dev          | 粗粒度dev 
-----------------|-------------------|------------------
==96.23(it=458)==|==96.92（it=458）==| ==96.94(it=458)==

==evaluate==
- testfile:172/home/cgong/WS/processconll/getDifferentGrains/ctbtest-grain.conll
- 总共200147个字

|  |P  | R |F
---|---|---|---
细粒度|0.976 | 0.968|0.972
粗粒度|0.973|0.970|0.972
joint|0.978|0.967|0.972




