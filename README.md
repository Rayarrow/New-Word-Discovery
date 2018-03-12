# New Words Discovery

## Setup

Requirements:

```
pandas
numpy
jieba
scipy
nltk
```

This implementation tries to discover four types of new words based on four parameters.

Four types of new words:

 1. latin words, including 
    
    1. pure digits (2333, 12315, 12306)
    
    2. pure letters (iphone, vivo)
    
    3. a mixture of both (iphone7, mate9)
    
 2. 2-Chinese-character unigram (unigrams are defined as the elements produced by the segmentator): 
 
    (马蓉，优酷，杨洋)
    
 3. 3-Chinese-character unigram:
 
    (李易峰，张一山，井柏然)
    
 4. bigrams, which are composed of two unigrams:
 
    (图片大全，英雄联盟，公交车路线，穿越火线)
    
    
Four parameters:

 1. term frequency (tf): The occurrences of a word. A larger `tf` indicates a larger confidence of the following 3 paramters.
 
 2. aggregation coefficient: A larger `agg_coef` indicates a larger possibility of the co-occurrence of the two words.
 
 $$
 \text{agg_coef} = \frac{P(w1, w2)}{P(w1) \times P(w2)} = \frac{C(w1, w2) / \text{#nr_of_bigrams}}{ (C(w1) / \text{#nr_of_bigrams}) \times (C(w2) / \text{#nr_of_bigrams})}
 $$
 
 where $C(w_1, w_2)$ indicates the counts of the pattern that $w_1$ is followed by $w_2$.
 
 $C(w_1)$ and $C(w_2)$ indicate the count of the counts of $w_1$ and $w_2$ respectively.
 
 3. minimum neighboring entropy
 
 4. maximum neighboring entropy
 
 The minimum and maximum neighboring entropy are the minimum and maximum of left neighboring entropy and right neighboring entropy respectively.
 
 A larger neighboring entropy of a word $w$ indicates that $w$ collocates with mores possible words, which in turn indicates that $w$ is an independent word. For instance, "我是" has a large `tf` and a large `agg_coef` but a small `minimum neighboring entropy` so it's not a word.
 
left entropy:

$$
Ent_{w} = -\sum_{w_l} P(w_l) log(P(w_l))
$$

where $w_l$ are the set of unigrams that appear to the left of word $w$. This above-mentioned formula also applies to the right neighboring entropy.

## Usage
 
An execution script example (Note that the double quotes cannot be omitted if the path you provided contains spaces):

```
python run_discover.py "G:\Documents\Exp Data\CCF_sogou_2016\sogouu8.txt" "G:\Documents\Exp Data\CCF_sogou_2016\reports" --latin 50 0 0 0 --bigram 20 80 0 1.5 --unigram_2 20 40 0 1 --unigram_3 20 41 0 1 --iteration 2 --verbose 2
```

Run

```
python run_discover.py
```

for further information and help.

Each iteration includes the following 11 steps:

1. cutting
2. counting characters
3. counting unigrams
4. counting bigrams 
5. counting trigrams 
6. calculating aggregation coefficients (for unigrams)
7. counting neighboring words (for unigrams)
8. Calculating boundary entropy (for unigrams)
9. calculating aggregation coefficients (for bigrams)
10. counting neighboring words (for bigrams)
11. calculating boundary entropy (for bigrams)

After each iteration, you will get four files reporting new words of type latin, 2-Chinese-character words, 3-Chinese-character words and bigram respectively. After the program exits, you will get four files which respectively merge each type of new words generated from each iteration.

If you encounter any problems, feel free to open an issue or contact me (rayarrow@qq.com).


====================================分隔线================================

# 新词发现

根据四个参数发现四种类型的新词。

四种类型的新词：

 1. 拉丁词，包括： 
    
    1. 纯数字 (2333, 12315, 12306)
    
    2. 纯字母 (iphone, vivo)
    
    3. 数字字母混合 (iphone7, mate9)
    
 2. 两个中文字符的unigram (unigrams被定义为分词器产生的元素): 
 
    (马蓉，优酷，杨洋)
    
 3. 三个中文字符的unigram unigram:
 
    (李易峰，张一山，井柏然)
    
 4. bigrams, 每个bigram由两个unigram组成
 
    (图片大全，英雄联盟，公交车路线，穿越火线)
    
    
四个参数：

 1. 词频 (tf): 一个词出现的次数。词频越大，表明下面三个参数的置信度越高。
 
 2. 凝聚系数: 凝聚系数越大表明两个（字）词共同出现的概率越大（越不是偶然）。
 
 $$
 \text{agg_coef} = \frac{P(w1, w2)}{P(w1) \times P(w2)} = \frac{C(w1, w2) / \text{#nr_of_bigrams}}{ (C(w1) / \text{#nr_of_bigrams}) \times (C(w2) / \text{#nr_of_bigrams})}
 $$
 
 其中$C(w_1, w_2)$是词$w_1$和$w_2$共同出现的次数。
 
 $C(w_1)$和$C(w_2)$是词$w_1$和$w_2$分别出现的次数。
 
 3. 最小边界信息熵
 
 4. 最大边界信息熵

 最小和最大边界信息熵分别是左边界信息熵和右边界信息熵二者的最小值和最大值。
 
 边界信息熵越大，表明一个词越能和更多词搭配，进而表明一个词是一个独立词。比如"我是"拥有大词频和大凝聚系数但是最小边界信息熵却很小，说明它不是一个词。
 
左边界信息熵:

$$
Ent_{w} = -\sum_{w_l} P(w_l) log(P(w_l))
$$

其中$w_l$是出现在$w$左边的所有unigram组成的集合，上面的公式同样适用于右边界信息熵的计算。

## How-to
 
其中一个运行示例（注意如果路径中有空格那么两端的双引号不可省略）

```
python run_discover.py "G:\Documents\Exp Data\CCF_sogou_2016\sogouu8.txt" "G:\Documents\Exp Data\CCF_sogou_2016\reports" --latin 50 0 0 0 --bigram 20 80 0 1.5 --unigram_2 20 40 0 1 --unigram_3 20 41 0 1 --iteration 2 --verbose 2
```

运行

```
python run_discover.py
```

来获取更多帮助。

每次迭代包含以下11个步骤：

1. cutting
2. counting characters
3. counting unigrams
4. counting bigrams 
5. counting trigrams 
6. calculating aggregation coefficients (for unigrams)
7. counting neighboring words (for unigrams)
8. Calculating boundary entropy (for unigrams)
9. calculating aggregation coefficients (for bigrams)
10. counting neighboring words (for bigrams)
11. calculating boundary entropy (for bigrams)

每次迭代之后会产生4个文件分别报告拉丁新词，两个中文的unigram新词，三个中文的unigram新词和bigram新词。程序运行结束后，你会额外得到4个文件，每个文件是一个类型的新词，由之前每次迭代的结果综合而成。

如果遇到任何问题，欢迎提出issue或者联系我 (rayarrow@qq.com).
