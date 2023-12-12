# HW3 中文轉台羅拼音翻譯

### Relative
> Kaggle : https://www.kaggle.com/competitions/machine-learning-2023nycu-translation/leaderboard

> Colab : https://colab.research.google.com/?authuser=0#create=true

> Drive : https://drive.google.com/drive/folders/1Ug8YAAPknDUBSL8ZY9wjc3nmzK7cBdNF?usp=sharing

## Target
給予中文和對應的台羅拼音，利用以上資料去訓練出中文轉台羅拼音的translation model !!!

> *後面有寫出分數高的人作法，可直接跳到後面*

## Final Result
Kaggle 最後得出來的分數(沒有很高)
* public : 9.75520

* private : 9.36971

## File Execution (bash HW3.sh)
```bash
python NLP_ZH_Tailo.ipynb
```

## 檔案內容
```
└─translation
        submission.csv
        test-ZH-nospace.csv
        train-TL.csv
        train-ZH.csv
```

## Initial Idea
一開始的想法就是想說使用Transformer 去跑但是之前小人我沒做過所以先去網路上看了教學手刻了一個，最後訓練出來的結果很糟....

### Dictionary process
一開始想法使用詞來建立字典(包含BOS、EOS)，但是就會發生測試集的資料都沒看過，會有很多錯誤。
``` text=
ex:

"Obama 大勝 美國 頭一位 烏人 總統" -> [Obama,大勝,美國,頭一位,烏人,總統]

```

(手刻的資訊在 Reference-1，那作者講transformer的實作部分很細，強烈建議看)

## Second Test
想說直接使用pytorch 現成的nn.Transformer 去 train，但是結果不知道為何輸出的結果都長一樣，debug了好幾次找不出問題，在此做個紀錄。

``` text=
ex:

"Obama 大勝 美國 頭一位 烏人 總統" -> aaaa bbbb cccc dddd eeee

"駐美 特派員 曹郁芬 華府 報導" -> aaaa bbbb cccc dddd eeee
```
### Dictionary process
直接使用現成的token encode去做(包含BOS、EOS)，這個其實可以，但是模型輸出一直很怪，無法得知字典好壞。

## Final Test

由於台羅拼音是由英文和數字所組成的，所以想說直接使用現成的中文轉英文訓練好的model(Reference-3)，在做pretrain，概念蠻類似的，個人感覺還不錯，最後train了10個epoch就交了 !!!


### Dictionary process
改用別人訓練好的模型的字典(包含BOS、EOS)，所以直接套用即可，這裡字典encode部分改用一個字一個字做去了(效果不錯)。

> *以上是朋友的建議與參考，感恩我親愛的朋友!!*

## Training Loss
![img](https://github.com/ChouGiGiNYCU/Machine_Learning_HW_Fall/blob/main/img/Transformer_loss.jpg)

### 同儕高手做法
(可參考)

第一位
```
使用RNN Encoder、Decoder ，訓練了 100 epoch(實際看loss function 50就差不多了)

資料處理:

中文英文猜開、訓練集有空格利用特殊的token帶入(<SPACE>)，也是一個一個字的字典。

翻譯的時後，有英文的話，先丟入英文前面的句子做翻譯，再丟入後面的句子，最後做Combine。

總結 : 此翻譯方法不太適合和語言翻譯，因為字跟字會有前後文關係，但是此中翻台羅沒有問題而已。
```

第二位、第三位
```
個人認為大同小異，他們是使用Transformer 去訓練(不知道train幾次)，字典也是一個一個字去建立的(空格也是一個字喔)，至於Transformer怎麼做出來的請看 Reference-2。
```
## Reference

1. [Transformer 參考(強烈建議新手看)](https://leemeng.tw/neural-machine-translation-with-transformer-and-tensorflow2.html)

2. [pytorch 官網](https://pytorch.org/tutorials/beginner/translation_transformer.html)

3. [NLP 大量的翻譯現成模型(沒有中文轉台羅)](https://huggingface.co/)

4. [套nn.Transformer的參考](https://blog.51cto.com/u_11466419/5983209)


#### 給自己的話
如果哪天真的回去修機器學習的話給自己一個紀錄 ><