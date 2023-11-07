# HW2 Predict the Characters in The Simposons

## Relative:
> Kaggle : https://www.kaggle.com/competitions/machine-learning-2023nycu-classification?rvi=1

> Colab : https://colab.research.google.com/?authuser=0#create=true

> Drive : https://drive.google.com/drive/folders/1d3DRBGYR66FVmLq-GTpRar8XBDQLJWZ0?usp=drive_link
## 目標
給予 Simposons 動畫中50種人物圖片資料，去預測test-final檔案裡面的圖片人物，並且在Kaggle做排名。

## 最後結果
* public : 0.96848

* private : 0.96849

## 檔案執行 (bash HW2.sh)
```bash
python CNN_Classification.ipynb
```

## 檔案內容
```
├─test-final
  └─test-final
├─train
    └─train
        ├─abraham_grampa_simpson
        ├─agnes_skinner
        ├─apu_nahasapeemapetilon
        ├─barney_gumble
        ├─bart_simpson
        ├─brandine_spuckler
        ├─carl_carlson
        ├─charles_montgomery_burns
        ├─chief_wiggum
        ├─cletus_spuckler
        ├─comic_book_guy
        ├─disco_stu
        ├─dolph_starbeam
        ├─duff_man
        ├─edna_krabappel
        ├─fat_tony
        ├─gary_chalmers
        ├─gil
        ├─groundskeeper_willie
        ├─homer_simpson
        ├─jimbo_jones
        ├─kearney_zzyzwicz
        ├─kent_brockman
        ├─krusty_the_clown
        ├─lenny_leonard
        ├─lionel_hutz
        ├─lisa_simpson
        ├─lunchlady_doris
        ├─maggie_simpson
        ├─marge_simpson
        ├─martin_prince
        ├─mayor_quimby
        ├─milhouse_van_houten
        ├─miss_hoover
        ├─moe_szyslak
        ├─ned_flanders
        ├─nelson_muntz
        ├─otto_mann
        ├─patty_bouvier
        ├─principal_skinner
        ├─professor_john_frink
        ├─rainier_wolfcastle
        ├─ralph_wiggum
        ├─selma_bouvier
        ├─sideshow_bob
        ├─sideshow_mel
        ├─snake_jailbird
        ├─timothy_lovejoy
        ├─troy_mcclure
        └─waylon_smithers
```

## 人物種類
| idx  | character |
| -------- | -------- |
|1	|abraham_grampa_simpson|
|2	|agnes_skinner|
|3	|apu_nahasapeemapetilon|
|4	|barney_gumble|
|5	|bart_simpson|
|6	|brandine_spuckler|
|7	|carl_carlson|
|8	|charles_montgomery_burns|
|9	|chief_wiggum|
|10	|cletus_spuckler|
|11	|comic_book_guy|
|12	|disco_stu|
|13	|dolph_starbeam|
|14	|duff_man|
|15	|edna_krabappel|
|16	|fat_tony|
|17	|gary_chalmers|
|18	|gil|
|19	|groundskeeper_willie|
|20	|homer_simpson|
|21	|jimbo_jones|
|22	|kearney_zzyzwicz|
|23	|kent_brockman|
|24	|krusty_the_clown|
|25	|lenny_leonard|
|26	|lionel_hutz|
|27	|lisa_simpson|
|28	|lunchlady_doris|
|29	|maggie_simpson|
|30	|marge_simpson|
|31	|martin_prince|
|32	|mayor_quimby|
|33	|milhouse_van_houten|
|34	|miss_hoover|
|35	|moe_szyslak|
|36	|ned_flanders|
|37	|nelson_muntz|
|38	|otto_mann|
|39	|patty_bouvier|
|40	|principal_skinner|
|41	|professor_john_frink|
|42	|rainier_wolfcastle|
|43|	ralph_wiggum|
|44|	selma_bouvier|
|45|	sideshow_bob|
|46|	sideshow_mel|
|47	|snake_jailbird|
|48|	timothy_lovejoy|
|49	|troy_mcclure|
|50|	waylon_smithers|

### 人數數量圖表
![img](https://github.com/ChouGiGiNYCU/Machine_Learning_HW_Spring/blob/main/img/num.png)

## 初始想法
想說直接套一個CNN Classification 模型架構去跑，剛開始使用 ResNet50 沒做任何資料處理。

Accuracy : 0.6 (沒有很好,epoch=10)

### **tips**
因為只有預測50個人物，最後一層要改成50
``` python=
model = models.efficientnet_v2_s().to(device)
PATH = '/content/drive/MyDrive/machine_learning/HW2/model_epoch_50.4.pth'
model.classifier = nn.Linear(1280, 50).to(device)
model.load_state_dict(torch.load(PATH,map_location=torch.device('cpu')))
```

## 第一次改進
做了一些資料增強(水平反轉，垂直反轉等等...)，模型架構不變。

Accuracy : 0.8 (適中,epoch=10)

#### *資料增強程式*
```python
#org data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
#data aug
transformAug1 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomApply([transforms.RandomHorizontalFlip()], p=0.1),
    transforms.RandomApply([transforms.RandomVerticalFlip()], p=0.1),
    transforms.RandomApply([transforms.RandomRotation(10)], p=0.1),

    transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomInvert(p=0.1),
    transforms.RandomPosterize(bits=2, p=0.1),
    transforms.RandomApply([transforms.RandomSolarize(threshold=1.0)], p=0.1),
    transforms.RandomApply([transforms.RandomAdjustSharpness(sharpness_factor=2)], p=0.1),

    transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.6, p=1.0)], p=0.1),
    transforms.RandomApply([transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))], p=0.1),
    transforms.RandomApply([transforms.ElasticTransform(alpha=250.0)], p=0.1),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))], p=0.1),

    transforms.ToTensor(),
    transforms.RandomApply([AddGaussianNoise(0., 0.05)], p=0.1),  # mean and std
    transforms.RandomApply([AddSpeckleNoise(noise_level=0.1)], p=0.1),
    transforms.RandomApply([AddPoissonNoise(lam=0.1)], p=0.1),
    transforms.RandomApply([AddSaltPepperNoise(salt_prob=0.05, pepper_prob=0.05)], p=0.1),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transformAug2 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomApply([transforms.RandomHorizontalFlip()], p=0.2),
    transforms.RandomApply([transforms.RandomVerticalFlip()], p=0.2),
    transforms.RandomApply([transforms.RandomRotation(10)], p=0.2),

    transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.2),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomInvert(p=0.2),
    transforms.RandomPosterize(bits=2, p=0.2),
    transforms.RandomApply([transforms.RandomSolarize(threshold=1.0)], p=0.2),
    transforms.RandomApply([transforms.RandomAdjustSharpness(sharpness_factor=2)], p=0.2),

    transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.6, p=1.0)], p=0.2),
    transforms.RandomApply([transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))], p=0.2),
    transforms.RandomApply([transforms.ElasticTransform(alpha=250.0)], p=0.2),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))], p=0.2),

    transforms.ToTensor(),
    transforms.RandomApply([AddGaussianNoise(0., 0.05)], p=0.2),  # mean and std
    transforms.RandomApply([AddSpeckleNoise(noise_level=0.1)], p=0.2),
    transforms.RandomApply([AddPoissonNoise(lam=0.1)], p=0.5),
    transforms.RandomApply([AddSaltPepperNoise(salt_prob=0.05, pepper_prob=0.05)], p=0.2),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
```

## 第二次改進
改用 Efficientnet_v2_s，且一樣有用資料增強，發現效果其實還不錯。

Accuracy : 0.9 (不錯,epoch=10)

## 第三次改進
增加訓練的次數，把 epoch 提高到20~30區間

Accuracy : 0.97x (不錯,epoch=30)

## Train_Loss_Function
![img](https://github.com/ChouGiGiNYCU/Machine_Learning_HW_Spring/blob/main/img/cnn_train_lose.png)

## Filter
# 獲取模型的第一个卷積層
```python
conv1 = None
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        conv1 = module
        break
if conv1 is not None:
    weights = conv1.weight.data
    weight_array = weights.cpu().numpy()
    num_kernels = weight_array.shape[0]
    plt.figure(figsize=(10, 10))
    for i in range(num_kernels):
        plt.subplot(8, 8, i + 1)
        plt.imshow(weight_array[i, 0, :, :], cmap='viridis')
        plt.axis('off')
    plt.show()
else:
    print("layer not found")
看一下第一層的filter 模型最後使用的是哪些filter
```
![img](https://github.com/ChouGiGiNYCU/Machine_Learning_HW_Spring/blob/main/img/filter.png)

## Feature_Map
把圖片經過第一層 conv layer 後的圖片印出來觀察
```python
# 載入一張圖片（替換為你的圖片路徑）
from PIL import Image
img_path = '/content/drive/MyDrive/machine_learning/HW2/img/train/agnes_skinner/ENQ_13.jpg'
img = Image.open(img_path)

# 適當的圖像預處理
preprocess = transforms.Compose([
    transforms.Resize(224),  # 調整圖像大小
    transforms.CenterCrop(224),  # 中心裁剪
    transforms.ToTensor(),  # 將圖像轉換為Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img = preprocess(img)
img = img.unsqueeze(0)  # 添加批次維度


# 獲得模型的第一層卷積層（Conv2d）
first_conv_layer = model.features[0]

# 創建一個模型，只包含第一層卷積層
first_layer_model = torch.nn.Sequential(first_conv_layer)

# 將圖片傳遞給第一層卷積層
with torch.no_grad():
    feature_map = first_layer_model(img)

# 打印第一層特徵圖的形狀
print("Shape of the first layer feature map:", feature_map.shape)

# 顯示第一層特徵圖的其中一個通道
for i in range(feature_map.size(1)):
    plt.subplot(feature_map.size(1)//5 + 1, 5, i + 1)  # 創建子圖
    plt.imshow(feature_map[0, i].cpu().numpy())
    # plt.title(f'Channel {i}')
    # plt.axis('off')
    plt.axis('off')  # 關閉座標軸

plt.show()
```
![img](https://github.com/ChouGiGiNYCU/Machine_Learning_HW_Spring/blob/main/img/feature_map.png)

## Confusion Matrix
```python
predicted_labels = []
with torch.no_grad():
    for data in validationSetloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        predicted_labels.extend(predicted.cpu().numpy())
true_labels = [label for _, label in validationSet]
# 计算混淆矩阵
cm = confusion_matrix(true_labels, predicted_labels, labels=range(50))
plt.figure(figsize=(15, 10))
sns.heatmap(cm, annot=False, cmap="coolwarm")
# 设置横轴和纵轴的刻度范围
plt.xticks(np.arange(50), np.arange(50))
plt.yticks(np.arange(50), np.arange(50))
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")

plt.show()
```
![img](https://github.com/ChouGiGiNYCU/Machine_Learning_HW_Spring/blob/main/img/confuse_matrix.png)


## 改善部分
* 可以使用更多model(Resnet50、18)去做 Staking ，把最後得出來得機率去做，由於每個model學習到的feature都不一樣，所以綜合所有的model能力會遠大於一個。
* 資料前處理，說不定更多資料增強(訓練資料集變多)，說不定準確率上去。

