# TableNet-pytorch
Pytorch Implementation of TableNet 
Research Paper : https://arxiv.org/abs/2001.01469

![TableNet Architecture](./images/model_arch.png)

## Description
In this project we will implement an end-to-end Deep learning architecture which will not only localize the Table in an image, but will also generate structure of Table by segmenting columns in that Table. After detecting Table structure from the image, we will use Pytesseract OCR package to read the contents of the Table.

To know more about the approach, refer my medium blog post,

Part 1: https://asagar60.medium.com/tablenet-deep-learning-model-for-end-to-end-table-detection-and-tabular-data-extraction-from-b1547799fe29

Part 2: https://asagar60.medium.com/tablenet-deep-learning-model-for-end-to-end-table-detection-and-tabular-data-extraction-from-a49ac4cbffd4

## Data
We will use both Marmot and Marmot Extended dataset for Table Recognition. Marmot dataset contains Table bounding box coordinates and extended version of this dataset contains  Column bounding box coordinates.

Marmot Dataset : https://www.icst.pku.edu.cn/cpdp/docs/20190424190300041510.zip
Marmot Extended dataset : https://drive.google.com/drive/folders/1QZiv5RKe3xlOBdTzuTVuYRxixemVIODp

Download processed Marmot dataset: https://drive.google.com/file/d/1irIm19B58-o92IbD9b5qd6k3F31pqp1o/view?usp=sharing

## Model
We will use DenseNet121 as encoder and build model upon it.

### Trainable Params
![Params](./images/trainable_params.png)

Download saved model : https://drive.google.com/file/d/1TKALmlwUM_n4gULh6A6Q35VPRUpWDmJZ/view?usp=sharing

### Performance compared to other encoder models ( Resnet18, EfficientNet-B0, EfficientNet-B1, VGG19 )

#### Table Detection - F1
![Table F1](./images/table_f1.PNG)

#### Table Detection - Loss
![Table Loss](./images/table_loss.PNG)

#### Column Detection - F1
![Column F1](./images/column_f1.PNG)

#### Column Detection - Loss
![Column Loss](./images/column_loss.PNG)

## Predictions

### Predictions from the model
![Prediction 1](./images/pred_1.PNG)

### After fixing table mask using contours
![Prediction 2](./images/pred_2.png)

### After fixing column mask using contours
![Prediction 3](./images/pred_3.png)

### After processing it through pytesseract
![Prediction 4](./images/pred_4.PNG)

## Deployed application 
https://vimeo.com/577282006


## Future Work
- [ ] Deploy this application on a remote server using AWS /StreamLit sharing/heroku.
- [ ] Model Quantization for faster inference time.
- [ ] Train for more epochs and compare the performances.
- [ ] Increase data size by adding data from ICDAR 2013 Table recognition dataset.


## References
1. [Table Net Research Paper](https://arxiv.org/abs/2001.01469)
2. [7 tips for squeezing maximum performance from pytorch](https://towardsdatascience.com/7-tips-for-squeezing-maximum-performance-from-pytorch-ca4a40951259)
3. [StreamLit](https://docs.streamlit.io/en/stable/)
4. [AppliedAI Course](https://www.appliedaicourse.com/course/11/Applied-Machine-learning-course)
