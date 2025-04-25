# Adaptive Multimodal Deep Prompt Learning for Blind Image Quality Assessment

## Introduction
In this paper, we propose a novel multimodal deep prompt learning framework for blind image quality assessment (BIQA), named AdmpIQA. Specifically, we introduce learnable prompts in both the language and visual branches of the CLIP model, enabling fine-grained adjustments to vision-language representations. Additionally, we incorporate a visual feature adaptation module to enhance the expressive capability of the visual branch, allowing it to better capture the characteristics of image distortion.
![xx](./fig/framework.png)

## Train and Test
First, download datasets.

Second, update path of datasets defined in ```train_test_clip.py```

```
path = {
    'CSIQ': '/home/datasets/IQA_Datasets/CSIQ',
    ......
}
```

Third, train and test the model using the following command:
```
python train_test_clip.py --dataset CSIQ --model AdmpIQA
```
Finally, check the results in the folder `./log`.

## Acknowledgement
This project is based on [MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning), [DBCNN](https://github.com/zwx8981/DBCNN-PyTorch), and [CLIP-IQA](https://github.com/IceClear/CLIP-IQA). Thanks for these awesome works.
