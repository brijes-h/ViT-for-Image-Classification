# Vision-Transformers-for-Image-Classification

ViT (Vision Transformer) is a deep learning model architecture for image classification tasks. It was proposed by researchers at Google in 2020 as an alternative to traditional convolutional neural networks (CNNs).

# Model and working

<img src = "https://user-images.githubusercontent.com/91772980/222978090-faa156e9-cbab-4d76-bd43-62583fca8d8a.png" width="500" height="350"/>


The image is divided into patches and processed as a sequence by the transformer. The standard transformer receives an input of 1D sequence of token embeddings. For 2D images, the image is reshaped into a sequence of flattened 2D patches.

The number of patches can be calculated using the following formula: `Number of Patches = Width x Height / (Patch Width x Patch Height)`

# References

**Vision Transformers (ViT) in Image Recognition – 2022 Guide:** [https://viso.ai/deep-learning/vision-transformer-vit/](https://viso.ai/deep-learning/vision-transformer-vit/)

****An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (Paper Explained), Yannic Kilcher:**** [https://www.youtube.com/watch?v=TrdevFK_am4](https://www.youtube.com/watch?v=TrdevFK_am4)

**ViT explained (with gifs):**  [https://www.analyticsvidhya.com/blog/2021/03/an-image-is-worth-16x16-words-transformers-for-image-recognition-at-scale-vision-transformers/](https://www.analyticsvidhya.com/blog/2021/03/an-image-is-worth-16x16-words-transformers-for-image-recognition-at-scale-vision-transformers/)

****************Papers:**************** 

**********An image is worth 16x16 words: Transformers for Image recognition at scale**********

[https://arxiv.org/pdf/2010.11929v2.pdf](https://arxiv.org/pdf/2010.11929v2.pdf)
