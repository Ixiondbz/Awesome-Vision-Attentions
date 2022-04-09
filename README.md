# This repo is built for paper: Attention Mechanisms in Computer Vision: A Survey  [paper](https://arxiv.org/abs/2111.07624)

## 介绍该论文的中文版博客 [链接](https://mp.weixin.qq.com/s/0iOZ45NTK9qSWJQlcI3_kQ )



## Citation

If it is helpful for your work, please cite this paper:

```
@misc{guo2021attention_survey,
      title={Attention Mechanisms in Computer Vision: A Survey}, 
      author={Meng-Hao Guo and Tian-Xing Xu and Jiang-Jiang Liu and Zheng-Ning Liu and Peng-Tao Jiang and Tai-Jiang Mu and Song-Hai Zhang and Ralph R. Martin and Ming-Ming Cheng and Shi-Min Hu},
      year={2021},
      eprint={2111.07624},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


![image](https://github.com/MenghaoGuo/Awesome-Vision-Attentions/blob/main/imgs/fuse.png)


<!-- ![image](https://github.com/MenghaoGuo/Awesome-Vision-Attentions/blob/main/imgs/attention_category.png) -->



- [Vision-Attention-Papers](#vision-attention-papers)
  * [Channel attention](#channel-attention)
  * [Spatial attention](#spatial-attention)
  * [Temporal attention](#temporal-attention)
  * [Branch attention](#branch-attention)
  * [Channel \& Spatial attention](#channelspatial-attention)
  * [Spatial \& Temporal attention](#spatialtemporal-attention)



* TODO : Code about different attention mechanisms based on [Jittor](https://github.com/Jittor/jittor) will be released gradually.
* TODO :  [Code]() link will come soon.
* TODO :  collect more related papers. Contributions are welcome. 

🔥 (citations > 200)  


## Channel attention

* (**IMPLEMENTED**) Squeeze-and-Excitation Networks (CVPR 2018) [pdf](https://arxiv.org/pdf/1709.01507), (PAMI2019 version) [pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8701503)  🔥
* Image superresolution using very deep residual channel attention networks (ECCV 2018) [pdf](https://arxiv.org/pdf/1807.02758)   🔥 
* Context encoding for semantic segmentation (CVPR 2018) [pdf](https://arxiv.org/pdf/1803.08904)   🔥 
* Spatio-temporal channel correlation networks for action classification (ECCV 2018)  [pdf](https://arxiv.org/pdf/1806.07754)
* (**TRIED IMPLEMENTING FAILED**) Global second-order pooling convolutional networks (CVPR 2019) [pdf](https://arxiv.org/pdf/1811.12006)
* Srm : A style-based recalibration module for convolutional neural networks (ICCV 2019)  [pdf](https://arxiv.org/pdf/1903.10829) 
* You look twice: Gaternet for dynamic filter selection in cnns (CVPR 2019)  [pdf](https://arxiv.org/pdf/1811.11205)
* Second-order attention network for single image super-resolution (CVPR 2019) [pdf](https://openaccess.thecvf.com/content_CVPR_2019/papers/Dai_Second-Order_Attention_Network_for_Single_Image_Super-Resolution_CVPR_2019_paper.pdf)  🔥 
* DIANet: Dense-and-Implicit Attention Network (AAAI 2020)[pdf](https://arxiv.org/pdf/1905.10671.pdf)
* Spsequencenet: Semantic segmentation network on 4d point clouds (CVPR 2020)  [pdf](https://openaccess.thecvf.com/content_CVPR_2020/html/Shi_SpSequenceNet_Semantic_Segmentation_Network_on_4D_Point_Clouds_CVPR_2020_paper.html)
* (**IMPLEMENTED**)Ecanet: Efficient channel attention for deep convolutional neural networks (CVPR 2020) [pdf](https://arxiv.org/pdf/1910.03151)   🔥 
* Gated channel transformation for visual recognition (CVPR2020)  [pdf](https://arxiv.org/pdf/1909.11519) 
* Fcanet: Frequency channel attention networks (ICCV 2021)  [pdf](https://arxiv.org/pdf/2012.11879)

## Spatial attention

- Recurrent models of visual attention (NeurIPS 2014), [pdf](https://arxiv.org/pdf/1406.6247)   🔥 
- Spatial transformer networks (NeurIPS 2015) [pdf](https://arxiv.org/pdf/1506.02025)   🔥 
- Multiple object recognition with visual attention (ICLR 2015) [pdf](https://arxiv.org/pdf/1412.7755)   🔥 
- Look closer to see better: Recurrent attention convolutional neural network for fine-grained image recognition (CVPR 2017) [pdf](https://openaccess.thecvf.com/content_cvpr_2017/papers/Fu_Look_Closer_to_CVPR_2017_paper.pdf)   🔥 
- Learning multi-attention convolutional neural network for fine-grained image recognition (ICCV 2017) [pdf](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zheng_Learning_Multi-Attention_Convolutional_ICCV_2017_paper.pdf)   🔥 
- Diversified visual attention networks for fine-grained object classification (TMM 2017) [pdf](https://arxiv.org/pdf/1606.08572)   🔥 
- Non-local neural networks (CVPR 2018) [pdf](https://arxiv.org/pdf/1711.07971)   🔥 
- Relation networks for object detection (CVPR 2018) [pdf](https://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Relation_Networks_for_CVPR_2018_paper.pdf)   🔥 
- **a2-nets: Double attention networks (NeurIPS 2018)** [pdf](https://arxiv.org/pdf/1810.11579)   🔥 
- **Tell me where to look: Guided attention inference network**(CVPR 2018) [pdf](https://arxiv.org/pdf/1802.10171)   🔥 
- Learn to pay attention (ICLR 2018) [pdf](https://arxiv.org/pdf/1804.02391.pdf)   🔥
- **Attention U-Net: Learning Where to Look for the Pancreas** (MIDL 2018) [pdf](https://arxiv.org/pdf/1804.03999.pdf)   🔥
- **Attention augmented convolutional networks** (ICCV 2019) [pdf](https://arxiv.org/pdf/1904.09925)   🔥 
- Local relation networks for image recognition(Swin Transformers) (ICCV 2019) [pdf](https://arxiv.org/pdf/1904.11491)
- Latentgnn: Learning efficient nonlocal relations for visual recognition (ICML 2019) [pdf](https://arxiv.org/pdf/1905.11634)
- Gcnet: Non-local networks meet squeeze-excitation networks and beyond (ICCVW 2019) [pdf](https://arxiv.org/pdf/1904.11492)   🔥 
- Asymmetric non-local neural networks for semantic segmentation (ICCV 2019) [pdf](https://arxiv.org/pdf/1908.07678)   🔥 
- **Looking for the devil in the details: Learning trilinear attention sampling network for fine-grained image recognition** (CVPR 2019) [pdf](https://arxiv.org/pdf/1903.06150) 
- **Diagnose like a radiologist: Attention guided convolutional neural network for thorax disease classification**(arXiv 2019) [pdf](https://arxiv.org/pdf/1801.09927)
- **Exploring self-attention for image recognition (CVPR 2020) [pdf](https://arxiv.org/pdf/2004.13621)**
- Disentangled non-local neural networks (ECCV 2020) [pdf](https://arxiv.org/pdf/2006.06668) 
- End-to-end object detection with transformers (ECCV 2020) [pdf](https://arxiv.org/pdf/2005.12872)   🔥 
- An image is worth 16x16 words: Transformers for image recognition at scale (ICLR 2021) [pdf](https://arxiv.org/pdf/2010.11929)   🔥 
- Is Attention Better Than Matrix Decomposition? (ICLR 2021) [pdf](https://arxiv.org/abs/2109.04553) 
- Beit: Bert pre-training of image transformers (arxiv 2021) [pdf](https://arxiv.org/pdf/2106.08254)
- **Beyond Self-attention: External attention using two linear layers for visual tasks**(arxiv 2021) [pdf](https://arxiv.org/pdf/2105.02358)
- Transformer in transformer (arxiv 2021) [pdf](https://arxiv.org/pdf/2103.00112)


## ChannelSpatial attention

- **Residual attention network for image classification** (CVPR 2017) [pdf](https://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_Residual_Attention_Network_CVPR_2017_paper.pdf) 🔥
- **SCA-CNN: spatial and channel-wise attention in convolutional networks for image captioning** (CVPR 2017) [pdf](https://openaccess.thecvf.com/content_cvpr_2017/papers/Chen_SCA-CNN_Spatial_and_CVPR_2017_paper.pdf) 🔥
- **CBAM: convolutional block attention module** (ECCV 2018) [pdf](https://openaccess.thecvf.com/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf)  🔥
- Recalibrating fully convolutional networks with spatial and channel “squeeze and excitation” blocks (TMI 2018) [pdf](https://arxiv.org/pdf/1808.08127.pdf)
- Bam: Bottleneck attention module(BMVC 2018) [pdf](http://bmvc2018.org/contents/papers/0092.pdf) 🔥
- Learning what and where to attend (ICLR 2019) [pdf](https://openreview.net/pdf?id=BJgLg3R9KQ)
- **Improving convolutional networks with self-calibrated convolutions**(CVPR 2020) [pdf](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Improving_Convolutional_Networks_With_Self-Calibrated_Convolutions_CVPR_2020_paper.pdf)
- Strip Pooling: Rethinking spatial pooling for scene parsing (CVPR 2020) [pdf](https://openaccess.thecvf.com/content_CVPR_2020/papers/Hou_Strip_Pooling_Rethinking_Spatial_Pooling_for_Scene_Parsing_CVPR_2020_paper.pdf)
- Rotate to attend: Convolutional triplet attention module, (WACV 2021) [pdf](https://arxiv.org/pdf/2010.03045.pdf)
- Simam: A simple, parameter-free attention module for convolutional neural networks (ICML 2021) [pdf](http://proceedings.mlr.press/v139/yang21o/yang21o.pdf)
