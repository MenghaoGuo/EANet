# Beyond Self-attention: External Attention using Two Linear Layers for Visual Tasks
paper : https://arxiv.org/abs/2105.02358
# EAMLP will come soon
# Jittor code will come soon

# Pascal VOC test result [link](http://host.robots.ox.ac.uk:8080/anonymous/T4OS1E.html)
# Pascal VOC pretrained model [link](https://cloud.tsinghua.edu.cn/f/1e7253ae0748470482e4/)
You can download the pretrained model and then run  python test.py to reproduce the pascal voc test result.

### Other implementation:
Pytorch  : https://github.com/xmu-xiaoma666/External-Attention-pytorch

## Acknowledgments 

We would like to sincerely thank [HamNet_seg](https://github.com/Gsunshine/Enjoy-Hamburger), [EMANet_seg](https://github.com/XiaLiPKU/EMANet), [openseg](https://github.com/openseg-group/openseg.pytorch), [T2T-ViT](https://github.com/yitu-opensource/T2T-ViT), [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) and [PyTorch-StudioGAN](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN) for their awesome released code. 


## Astract

Attention mechanisms, especially self-attention, play an increasingly important role in deep feature representation in visual tasks. Self-attention updates the feature at each position by computing a weighted sum of features using pair-wise affinities across all positions to capture long-range dependency within a single sample. However, self-attention has a quadratic complexity and ignores potential correlation between different samples. This paper proposes a novel attention mechanism which we call external attention, based on two external, small, learnable, and shared memories, which can be implemented easily by simply using two cascaded linear layers and two normalization layers; it conveniently replaces self-attention in existing popular architectures. External attention has linear complexity and implicitly considers the correlations between all samples. Extensive experiments on image classification, semantic segmentation, image generation, point cloud classification and point cloud segmentation tasks reveal that our method provides comparable or superior performance to the self-attention mechanism and some of its variants, with much lower computational and memory costs.


## Jittor

Jittor is a  high-performance deep learning framework which is easy to learn and use. It provides interfaces like Pytorch.

You can learn how to use Jittor in following links:

Jittor homepage:  https://cg.cs.tsinghua.edu.cn/jittor/

Jittor github:  https://github.com/Jittor/jittor

If you has any questions about Jittor, you can ask in Jittor developer QQ Group: 761222083


## Citation

If it is helpful for your work, please cite this paper:
```
@misc{guo2021attention,
      title={Beyond Self-attention: External Attention using Two Linear Layers for Visual Tasks}, 
      author={Meng-Hao Guo and Zheng-Ning Liu and Tai-Jiang Mu and Shi-Min Hu},
      year={2021},
      eprint={2105.02358},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

