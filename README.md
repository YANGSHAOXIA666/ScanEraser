# ScanEraser: Lightweight Neural Networks for On-device Handwriting Erasure
The ScanEraser architecture consists of two fundamental components: a macrofield cleaner and a microfield refiner. The macrofield cleaner implements a mask-based handwriting removal mechanism with explicit erasure guidance. A feature-grabbing block enhances the capture
of contextual features around handwritten text. The network is compact and efficient, trained using a Generative Adversarial Network (GAN). The microfield refiner employs a progressive refinement network with constrained receptive fields to repair detailed textures that the
macrofield cleaner cannot address adequately.
# Data preparation
The Epaper and Ebook datasets used for handwritten text removal research can be downloaded through the following links:

Epaper - [Baidu Cloud](https://pan.baidu.com/s/1h-g5OwKR9Rd8TqSDHeISPQ) (Password : a653 )  

Ebook - [Baidu Cloud](https://pan.baidu.com/s/1_XZ0N8zIQ52-MFIJjtg_fw) (Password : a653 )

Note: This datasets can only be used for non-commercial research purpose. The training sets are available now, but requires a password to unzip. To use the databases, please fill out the [agreement form](https://github.com/YANGSHAOXIA666/ScanEraser/blob/main/Agreement_form.doc) and send it via email to us(666yyw666@gmail.com, 2371416@stu.neu.edu.cn). We will provide the unzipping password after receiving and approving your request.
# Training
Once the data is well prepared, you can begin training:

```bash
python train.py
```

# Testing
If you want to predict the results, run:

```bash
python test.py
```

# Citation
If you find our method or dataset useful for your reserach, please cite:
