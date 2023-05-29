#Title: Diffusion Models against Learnable Kernels for Video Panoptic Segmentation
##Subtitle: Ceraolo Martorella Minini DLAV2023 Final Project

##Introduction - Explaination of what’s VPS

Panoptic segmentation is an advanced computer vision task that combines instance segmentation and semantic segmentation to provide a comprehensive understanding of the visual scene in a video. It aims to label every pixel in the video with a class label and an instance ID, thereby differentiating between different objects and identifying their boundaries.
To understand panoptic segmentation in detail, let's break it down into its components:
1. Semantic Segmentation: Semantic segmentation involves dividing an image or video into multiple meaningful segments based on the semantic content of the scene. Each pixel is assigned a class label representing the object or region it belongs to. For example, in a street scene, semantic segmentation would classify pixels into categories like pedestrians, cars, buildings, roads, and trees. It focuses on understanding the high-level scene structure.
2. Instance Segmentation: Instance segmentation takes semantic segmentation a step further by not only identifying object categories but also differentiating between individual instances of objects. It assigns a unique ID to each object instance within a class, enabling pixel-level separation of objects. For example, if there are multiple cars in an image, instance segmentation would distinguish them by assigning a unique ID to each car.
3. Video Panoptic Segmentation: Video Panoptic segmentation extends the concept of instance and semantic segmentation to video sequences. It combines the two tasks to provide a more comprehensive understanding of the scene. Panoptic segmentation aims to label every pixel in the video with both a class label and an instance ID, thereby distinguishing between different objects and identifying their boundaries over time. This information is valuable for applications like object tracking, activity recognition, and scene understanding.
To achieve panoptic segmentation in videos, various deep learning techniques are employed, often building upon advancements in image-based panoptic segmentation. These techniques typically involve the use of convolutional neural networks (CNNs) to process video frames, extract features, and predict the class labels and instance IDs for each pixel. The models are trained on large annotated datasets to learn to generalize and accurately segment objects in different video scenes.
Panoptic segmentation in videos can be challenging due to several factors, including occlusions, motion blur, and changes in lighting conditions. Handling temporal coherence and tracking objects across frames are important considerations to achieve accurate and consistent segmentation results. Overall, video panoptic segmentation combines semantic segmentation and instance segmentation to provide a detailed understanding of the visual scene in a video. It plays a crucial role in various applications, including autonomous driving, video surveillance, augmented reality, and robotics, enabling machines to perceive and interpret the environment more comprehensively.


##Approach
The two methods that we studied and worked on are Pix2Seq-D and Video-k-Net.

###Pix2Seq-D
The paper proposes a generalist approach to panoptic segmentation, which involves assigning both semantic (category) and instance ID labels to every pixel in an image. The task is challenging because different permutations of instance IDs can be valid solutions, requiring the learning of a high-dimensional one-to-many mapping. State-of-the-art methods typically rely on custom architectures and task-specific loss functions to address this problem.
In contrast, the authors present a formulation of panoptic segmentation as a discrete data generation problem, without relying on specific task-related biases. They use a diffusion model based on analog bits, which is a probabilistic generative model, to represent panoptic masks. The proposed model has a simple and generic architecture and loss function, making it applicable to a wide range of panoptic segmentation tasks.
One key advantage of the proposed method is its ability to handle video data in a streaming setting. By incorporating past predictions as a conditioning signal, the model learns to automatically track object instances across consecutive frames. This feature allows the method to perform video panoptic segmentation without the need for additional specialized techniques.
The authors conducted extensive experiments to evaluate their generalist approach. The results demonstrate that their method can achieve competitive performance compared to state-of-the-art specialist methods in similar settings. By leveraging the power of the diffusion model and incorporating temporal information, the proposed approach offers a promising alternative for panoptic segmentation tasks without the need for task-specific architectures or loss functions.



###Videoknet
Video K-Net is a framework for fully end-to-end video panoptic segmentation. Video panoptic segmentation involves simultaneously segmenting and tracking "things" (individual objects) and "stuff" (background regions) in a video. The proposed method builds upon K-Net, a technique that unifies image segmentation using learnable kernels. The authors observe that the learnable kernels from K-Net, which encode object appearances and contexts, can naturally associate identical instances across different frames in a video. This observation motivates the development of Video K-Net, which leverages kernel-based appearance modeling and cross-temporal kernel interaction to perform video panoptic segmentation.
Despite its simplicity, Video K-Net achieves state-of-the-art results on several benchmark datasets, including Citscapes-VPS, KITTI-STEP, and VIPSeg, without relying on additional complex techniques. In particular, on the KITTI-STEP dataset, the method achieves nearly 12% relative improvement over previous approaches. 


##History - our project
We believe that the idea behind it is incredibly fascinating and powerful, and that’s why we were very keen on working with it, notwithstanding the issues we had. We spent several weeks trying to run the Tensorflow code provided by Google Research, but we encountered numerous issues that prevented us from using their code (see the section #issues for more details). We tried a huge amount of solutions, different setups, several GPUs and GPU providers, and so on, without success. So more recently, we decided to embark on an ambitious mission: rewriting the Pix2Seq-D codebase in PyTorch. Fortunately, individual contributors on Github already converted some sub-parts of the project (e.g. Bit-diffusion, Pix2Seq). After some heavy work, we actually managed to have a draft of the full project. It is now running the training for the very first time, so we don’t expect perfect results yet. We plan on pursuing and completing this project also after the milestone deadline.
 
In parallel, since we knew about the uncertainty of such a challenge, we also setup and run the training of another architecture, Video-k-net, so that we also have good outputs to show, and a baseline performance to compare the results of our main contribution.


##Contribution overview
Our main contribution is within the Pix2Seq-D architecture. The contributions are three: the application of the architecture to the task of Video Panoptic Segmentation (the authors used it for other tasks, namely MISSING). 


First of all, we believe that our re-implementation in Pytorch can help the scientific community, ensuring more interoperability, clarity, and extending the audience that can build solutions on top of Pix2Seq-D. Diffusion models are very recent and proved to be very powerful, so the more researchers can have access to resources and codebases, the faster innnovations will come. 

Secondly, our endgoal is to try to improve Pix2Seq-D. The solution they propose to do Video Panoptic Segmentation is by using the Diffusion process, and when predicting the frame at time t, they guide the diffusion by conditioning on the previous panoptic mask. Our idea is the following: instad of conditioning only on the previous mask, we plan on finding a way to compute the difference between the current frame and the previous one, and condition on such difference, together with the previous mask. The idea is that, given two frames, it is very likely that there will not be extreme changes, but mainly instances that moved by some pictures. The difference between the frames will highlight what changed, and hence may make the diffusion process to find the mask faster, and more accurate. Since we had to re-write the whole codebase, we were not able to implement such configuration yet.

Finally, we ran another architecture, Video-k-net, in order to have a solid benchmark, with same pre-training and training, and become familiar with the panoptic mask generation process.


##Experimental setup
With both architectures, we kept consistency of the training procedure. In both cases, we used the following experimental setup:
- Backbone: ResNet50 pre-trained on ImageNet
- pre-training on Cityscapes
- training on KITTI-step
See the following section for more information on datasets.

We both did qualitative and quantitative evaluation. The qualitative comprised creating a script to visualize a gif video to see the colored panoptic masks. The quantitative comprised the following measures:
- Video Panoptic Quality (VPQ) 
- Segmentation and Tracking Quality (STQ)

Finally, since the training processes were heavy, we used powerful GPUs. More specifically, we used:
- an RTX3090 for Videokent
- two A100 of 80GB for PIx2Seq-d

##Data
The dataset used for pre-training is Cityscapes. Cityscapes dataset is a high-resolution road-scene dataset which contains 19 classes. (8 thing classes and 11 stuff classes). It contains 2975 images for training, 500 images for validation and 1525 images for testing.

Our pre-training for Pix2Seq-D was done with the images in .png format. The dataset can be downloaded with the official scripts https://github.com/mcordts/cityscapesScripts

The dataset used for pre-training is KITTI-STEP. 
It can be obtained from Huggingface, thanks to the authors of Video-k-net. We obtained the dataset from there both for training Pix2Seq-D and Video-k-net. https://huggingface.co/LXT/VideoK-Net/tree/main
Description of the dataset
This dataset contains videos recorded by a camera mounted on a driving car. KITTI-STEP has 21 and 29 sequences for training and testing, respectively. The training sequences are further split into training set (12 sequences) and validation set (9 sequences). Since the sequence length sometimes reaches over 1000 frames, long-term tracking is required. Prior research has shown that with increasing sequence length, the tracking difficulty increases too.
The labels of the dataset, which are the panoptic masks, are provided in .png format. More specifically, the groundtruth panoptic map is encoded as follows in PNG format:

```
R = semantic_id
G = instance_id // 256
B = instance % 256
```


The possible things classes are:
Label Name     | Label ID
-------------- | --------
road           | 0
sidewalk       | 1
building       | 2
wall           | 3
fence          | 4
pole           | 5
traffic light  | 6
traffic sign   | 7
vegetation     | 8
terrain        | 9
sky            | 10
person&dagger; | 11
rider          | 12
car&dagger;    | 13
truck          | 14
bus            | 15
train          | 16
motorcycle     | 17
bicycle        | 18
void           | 255

##Results
Qualitative and Quantitative results of your experiments. 

Qualitative: Put the gifs here!!
Quantitative: metrics


##Code
Here you can find the link to our implementation of Pix2Seq-d: Link to repo 1 
Here you can find the link to our repository for Video-k-net: Link to repo 2

##Side notes: issues we had
We nelieve it can be useful to share with the EPFL community the issues that we encountered…

Given the time constraint, we decided to switch to the use of external pay-per-use GPUs, namely runpod.io.
