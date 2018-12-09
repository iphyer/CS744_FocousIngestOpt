# IngestOptFocous_FinalProject_CS744Fall2018
Video Inference System Optimization CS744Fall2018

![Detailed Plan](https://github.com/iphyer/IngestOptFocous_FinalProject_CS744Fall2018/blob/master/images/system.png)

## Introduction

This is the Github repository for [CS 744 Fall 2018 of UW-Madison](http://pages.cs.wisc.edu/~shivaram/cs744-fa18/) under Professor [Shivaram Venkataraman](http://shivaram.org/)'s instruction.

We want to optimize the ingest time of Focus system. Focus is a video inference system that is published on the 13th USENIX Symposium on Operating Systems Design and Implementation (OSDI ’18), October 8–10, 2018, Carlsbad, CA, USA. For more details check [Focus: Querying Large Video Datasets with Low Latency and Low Cost](https://www.usenix.org/conference/osdi18/presentation/hsieh). 

One of the key features of Focus is that it splits the video inference process into two part, one is called ingest and another one is called query. In ingest time, input videos are processed to generated {Frame number, Catergory} pair and during query stage, the query only needs to check the database for all records of the wanted category instead of running object detection system again. 

## Task Summary

In this course project, we want to optimize the ingest time of Focus system and in Focus, there are 2 neural networks, one is cheap Resnet18 for {Frame number, Catergory} pair generation and another is more expensive Resnet152 for more accurate {Frame number, Catergory} pair refining as shown in picture below. We also show the planned improvements. 

![layout of Focus and planned improvements](https://github.com/iphyer/IngestOptFocous_FinalProject_CS744Fall2018/blob/master/images/layout.jpg)

### Task 0

We decide to use the videos provided by [Urban Tracker](https://www.jpjodoin.com/urbantracker/index.htm) to as the target video to study in this task. Urban Tracker is the companion web page for **[Urban Tracker: Multiple Object Tracking in Urban Mixed Traffic](https://ieeexplore.ieee.org/document/6836010)**. It contains annotated dataset, results, source code and tools. 


### Task 0 - 1

We found it is too hard to get our model to the best performance so we adjust the method to manually label all the patches to get the ground truth data set.

We use the following convention to label the ground truth data set.

* 0 : non sense background
* 1 : cars
* 2 : pedestrians 

### Task1 : Resnet 50 classification

Sine the Resnet 50 shares the smame 14 begining layers with Resnet 152, we change Resnet 18 in Focus to Resnet 50 for our project.

Using Resnet 50 to build the cheap digest **{Frame_Number_Patch_Number.jpg, Catergory}** pair result as the benchmark result.

### Task2 : Resnet 152 classification

Using Resnet 152 to build the expensive digest **{Frame_Number_Patch_Number.jpg, Catergory}** pair result as the ground truth result.

### Task3 : align Resnet 50 and Resnet 152 together

Try to align some layers of Resnet 50 and Resnet 152 for simplicity to prove that two Resnets can be assembled together to make  **{Frame_Number_Patch_Number.jpg, Catergory}**  pair result.

No Training of either Resnet here. We only use the pre-trained weights. Compare this pre-trained alignment results with the benchmark. 


### Task4

Fix Resnet 50 and train Resnet 152 to fine tune the expensive parts of the neural network.

Using this trained Resnet 152 to generate  **{Frame_Number_Patch_Number.jpg, Catergory}**  pair result. Compare this pre-trained alignment results with the benchmark. 

### Task5

If we still have time, we plan to explore better clustering algorithm for patches clustering rather than the naive implementation used in Focus. 

