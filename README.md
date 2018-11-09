# IngestOptFocous_FinalProject_CS744Fall2018
Video Inference System Optimization CS744Fall2018

## Introduction

This is the Github repository for [CS 744 Fall 2018 of UW-Madison[(http://pages.cs.wisc.edu/~shivaram/cs744-fa18/) under Professor [Shivaram Venkataraman](http://shivaram.org/)'s instruction.

We want to optimize the ingest time of Focus system. Focus is a video inference system that is published on the 13th USENIX Symposium on Operating Systems Design and Implementation (OSDI ’18), October 8–10, 2018, Carlsbad, CA, USA. For more details check [Focus: Querying Large Video Datasets with Low Latency and Low Cost](https://www.usenix.org/conference/osdi18/presentation/hsieh). 

One of the key features of Focus is that it splits the video inference process into two part, one is called ingest and another one is called query. In ingest time, input videos are processed to generated {Frame number, Catergory} pair and during query stage, the query only needs to check the database for all records of the wanted category instead of running object detection system again. 

## Task Summary

In this course project, we want to optimize the ingest time of Focus system and in Focus, there are 2 neural networks, one is cheap ResetNet18 for {Frame number, Catergory} pair generation and another is more expensive Resetnet152 for more accurate {Frame number, Catergory} pair refining as shown in picture below. We also show the planned improvements. 

![layout of Focus and planned improvements](https://github.com/iphyer/IngestOptFocous_FinalProject_CS744Fall2018/blob/master/layout.jpg)

### Task1 

Using ResetNet18 to build the cheap digest **{Frame number, BBoxID,{BBox x1,y1,x2,y2}, Catergory}** pair result as the benchmark result.

### Task2 

Using ResetNet152 to build the expensive digest **{Frame number, BBoxID,{BBox x1,y1,x2,y2}, Catergory}** pair result as the ground truth result.

### Task3

Try to align some layers of ResetNet18 and ResetNet152 or ResetNet34 for simplicity to prove that two ResetNet can be assembled together to make  **{Frame number, BBoxID,{BBox x1,y1,x2,y2}, Catergory}**  pair result.

No Training of either ResetNet here. We only use the pre-trained weights. Compare this pre-trained alignment results with the benchmark. 


### Task4

Fix ResetNet18 and train ResetNet152 to fine tune the expensive parts of the neural network.

Using this trained ResetNet152 to generate  **{Frame number, BBoxID,{BBox x1,y1,x2,y2}, Catergory}**  pair result. Compare this pre-trained alignment results with the benchmark. 

### Task5

If we still have time, we plan to explore better clustering algorithm for patches clustering rather than the naive implementation used in Focus. 

