# IngestOptFocous_FinalProject_CS744Fall2018
Video Inference System Optimization CS744Fall2018

## Introduction

This is the Github reposotyr for [CS 744 Fall 2018 of UW-Madison[(http://pages.cs.wisc.edu/~shivaram/cs744-fa18/) under Professor [Shivaram Venkataraman](http://shivaram.org/)'s instruction.

We want to optimize the ingest time of Focus system. Foucs is a video inference system that is published on 13th USENIX Symposium on Operating Systems Design and Implementation (OSDI ’18),October 8–10, 2018, Carlsbad, CA, USA. For more details check [Focus: Querying Large Video Datasets with Low Latency and Low Cost](https://www.usenix.org/conference/osdi18/presentation/hsieh). 

One of the key feature of Foucs is that it splits the video inference process into two part, one is called ingest and another one is called query. In ingest time, input videos are processed to generated {Frame number, Catergory} pair and during query stage, the query only need to check the database for all records of wanted catergory instead of running object detection system again. 

## Task Summary

In this course project, we want to optimize the ingest time of Foucs system and in Foucus, there are 2 neural networks, one is cheap ResetNet18 for {Frame number, Catergory} pair generation and another is more expensive Resetnet152 for more accurate {Frame number, Catergory} pair refining as shown in picture below.


