# CMSC 473 Project
## Jacob Barnett, Steve Calderon, Nimit Patel

This github repo contains code for our cs473 group project in which our team chose to evaluate the performance of Naive Bayes, MaxEnt, and Deep Neural Net models for the task of research paper classification for the Web Of Science(WOS) dataset located at https://data.mendeley.com/datasets/9rw3vkcfy4/6, 
the NSF Research Award Abstracts 1990-2003 Data Set(NSF) located at: https://archive.ics.uci.edu/ml/datasets/NSF+Research+Award+Abstracts+1990-2003, and the ARXIV research paper database metadata used as a dataset(ARXIV) that was pulled down from their OAI endpoint described here https://arxiv.org/help/bulk_data using metha OAI harvester located at https://github.com/miku/metha. 

For parsing and featurizing ARXIV it is recommended to have 16GB of ram since the dataset is large and this code is pretty unoptimized. For training and evaluating DNN ARXIV it is also recommended both a dedicated graphics card that supports CUDA and 16GB of ram. 

If you wish to be able to run any arbitrary python script within the project without having to first retrieve the datasets and run through the project's pipeline, you can download this file https://drive.google.com/file/d/15JzTPMWe5nHKKLn7v_2TBCvDe3sYAiMV/view?usp=sharing and extract it using 7zip(https://www.7-zip.org/). Otherwise, if you are cloning the project directly from git and don't use the files under /Data/ in the archive in the /Data/ folder in the location you cloned the project to, you will have to run through each step in the pipeline for the project described in the image below(A SVG version is available under /images/ in the repo in case you want to scale up the image to better read the text):

BE SURE TO RUN ALL SCRIPTS IN THE SAME DIRECTORY THEY ARE LOCATED IN SINCE THEY USE RELATIVE PATHS TO FIND THE LOCATION OF THE DATA FOLDER

![Pipeline](images/pipeline.png)
