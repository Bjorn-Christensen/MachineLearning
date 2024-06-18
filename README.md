# RCNN
Various RCNN implementations for practice

/*  
Author: Bjorn Christensen
Date: 6/16/2024
*/

This is a simple project to test and remind myself of methods used in finetuning pretrained models.
I did not run the finetuning to create an optimally trained model, however I was still able to see very solid results after only a few epochs.
The code in the FacemaskModel: "FinetuneFasterRCNNFacemask.ipynb" and its comments rely on the assumption that one is using Google Colab and CUDA, however it does include CPU code variations in the comments.
The inclusion of the Golfball model was intended to help myself practice with using different formats of datasets and data extraction. It is more or less a copy of the Facemask notebook but has allowed me to test the effectiveness of my personal hardware (1060 ti) against colab's cpu and limited T4 gpu usage.
