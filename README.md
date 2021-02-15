# ImageSegmentation
The system functionality includes:  
- loading images and folders to system, 
- creating noise in images with use of 3 diffrent types of algorithms:  
  + salt-peper,
  + gaussian,  
  + speckle, 
- clearing images from noise with use of 3 diffrent types of algorithms:  
  + gaussian,
  + median,  
  + bilateral,  
- dividing images into groups before segmentation, based on:  
  + edges,  
  + blurr,
- segmentation of images using 3 diffrent types of algorithms,    
  + watershed,  
  + edges,  
  + region-based,  
- saving segmentation results,  
- evaluating results of segmentation with ground truth (diffrent types of algorithms):  
  + jaccarda index,   
  + f1 score,  
  + mean squared error,  
  + explained variance score.  
  
## How it looks
![1](https://user-images.githubusercontent.com/44240730/107938680-04b1e400-6f86-11eb-95e1-bf88d3f4fcf5.PNG)
![2](https://user-images.githubusercontent.com/44240730/107938683-054a7a80-6f86-11eb-9a22-f8ea46786452.PNG)
![3](https://user-images.githubusercontent.com/44240730/107938688-054a7a80-6f86-11eb-978b-8af7cdf0a104.PNG)
![4](https://user-images.githubusercontent.com/44240730/107938689-05e31100-6f86-11eb-8d6b-64d8e2d72849.PNG)
![5](https://user-images.githubusercontent.com/44240730/107938676-04194d80-6f86-11eb-81d1-905f4b92b3d2.PNG)

## How to run
Run by starting true_main.py.
