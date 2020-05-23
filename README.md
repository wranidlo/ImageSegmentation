# ImageSegmentation
The system functionality includes:  
- loading images and folders to system, 
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
  
The system includes GUI.  
Run by starting true_main.py.
