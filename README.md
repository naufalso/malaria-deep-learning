"# malaria-deep-learning" 


For  preprocessing,  we  perform  4  steps:  rescaling,  sheer range,  zoom  range,  and horizontal  flipping.  Rescalingis  to normalize the  input value,  which is the  pixel  of the  imagefor this  instance.  All  the  pixel  of  the  images  is  divided  with the value of 255 to make it fall into 0to 1 range value. Sheer range, zoom  range,  and  horizontalflip  are  for  data  augmentation purpose. Shear range is shearing the angle in counterclockwisedirection in degrees. The degree that we implement here is 0.2. Zoom  range  is  randomly  zoom  the  input.  Horizontal  flip  is randomly  flipping  inputs  horizontally. Finally,  all  the  data  is resized into 100x100 to be fed as the input.For  CNN  training,we  employ  Adam  for  the  optimization algorithm and  Cross  Entropy  function  to  calculate  loss.  The total  epoch  is  50,with  total  step  per  epoch equals  thetotal training data divided by 10 (1,929). Hence, there area total of 96,450 iterations in this training. From the total of 27,554 data, we  split  it into  70%  for  training  (19.290), and 15% each for testing (4132), and validating (4132).
