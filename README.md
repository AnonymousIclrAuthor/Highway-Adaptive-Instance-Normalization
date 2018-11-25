# Local Image to Image Translation Via Pixel wise Highway Adaptive Instance Normalization

### Demo
* A demo is in a following link : http://123.108.168.4:5000/

##### Prerequisite
* python3.6  
* Pytorch0.4
* matplotlib 2.2.2

##### To download the CelebA:
>$ bash data_download.sh

##### Train
* We have used a jupyter notebook for the training. (An example is in "Hair color, Non-smile.ipynb")

### The result of LOcal Mask based Image Translation (LOMIT)
##### Comparison with StarGAN (black, smile) => (blonde, non-smile)
<img src="https://user-images.githubusercontent.com/20943085/46212917-2b99fc00-c372-11e8-8da0-384b7ba7418c.png" width="90%"></img>

### Our model overview
<img src="https://user-images.githubusercontent.com/20943085/47040629-a57b1380-d1c1-11e8-9635-32d449db0227.png" width="90%"></img>

### The cosegmentation module
<img src="https://user-images.githubusercontent.com/20943085/46211908-aada0080-c36f-11e8-9b1a-7d8f683313e4.png" width="90%"></img>

### Other results (Facial Hair and Gender translation)
<img src="https://user-images.githubusercontent.com/20943085/46211908-aada0080-c36f-11e8-9b1a-7d8f683313e4.png" width="90%"></img>
