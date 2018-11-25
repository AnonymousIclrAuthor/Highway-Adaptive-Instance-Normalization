# Local Image to Image Translation Via Pixel wise Highway Adaptive Instance Normalization

## Demo
* A demo is in a following link : http://123.108.168.4:5000/

## Usage

##### Prerequisite
* python3.6  
* Pytorch0.4
* matplotlib 2.2.2

##### To download the CelebA:
>$ bash data_download.sh

##### Train
* We have used a jupyter notebook for the training. (An example is in "Hair color, Non-smile.ipynb")


## The result of LOcal Mask based Image Translation (LOMIT)

##### Comparison with StarGAN (black, smile) => (blonde, non-smile)
<img src="https://user-images.githubusercontent.com/45184715/48979753-a2e5d500-f102-11e8-9ef2-d9395ca05a7d.png" width="90%"></img>

From the top, each row denotes input image, an exemplar and a corresponding output.

##### Other results (Facial Hair and Gender translation)

First row of each macro row denotes an input image and the second row indicates an exemplar. The third row is a corresponding output.

<img src="https://user-images.githubusercontent.com/45184715/48979761-aed19700-f102-11e8-8eae-4bba2d08e28d.png" width="90%"></img>

------------------------------------------------------------------------------------------------------------------------------------------

<img src="https://user-images.githubusercontent.com/45184715/48979765-b6913b80-f102-11e8-8af2-f561f94ae0e8.png" width="90%"></img>

------------------------------------------------------------------------------------------------------------------------------------------

<img src="https://user-images.githubusercontent.com/45184715/48979770-bf820d00-f102-11e8-8f13-cc9f6e815c92.png" width="90%"></img>
