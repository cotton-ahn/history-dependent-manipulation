# History-Dependent-Manipulation
An official data + code for "Visually Grounding Language Instruction for History-Dependent Manipulation" (ICRA 2022)

## Dataset
- After cloning this repository, you would see the data folder, which has everything for training and testing.
- It consists of 300 history-dependent-manipulation tasks.
- Each "scene_xxxx" folder has: bbox, heatmap, image, meta folders
    - "image" has images observing the workspace from bird-eye-view and from front-side.
    - "bbox" has information of cube objects existing in image.
    - "meta" has (1) language annotation and (2) bounding-box information for pick-and-place (3) whether it has explicit/implicit dependency.
        - "0_1.json" has manipulation information between image 0 and image 1. 
    - "heatmap" has ground truth heatmaps for pick and place
        - "pick/place_0_1.json" show heatmap for manipulation between image 0 and image 1.

## Network Training


## Network Validation


## Installation for Data Generation
#### Pre-requisite
- Blender (tested on v.2.78)
- Python 3.7

#### Instructions for Image generation
1. Clone the repository
```
git clone https://github.com/cotton-ahn/history-dependent-manipulation
cd history-dependent-manipulation
```
2. Clone the CLEVR dataset generation code for data generation
```
git clone https://github.com/facebookresearch/clevr-dataset-gen
```
3. Follow instructions from [CLEVR](https://github.com/facebookresearch/clevr-dataset-gen), and make sure your system is possible to generate images.
4. run `./generation_setup.sh` to copy files for proper image generation
5. go to `clevr-dataset-gen/image_generation`
6. run as below to generate image
```
# with GPU, generate episode that always stack at least one block.
blender --background --python render_images_with_stack.py -- --use_gpu 1

# with GPU, generate episode without constraints about stack as above.
blender --background --python render_images_wo_stack.py -- --use_gpu 1

# with GPU, generate scene with only rubber blocks, and move block for 5 times.
blender --background --python render_images_wo_stack.py -- --use_gpu 1 --materials rubber --num_moves 5

```
7. Image data will be saved to `clevr-dataset-gen/output`.
8. To find how to annotate bounding box from the generated files, 
   - refer to `{this repository}/find_bbox_info.pynb`

## TO-DOs
- organize data
- count the number of data again
- organize/upgrade code
- check the reproducibility with the best model and test cases. 
- check whether the similar result can be reproduced.
- Organize and Uploade code for image synthesis. 

## Code for model training
### Environment Preparation

## Data
### Download
### Usage and Visualization

## Code for synthetic image generation
### Environment Preparation
### How to run this code
