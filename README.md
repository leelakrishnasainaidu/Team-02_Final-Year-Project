# Cartoon-GAN

## Description
This project takes on the problem of transferring
the style of cartoon images to real-life photographic images by
implementing previous work done by CartoonGAN. We trained
a Generative Adversial Network(GAN) on over 60 000 images.

## Dependencies

1.  Install Anaconda from https://www.anaconda.com/

2.  Install pytorch at: https://pytorch.org/get-started/locally/

3.  Install dependencies:

    ```
    python -m pip install tqdm pillow numpy matplotlib opencv-python
    ```

4. For predicting videos you will also need ffmpeg

## Weights
Weights for the presented models can be found [here](https://drive.google.com/drive/folders/1d_GsZncTGmMdYht0oUWG9pqvV4UqF_kM?usp=sharing)


## Training

All training code can be found in `experiment.ipynb`

## Predict

Predict by running `predict.py`.

Example:

```
python predict.py -i C:/folder/input_image.png -o ./output_folder/output_image.png
```

Predictions can be made on images, videos or a folder of images/videos.
