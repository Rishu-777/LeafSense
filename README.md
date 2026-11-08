# LeafSense — Plant Disease Detection

Plant disease detection using deep learning. A convolutional neural network in PyTorch classifies leaf images into 39 categories. Training uses the PlantVillage dataset.

Live repo: https://github.com/Rishu-777/LeafSense

## Run on your machine

1. Install **Python 3.8+**.
2. Create and activate a virtual environment.
3. Install dependencies:

```
pip install -r requirements.txt
```

4. Go to the `Flask Deployed App` folder.
5. Download the pre-trained model file `plant_disease_model_1_latest.pt` from [Google Drive](https://drive.google.com/drive/folders/1ewJWAiduGuld_9oGSrTuLumg9y62qS6A?usp=share_link) and put it in `Flask Deployed App`. Model weight files are not stored in this GitHub repo because they are larger than GitHub’s file limit.
6. Run the Flask app:

```
python app.py
```

You can also use the downloaded model in the `Model` section with Jupyter Notebook.

## Testing images

If you do not have leaf images, use samples in the `test_images` folder. Each image has a corresponding disease name so you can check whether the model is working.
