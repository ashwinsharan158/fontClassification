# Font Classification
The task for this is to create a Font Classifier. The classifier should take a single image as input and return the name (or label) of the font used in the image from a predefined list of 10 fonts.
Fonts to Classify:
1.	Aguafina Script
2.	Alex Brush
3.	Allura
4.	Alsscrp
5.	Canterbury
6.	Great Vibes
7.	Holligate Signature
8.	I Love Glitter
9.	James Fajardo
10.	Open Sans

## How to Setup and Run the project
### If you want to learn about whole development process?
Please run the font_classification.ipynb file cell by cell and the text will explain the data collection and preparation techniques, model development and evaluation metrics. The .ipynb must rest in the same directory as the supplied data and font folder in this repo as it will make use of them to generate 20000 images which will be stored in data_train/ and test/ folders.
<br><br>
To create new model weights uncomment code cell 13 under Results.

### If you want to test the model against a held out dataset?
Run the first code cell of the .ipynb file and then go the 14th code block which is under the title "Model Evaluation on Test samples" to line 45 and replace "unseen_data_path = 'test'" with "unseen_data_path = 'path to the held out dataset'", then run the cell. Make sure your data is structured the same as that of the data/ folder. This will generate the Accuracy, precision, recall, F1 score, ROC and AUC score for the held out data for the model.

### If you want to just run a font image to find its font?
Make use of the fontPrediction.py by passing the image path as an argument.
Usage: `python fontPrediction.py image\path`
This file need to be in the same folder as the "final_font_class_10_model.pth" (provided in the repo) which are the model weights for the resnet. It is slow as it only takes a single image at a time. This file is not to be used in any other application its purpose is to just show the workings of the model.

## How in the model structured
The model is a pretrained resnet, with a extra fully connected layer to accomadate dropout. We load the model weights "final_font_class_10_model.pth" the input needs to be 128 X 128 tensor normalized mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225] To maximize accuracy of the model the input needs to be greyscale and needs to be fed into the model in patches of 128 X 128 at a stride of 28 pixels output is the majority prediction of all the image patches as demostrated in the <b>fontPrediction.py</b>

Please read fontPrediction.py to get a better understanding of the model.

