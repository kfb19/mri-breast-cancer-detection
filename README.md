
# Breast Cancer Detection - Data Collection, Training, API and Frontend 

The project includes various components such as data collection, CNN training, and an API for breast cancer detection. It also encompasses experiments on different types of convolutional neural networks. 


## Installation

First, Python 3.10.7 should be installed, as well as a code editor (i.e., Visual Studio Code). 

Nagivate to the program directory and install the packages defined in requirements.txt . 

```bash
pip install -r requirements.txt

```

This should install all requirements for the program. In order to run the API, Docker will also need to be installed. Please see the API README for more information. 
## Run Locally

Nagivate to the program directory. For every training, data collection and evaluation file, please change "PATH" to the filepath of where your data is/will be stored. 

To run the data collection program, first download the dataset from the [Cancer Imaging Archive](https://nbia.cancerimagingarchive.net/nbia-search/?MinNumberOfStudiesCriteria=1&CollectionCriteria=Duke-Breast-Cancer-MRI). Next, run these commands: 

```bash
  cd Backend
  cd data_collection
  python chosen_data_collection_file.py 
```

To run the CNN training: 

```bash
  cd Backend
  cd classification
  cd cnn_type_folder
  python chosen_cnn_to_train.py 
```

Please view the README for the API for information on how to run locally. 

## File Structure

The file structure in this repository is as follows: 

### Backend 

#### classification (Python files for CNN training)
- alexnet
- densenet
- resnet50 
- vgg 

Each of the above folders these files: 
- evaluation.py, a class used to calculate the evaluation metrics. 
- early_stopper.py, a class used to prevent overfitting. 

They also contain the CNN training files. The title key is explained here: 
- The first word is the CNN type, e.g. "alexnet". 
- The second word is the training type; "consec", "scantype" or "single". 
- If it includes the word "pretrained" then the network is using transfer learning, otherwise it is trained from scratch. 
- If it includes the word "multi", it is using multi-margin loss, otherwise it is using cross entropy loss. 
- E.g., alexnet_consec_pretrained.py 

#### data_collection (Python files for collecting and pre-processing training data)

data_collection_consec_classify.py (consecutive image input data)

data_collection_scantype_classify.py (scantype image input data)

data_collection_single_classify.py (single channel image input data)

#### django 

This folder contains the code for the API. Please see the API README for more information. 

## Developer Documentation

### Data Collection Functions 

#### read_data()

This function reads the DICOM data information about filename mapping through the mapping path and what scan slices contain tumours through the bounding box data. 

#### save_dicom_to_bitmap()

This function saves each DICOM file to a bitmap, in a folder depending on its label ("pos" or "neg"). For consecutive and scantype three channel input, the data is stored in sub-folders in groups of three. 

#### determine_pos_neg()

This function is only present in the consecutive scan data collection. It determines whether a group of three consecutive images are overall "cancerous" or "non-cancerous"; if at least one of the three scans contain a tumour, the overall label is "cancerous". 

#### main()

This function runs the data collection code in order to save the data required for training. 

### Training Functions 

#### class ScanDataset(Dataset)

This class is used to prepare the data for training, testing and validating a CNN. 

#### \_\_init__()

This function initialises the ScanDataset class. 

#### create_labels()

This function creates and stores labels (positive or negative) for scans within the dataset. It converts scans to tensors. Each label is a tuple: (image tensor, label number (0 or 1)). For three channel input methods, three images are concatenated into the RGB channels on tensor creation. 

#### normalize()

This function normalises image pixel values to range [0, 255].

#### \_\_getitem__()

This function is used to return an item from the dataset. 

#### \_\_len__()

This is the last ScanDataset class function. It is used to return the length of the dataset. 

#### plot_imgbatch()

This is a helper function to plot a random batch of images for the model to make predictions on. 

#### main()

This function calls the ScanDataset class to create data loaders for training, validation and testing. It then runs the training loop over 100 epochs for each neural network. It sets the network, evaluates it, and displays and saves the model and evaluation metric results. 

### API and Frontend Functions 

Please see the API README for more information on its specific functions. 
## Features

- Data collection for single, consecutive and scantype CNNs.
- Training for several ResNet50 models.
- Training for several VGG19 models.
- Training for several DenseNet201 models.
- Training for several AlexNet models.
- A Django API using the resulting best models. 

## Tech

The code was developed in Visual Studio Code. It was written in Python 3.10.7, using PyTorch 1.13.1+cu116. The API was developed in Django, using Nginx and Gunicorn to run in Docker (please view the API README file for more information). 

## Support

For support, email kate.belson@hotmail.com .


## Authors

- [Kate Belson](https://github.com/kfb19)

## Code References

Al Husaini, M.A.S., Habaebi, M.H., Gunawan, T.S., Islam, M.R., Hameed, S.A.: Automatic Breast Cancer Detection Using Inception V3 in Thermography. In: 2021 8th International Conference on Computer and Communication Engineering (ICCCE), pp. 255–258 (2021). IEEE

Eskreis-Winkler, S., Onishi, N., Pinker, K., Reiner, J.S., Kaplan, J., Morris, E.A., Sutton, E.J.: Using Deep Learning to Improve Non-systematic Viewing of Breast Cancer on MRI. Journal of Breast Imaging 3(2), 201–207 (2021) https://doi.org/10.1093/jbi/wbaa102 https://academic.oup.com/jbi/article-pdf/3/2/201/36648790/wbaa102.pdf

Goncalves, C.B., Souza, J.R., Fernandes, H.: Classification of Static Infrared Images using Pre-Trained CNN for Breast Cancer Detection. In: 2021 IEEE 34th International Symposium on Computer-Based Medical Systems (CBMS), pp. 101–106 (2021). IEEE

Hui, L., Belkin, M.: Evaluation of Neural Architectures Trained with Square Loss vs Cross-Entropy in Classification Tasks. arXiv preprint arXiv:2006.07322 (2020)

Konz, N.: Train a Neural Network to Detect Breast MRI Tumors with PyTorch. 22 Towards Data Science (2022). Accessed: 15-04-23

Mahoro, E., Akhloufi, M.A.: Breast Cancer Classification on Thermograms Using Deep CNN and Transformers. Quantitative InfraRed Thermography Journal 0(0), 1–20 (2022) https://doi.org/10.1080/17686733.2022.2129135

Nguyen, B.H., Le, B., Huynh, T., Le, H., Pham, T.H.: Breast Cancer Diagnosis Based on Detecting Lymph Node Metastases Using Deep Learning. VNUHCM Journal of Science and Technology Development 25(2), 2381–2389 (2022) https: //doi.org/10.32508/stdj.v25i2.3894

Saha, A., Harowicz, M., Grimm, L., Kim, C., Ghate, S., Walsh, R., Mazurowski, M.: A Machine Learning Approach to Radiogenomics of Breast Cancer: a Study of 922 Subjects and 529 DCE-MRI Features. The British Journal of Cancer 119(4), 508–516 (2018)

Saha, A., Harowicz, M., Grimm, L., Weng, J., Cain, E., Kim, C., Ghate, S., Walsh, R., Mazurowski, M.: Dynamic Contrast-Enhanced Magnetic Resonance Images of Breast Cancer Patients with Tumor Locations \[Dataset]. The Cancer Imaging Archive (2021)

Truhn, D., Schrading, S., Haarburger, C., Schneider, H., Merhof, D., Kuhl, C.: Radiomic versus Convolutional Neural Networks Analysis for Classification of Contrast-Enhancing Lesions at Multiparametric Breast MRI. Radiology 290(2), 290–297 (2019) https://doi.org/10.1148/radiol.2018181352

## License

[MIT](https://choosealicense.com/licenses/mit/)

