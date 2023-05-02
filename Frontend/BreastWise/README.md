
# BreastWise Frontend

This C# Windows Forms project acts as a frontend for the BreastWise API. It allows users to upload breast MRI scans and analyse them, so they can view which MRI scan slices may be cancerous. Data can then be exported to a .csv file. 


## Installation

This project should be downloaded from https://download.breast-wise.co.uk/ and unzipped. This must be done on a Windows machine. Find the BreastWise.exe file - this is the executable, click on it and the application will run. 
## Developer Notes

### Splash Screen 

#### public SplashScreenForm()

Defines the Splash Screen form. 

#### private void SplashScreenForm_Load(object sender, EventArgs e)

Loads the splash screen with the BreastWise logo. 

#### private void timer1_Tick_2(object sender, EventArgs e)

Shows the splash screen for 5 seconds before calling UploadScan(). 

### Upload Scan 

#### public UploadScan()

Initialises the upload scan form and the states of buttons and labels. 

#### private void UploadScan_FormClosing(object sender, FormClosingEventArgs e)

On form closing, delete all the temporary files. 

#### private void uploadBtn_Click(object sender, EventArgs e)

Upload an MRI scan in a .zip format. The scan should contain four folders: 
- 1st_pass
- 2nd_pass 
- 3rd_pass 
- pre 
Each folder should contain the same number of DICOM files, corresponding to the MRI series as labelled by the folder name. The files in each folder should be named the same way, e.g. each folder contains a 001.dcm. 

The function returns an error if the .zip is not in the correct format, and the user is made aware. Otherwise, the "pre" folder is used to convert the scan to a bitmap and load it onto the screen for the user to view and scroll through the scan slices. 

#### private async void analyseBtn_Click(object sender, EventArgs e)

This function makes a request to the BreastWise API, with the .zip file in the request body. The buttons are locked and the application waits to receive a response. On recieving the response, the export button becomes available, and the scan is clearly labelled with the number of positive slices, and which slices are positive. 

#### private void scanScrollBar_Scroll(object sender, ScrollEventArgs e)

Allows users to scroll through the scan slices with the scroll bar, and if the scan has been analysed, displays whether the slices are positive or negative. 

#### private void deleteBtn_Click(object sender, EventArgs e)

Deletes the scan from the application, as well as all temporary folders created. 

#### private void exportBtn_Click(object sender, EventArgs e)

Exports the analysed scan data to a .csv file. 
## Tech

The program is written in C# Windows Forms. The best way to run this program is via Visual Studio 2022, which has great support for C# Windows Forms. Please note that the BreastWise API (downloadable from https://download.breast-wise.co.uk/) is required for this program to work effectively. 

## Usage

This application should be used for breast cancer detection. Please note, no diagnosis should be made without the advice of a breast radiologist or an oncologist. 
## Features

- Upload scan
- Analyse scan
- Delete scan
- Export scan data 


## Support

For support, email kfb206@exeter.ac.uk .


## Authors

- [Kate Belson](https://github.com/kfb19)


## License

[MIT](https://choosealicense.com/licenses/mit/)

