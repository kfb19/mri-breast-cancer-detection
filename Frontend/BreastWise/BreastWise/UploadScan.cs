using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO.Compression;
using FellowOakDicom;
using FellowOakDicom.Imaging;
using System.Drawing;
using System.Drawing.Imaging;
using Aspose.Imaging.FileFormats.Dicom;
using Aspose.Imaging.ImageOptions;
using Newtonsoft.Json.Linq;
using System.Net.Http.Headers;
using System.Net;
using System.Diagnostics.Eventing.Reader;

namespace BreastWise
{
    public partial class UploadScan : Form
    {
        private string[] dicomFiles;
        List<int> cancerousSlices;
        private PictureBox pictureBox;
        private string zipPath;
        private bool analysed;
        private int index;


        public UploadScan()
        {
            InitializeComponent();
            analysed = false;
            pictureBox = new PictureBox();
            pictureBox.SizeMode = PictureBoxSizeMode.AutoSize;
            pictureBox.Anchor = AnchorStyles.None;
            scanPanel.Controls.Add(pictureBox);
            exportBtn.Enabled = false;
            scanScrollBar.Enabled = false;
            resultsPanel.Visible = false;
            statusLab1.Visible = false;
            statusLab2.Visible = false;
            deleteBtn.Enabled = false;
            statusLab3.Visible = false;
            statusLab4.Visible = false;
            cancerSliceLab.Visible = false;
            cancerStatusLab.Visible = false;
            analyseBtn.BackColor = System.Drawing.ColorTranslator.FromHtml("#1C746A");
            analyseBtn.Enabled = false;
            this.FormClosing += new FormClosingEventHandler(UploadScan_FormClosing);
        }

        private void UploadScan_FormClosing(object sender, FormClosingEventArgs e)
        {
            // Delete the temporary directory and files
            string tempFolder = Path.GetTempPath();
            string extractedFolder = Path.Combine(tempFolder, "extractedBreastwise");

            if (Directory.Exists(extractedFolder))
            {
                Directory.Delete(extractedFolder, true);
            }
        }

        private void uploadBtn_Click(object sender, EventArgs e)
        {
            // Show the OpenFileDialog.
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "Zip files (*.zip)|*.zip|All files (*.*)|*.*";
            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                // Get the selected file path.
                string filePath = openFileDialog.FileName;

                // Save the file to a temporary directory.
                string tempFolder = Path.GetTempPath();
                string destinationPath = Path.Combine(tempFolder, "series.zip");
                File.Copy(filePath, destinationPath, true);

                // Rename the file.
                zipPath = Path.Combine(tempFolder, "series.zip");
                File.Move(destinationPath, zipPath);

                string newTemp = Path.GetTempPath();
                string extractedFolder = Path.Combine(newTemp, "extractedBreastwise");
                ZipFile.ExtractToDirectory(filePath, extractedFolder);


                string[] requiredFolders = { "pre", "1st_pass", "2nd_pass", "3rd_pass" };
                foreach (string folder in requiredFolders)
                {
                    string folderPath = Path.Combine(extractedFolder, folder);
                    if (!Directory.Exists(folderPath))
                    {
                        // Display an error message and delete the files.
                        MessageBox.Show("The required folder " + folder + " was not found in the zip file.");
                        Directory.Delete(extractedFolder, true);
                        File.Delete(filePath);
                        return;
                    }
                }

                int[] fileCounts = new int[4];
                for (int i = 0; i < 4; i++)
                {
                    string folderPath = Path.Combine(extractedFolder, requiredFolders[i]);
                    fileCounts[i] = Directory.GetFiles(folderPath).Length;
                }

                if (fileCounts.Distinct().Count() > 1)
                {
                    // Display an error message and delete the files.
                    MessageBox.Show("The number of files in the folders does not match.");
                    Directory.Delete(extractedFolder, true);
                    File.Delete(filePath);
                    return;
                }

                // Get the DICOM files in the "pre" folder.
                string preFolder = Path.Combine(extractedFolder, "pre");
                dicomFiles = Directory.GetFiles(preFolder, "*.dcm");

                if (dicomFiles.Length == 0)
                {
                    // Display an error message and delete the files.
                    MessageBox.Show("Error: No DICOM files found in the \"pre\" folder.");
                    File.Delete(destinationPath);
                    File.Delete(zipPath);
                    return;
                }


                if (dicomFiles.Length > 0)
                {
                    // Load first image
                    using (var fileStream = new FileStream(dicomFiles[0], FileMode.Open, FileAccess.Read))
                    {
                        using (Aspose.Imaging.FileFormats.Dicom.DicomImage dicomImage = new Aspose.Imaging.FileFormats.Dicom.DicomImage(fileStream))
                        {
                            dicomImage.ActivePage = (Aspose.Imaging.FileFormats.Dicom.DicomPage)dicomImage.Pages[0];

                            using (var ms = new MemoryStream())
                            {
                                var bmpOptions = new Aspose.Imaging.ImageOptions.BmpOptions();
                                dicomImage.Save(ms, bmpOptions);

                                pictureBox = new PictureBox();
                                pictureBox.Image = Image.FromStream(ms);
                                pictureBox.SizeMode = PictureBoxSizeMode.AutoSize;
                                pictureBox.Anchor = AnchorStyles.None;
                                pictureBox.Location = new Point((scanPanel.Width - pictureBox.Width) / 2, (scanPanel.Height - pictureBox.Height) / 2);
                                scanPanel.Controls.Add(pictureBox);
                                countLab.Text = "1/" + dicomFiles.Length.ToString();
                            }
                        }
                    }
                }
                // Set the scroll bar properties.
                int scrollMax = dicomFiles.Length - 1;
                scanScrollBar.Enabled = true;
                scanScrollBar.Maximum = scrollMax;
                scanScrollBar.Value = 0;
                scanScrollBar.SmallChange = 1;
                scanScrollBar.LargeChange = 1;
                analyseBtn.BackColor = System.Drawing.ColorTranslator.FromHtml("#4FA9A4");
                analyseBtn.Enabled = true;
                uploadBtn.BackColor = System.Drawing.ColorTranslator.FromHtml("#1C746A");
                uploadBtn.Enabled = false;
                deleteBtn.Enabled = true;
                // Set the scroll bar event handler.
                scanScrollBar.Scroll += new ScrollEventHandler(scanScrollBar_Scroll);
            }
        }

        private async void analyseBtn_Click(object sender, EventArgs e)
        {
            uploadMenu.BackColor = System.Drawing.ColorTranslator.FromHtml("#4FA9A4");
            analysingMenu.BackColor = System.Drawing.ColorTranslator.FromHtml("#1C746A");
            analyseBtn.BackColor = System.Drawing.ColorTranslator.FromHtml("#1C746A");
            analyseBtn.Enabled = false;
            deleteBtn.Enabled = false;
            int result = 0;
            scanScrollBar.Enabled = false;

            // Set API endpoint.
            string url = "http://127.0.0.1:8000/upload/";

            // Set up HTTP client.
            HttpClient client = new HttpClient();
            client.DefaultRequestHeaders.Accept.Clear();
            client.DefaultRequestHeaders.Accept.Add(
                new MediaTypeWithQualityHeaderValue("application/json"));
            client.Timeout = TimeSpan.FromSeconds(5000);

            // Create multipart form data.
            MultipartFormDataContent formData = new MultipartFormDataContent();
            FileStream fileStream = new FileStream(zipPath, FileMode.Open);
            formData.Add(new StreamContent(fileStream), "file", "series.zip");

            // Send POST request to API endpoint with form data.
            HttpResponseMessage response = await client.PostAsync(url, formData);

            // Check if API request was successful.
            if (response.IsSuccessStatusCode)
            {
                // Get JSON response from API.
                string jsonResponse = response.Content.ReadAsStringAsync().Result;

                // Deserialize JSON response into array.
                JArray jsonArray = JArray.Parse(jsonResponse);

                // Save array to variable.
                cancerousSlices = jsonArray.ToObject<List<int>>();
                result = 2;
            }
            else if (response.StatusCode == HttpStatusCode.InternalServerError)
            {
                MessageBox.Show("Server error, please try again.");
                result = 0;
            }
            else if (response.StatusCode == HttpStatusCode.BadRequest)
            {
                string errorMessage = await response.Content.ReadAsStringAsync();
                MessageBox.Show("Error: " + errorMessage);
                result = 1;
            }

            // At end.
            if (result == 2)
            {
                resultsPanel.Visible = true;
                statusLab1.Visible = true;
                statusLab2.Visible = true;
                statusLab3.Visible = true;
                statusLab4.Visible = true;
                exportBtn.Enabled = true;
                int totalCancer = cancerousSlices.Count;
                cancerSliceLab.Text = totalCancer + " SLICES";
                cancerSliceLab.Visible = true;
                scanScrollBar.Enabled = true;
                deleteBtn.Enabled = true;
                analysed = true;
                if (cancerousSlices.Contains(index))
                {
                    cancerStatusLab.Text = "POSITIVE";
                }
                else
                {
                    cancerStatusLab.Text = "NEGATIVE";
                }
                cancerStatusLab.Visible = true;
                analysingMenu.BackColor = System.Drawing.ColorTranslator.FromHtml("#4FA9A4");
                resultsMenu.BackColor = System.Drawing.ColorTranslator.FromHtml("#1C746A");
            }
            else
            {
                // Bad upload so gets ready to try again.
                UploadScan_FormClosing(this, new FormClosingEventArgs(CloseReason.None, false));
                pictureBox.Image = null;
                // Disable the scroll bar and remove the event handler.
                scanScrollBar.Scroll -= scanScrollBar_Scroll;
                scanScrollBar.Enabled = false;
                countLab.Text = "0/0";
                analysingMenu.BackColor = System.Drawing.ColorTranslator.FromHtml("#4FA9A4");
                resultsMenu.BackColor = System.Drawing.ColorTranslator.FromHtml("#4FA9A4");
                uploadMenu.BackColor = System.Drawing.ColorTranslator.FromHtml("#1C746A");
                resultsPanel.Visible = false;
                statusLab1.Visible = false;
                statusLab2.Visible = false;
                statusLab3.Visible = false;
                statusLab4.Visible = false;
                exportBtn.Enabled = false;
                cancerSliceLab.Visible = false;
                cancerStatusLab.Visible = false;
                uploadBtn.BackColor = System.Drawing.ColorTranslator.FromHtml("#4FA9A4");
                uploadBtn.Enabled = true;
                deleteBtn.Enabled = false;
                analyseBtn.Enabled = false;
            }
        }


        private void scanScrollBar_Scroll(object sender, ScrollEventArgs e)
        {
            index = e.NewValue; // Get the index of the selected DICOM file.

            // Open the selected DICOM file and display it on the scanPanel panel.
            using (var fileStream = new FileStream(dicomFiles[index], FileMode.Open, FileAccess.Read))
            {
                using (Aspose.Imaging.FileFormats.Dicom.DicomImage dicomImage = new Aspose.Imaging.FileFormats.Dicom.DicomImage(fileStream))
                {
                    dicomImage.ActivePage = (Aspose.Imaging.FileFormats.Dicom.DicomPage)dicomImage.Pages[0];

                    using (var ms = new MemoryStream())
                    {
                        var bmpOptions = new Aspose.Imaging.ImageOptions.BmpOptions();
                        dicomImage.Save(ms, bmpOptions);

                        pictureBox.Image = Image.FromStream(ms);
                        pictureBox.Location = new Point((scanPanel.Width - pictureBox.Width) / 2, (scanPanel.Height - pictureBox.Height) / 2);
                        countLab.Text = $"{index + 1}/{dicomFiles.Length}";
                    }
                }
            }
            if (analysed)
            {
                if (cancerousSlices.Contains(index))
                {
                    cancerStatusLab.Text = "POSITIVE";
                }
                else
                {
                    cancerStatusLab.Text = "NEGATIVE";
                }
            }
        }

        private void deleteBtn_Click(object sender, EventArgs e)
        {
            DialogResult result = MessageBox.Show("Are you sure you want to delete the uploaded scan?", "Delete", MessageBoxButtons.YesNo, MessageBoxIcon.Question);
            if (result == DialogResult.Yes)
            {
                UploadScan_FormClosing(this, new FormClosingEventArgs(CloseReason.None, false));
                pictureBox.Image = null;
                // Disable the scroll bar and remove the event handler.
                scanScrollBar.Scroll -= scanScrollBar_Scroll;
                scanScrollBar.Enabled = false;
                countLab.Text = "0/0";
                analysingMenu.BackColor = System.Drawing.ColorTranslator.FromHtml("#4FA9A4");
                resultsMenu.BackColor = System.Drawing.ColorTranslator.FromHtml("#4FA9A4");
                uploadMenu.BackColor = System.Drawing.ColorTranslator.FromHtml("#1C746A");
                resultsPanel.Visible = false;
                statusLab1.Visible = false;
                statusLab2.Visible = false;
                statusLab3.Visible = false;
                statusLab4.Visible = false;
                exportBtn.Enabled = false;
                analysed = false;
                cancerSliceLab.Visible = false;
                cancerStatusLab.Visible = false;
                uploadBtn.BackColor = System.Drawing.ColorTranslator.FromHtml("#4FA9A4");
                uploadBtn.Enabled = true;
                deleteBtn.Enabled = false;
            }
        }

        private void exportBtn_Click(object sender, EventArgs e)
        {
            if (cancerousSlices != null && cancerousSlices.Count > 0)
            {
                StringBuilder sb = new StringBuilder();
                sb.AppendLine("Cancerous Slices:");

                foreach (int slice in cancerousSlices)
                {
                    sb.AppendLine(slice.ToString());
                }

                File.WriteAllText("cancerous_slices.csv", sb.ToString());
                MessageBox.Show("Export successful!");
            }
            else if (cancerousSlices != null && cancerousSlices.Count == 0)
            {
                StringBuilder sb = new StringBuilder();
                sb.AppendLine("Cancerous Slices: None");

                File.WriteAllText("cancerous_slices.csv", sb.ToString());
                MessageBox.Show("Export successful!");
            }
            else
            {
                MessageBox.Show("Scan not analysed.");
            }
        }

    }
}
