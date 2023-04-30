using System;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Net.Http;
using System.Text.Json;
using System.Text.Json.Serialization;
using Amazon.Runtime.Internal;

namespace frontend
{
    public partial class uploadForm : Form
    {
        public string userEmail { get; set; }
        public uploadForm(String user_email)
        {
            InitializeComponent();
            userEmail = user_email; // Sets the variable to the value of the passed-in parameters.
        }

        private void analyseButton_Click(object sender, EventArgs e)
        {

        }

        private async void uploadButton_ClickAsync(object sender, EventArgs e)
        {
            // Show the OpenFileDialog.
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "Zip files (*.zip)|*.zip|All files (*.*)|*.*";
            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                // Get the selected file path.
                string filePath = openFileDialog.FileName;

                // Save the file to a temporary directory
                string tempFolder = Path.GetTempPath();
                string destinationPath = Path.Combine(tempFolder, "series.zip");
                File.Copy(filePath, destinationPath, true);

                // Rename the file.
                string renamedPath = Path.Combine(tempFolder, "series.zip");
                File.Move(destinationPath, renamedPath);
            }

            // Create the HTTP client.
            var client = new HttpClient();

            // Create the request body.
            var body = new
            {
                file = "series.zip"
            };

            // Serialize the body to JSON.
            var json = JsonSerializer.Serialize(body);

            // Create the request content with the serialized JSON.
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            // Send the POST request to the API endpoint.
            var response = await client.PostAsync("https://api.example.com/upload", content);

            // Check if the request was successful.
            if (response.IsSuccessStatusCode)
            {
                // Handle the successful response.
                var responseJson = await response.Content.ReadAsStringAsync();
                var result = JsonSerializer.Deserialize<Response>(responseJson);
                Console.WriteLine(result.Message);
            }
            else
            {
                // Handle the error response.
                var responseJson = await response.Content.ReadAsStringAsync();
                var error = JsonSerializer.Deserialize<ErrorResponse>(responseJson);
                MessageBox.Show("Invalid file upload: " + error.Message);

                // Delete the uploaded file
                string filePath = @"C:\MyFiles\series.zip";
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }

            }

        }

        private void uploadScanButton_Click(object sender, EventArgs e)
        {

        }
    }
}
