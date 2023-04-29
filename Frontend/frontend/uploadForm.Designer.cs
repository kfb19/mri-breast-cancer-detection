namespace frontend
{
    partial class uploadForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(uploadForm));
            this.exportButton = new System.Windows.Forms.Button();
            this.cancerResultsLab = new System.Windows.Forms.Label();
            this.results4Lab = new System.Windows.Forms.Label();
            this.results3Label = new System.Windows.Forms.Label();
            this.cancerSlidesLab = new System.Windows.Forms.Label();
            this.results2Lab = new System.Windows.Forms.Label();
            this.panel3 = new System.Windows.Forms.Panel();
            this.results1Label = new System.Windows.Forms.Label();
            this.sliceNoButton = new System.Windows.Forms.Button();
            this.deleteButton = new System.Windows.Forms.Button();
            this.uploadButton = new System.Windows.Forms.Button();
            this.scanScrollBar = new System.Windows.Forms.HScrollBar();
            this.analyseButton = new System.Windows.Forms.Button();
            this.panel2 = new System.Windows.Forms.Panel();
            this.analysingButton = new System.Windows.Forms.Button();
            this.viewResultsButton = new System.Windows.Forms.Button();
            this.saveScanButton = new System.Windows.Forms.Button();
            this.archiveButton = new System.Windows.Forms.Button();
            this.logOutButton = new System.Windows.Forms.Button();
            this.uploadScanButton = new System.Windows.Forms.Button();
            this.panel1 = new System.Windows.Forms.Panel();
            this.openFileDialog1 = new System.Windows.Forms.OpenFileDialog();
            this.panel3.SuspendLayout();
            this.panel2.SuspendLayout();
            this.SuspendLayout();
            // 
            // exportButton
            // 
            this.exportButton.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(28)))), ((int)(((byte)(116)))), ((int)(((byte)(106)))));
            this.exportButton.FlatAppearance.BorderColor = System.Drawing.Color.Black;
            this.exportButton.FlatAppearance.BorderSize = 0;
            this.exportButton.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.exportButton.Font = new System.Drawing.Font("Segoe UI Semibold", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.exportButton.Location = new System.Drawing.Point(807, 341);
            this.exportButton.Name = "exportButton";
            this.exportButton.Size = new System.Drawing.Size(106, 47);
            this.exportButton.TabIndex = 33;
            this.exportButton.Text = "Export";
            this.exportButton.UseVisualStyleBackColor = false;
            // 
            // cancerResultsLab
            // 
            this.cancerResultsLab.AutoSize = true;
            this.cancerResultsLab.Font = new System.Drawing.Font("Segoe UI Semibold", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.cancerResultsLab.Location = new System.Drawing.Point(10, 117);
            this.cancerResultsLab.Name = "cancerResultsLab";
            this.cancerResultsLab.Size = new System.Drawing.Size(77, 21);
            this.cancerResultsLab.TabIndex = 5;
            this.cancerResultsLab.Text = "POSITIVE";
            // 
            // results4Lab
            // 
            this.results4Lab.AutoSize = true;
            this.results4Lab.Font = new System.Drawing.Font("Segoe UI Semibold", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.results4Lab.Location = new System.Drawing.Point(24, 100);
            this.results4Lab.Name = "results4Lab";
            this.results4Lab.Size = new System.Drawing.Size(48, 17);
            this.results4Lab.TabIndex = 4;
            this.results4Lab.Text = "status:";
            // 
            // results3Label
            // 
            this.results3Label.AutoSize = true;
            this.results3Label.Font = new System.Drawing.Font("Segoe UI Semibold", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.results3Label.Location = new System.Drawing.Point(6, 83);
            this.results3Label.Name = "results3Label";
            this.results3Label.Size = new System.Drawing.Size(85, 17);
            this.results3Label.TabIndex = 3;
            this.results3Label.Text = "Current slide";
            // 
            // cancerSlidesLab
            // 
            this.cancerSlidesLab.AutoSize = true;
            this.cancerSlidesLab.Font = new System.Drawing.Font("Segoe UI Semibold", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.cancerSlidesLab.Location = new System.Drawing.Point(2, 39);
            this.cancerSlidesLab.Name = "cancerSlidesLab";
            this.cancerSlidesLab.Size = new System.Drawing.Size(95, 21);
            this.cancerSlidesLab.TabIndex = 2;
            this.cancerSlidesLab.Text = "5/160 slides";
            // 
            // results2Lab
            // 
            this.results2Lab.AutoSize = true;
            this.results2Lab.Font = new System.Drawing.Font("Segoe UI Semibold", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.results2Lab.Location = new System.Drawing.Point(6, 22);
            this.results2Lab.Name = "results2Lab";
            this.results2Lab.Size = new System.Drawing.Size(84, 17);
            this.results2Lab.TabIndex = 1;
            this.results2Lab.Text = "detected on:";
            // 
            // panel3
            // 
            this.panel3.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(28)))), ((int)(((byte)(116)))), ((int)(((byte)(106)))));
            this.panel3.Controls.Add(this.cancerResultsLab);
            this.panel3.Controls.Add(this.results4Lab);
            this.panel3.Controls.Add(this.results3Label);
            this.panel3.Controls.Add(this.cancerSlidesLab);
            this.panel3.Controls.Add(this.results2Lab);
            this.panel3.Controls.Add(this.results1Label);
            this.panel3.Location = new System.Drawing.Point(807, 150);
            this.panel3.Name = "panel3";
            this.panel3.Size = new System.Drawing.Size(106, 148);
            this.panel3.TabIndex = 21;
            this.panel3.Visible = false;
            // 
            // results1Label
            // 
            this.results1Label.AutoSize = true;
            this.results1Label.Font = new System.Drawing.Font("Segoe UI Semibold", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.results1Label.Location = new System.Drawing.Point(24, 5);
            this.results1Label.Name = "results1Label";
            this.results1Label.Size = new System.Drawing.Size(49, 17);
            this.results1Label.TabIndex = 0;
            this.results1Label.Text = "Cancer";
            // 
            // sliceNoButton
            // 
            this.sliceNoButton.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(28)))), ((int)(((byte)(116)))), ((int)(((byte)(106)))));
            this.sliceNoButton.FlatAppearance.BorderColor = System.Drawing.Color.Black;
            this.sliceNoButton.FlatAppearance.BorderSize = 0;
            this.sliceNoButton.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.sliceNoButton.Font = new System.Drawing.Font("Segoe UI Semibold", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.sliceNoButton.Location = new System.Drawing.Point(807, 500);
            this.sliceNoButton.Name = "sliceNoButton";
            this.sliceNoButton.Size = new System.Drawing.Size(106, 47);
            this.sliceNoButton.TabIndex = 32;
            this.sliceNoButton.Text = "11/160";
            this.sliceNoButton.UseVisualStyleBackColor = false;
            // 
            // deleteButton
            // 
            this.deleteButton.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(79)))), ((int)(((byte)(169)))), ((int)(((byte)(164)))));
            this.deleteButton.FlatAppearance.BorderColor = System.Drawing.Color.Black;
            this.deleteButton.FlatAppearance.BorderSize = 0;
            this.deleteButton.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.deleteButton.Font = new System.Drawing.Font("Segoe UI Semibold", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.deleteButton.Location = new System.Drawing.Point(807, 447);
            this.deleteButton.Name = "deleteButton";
            this.deleteButton.Size = new System.Drawing.Size(106, 47);
            this.deleteButton.TabIndex = 31;
            this.deleteButton.Text = "Delete";
            this.deleteButton.UseVisualStyleBackColor = false;
            // 
            // uploadButton
            // 
            this.uploadButton.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(79)))), ((int)(((byte)(169)))), ((int)(((byte)(164)))));
            this.uploadButton.FlatAppearance.BorderColor = System.Drawing.Color.Black;
            this.uploadButton.FlatAppearance.BorderSize = 0;
            this.uploadButton.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.uploadButton.Font = new System.Drawing.Font("Segoe UI Semibold", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.uploadButton.Location = new System.Drawing.Point(807, 394);
            this.uploadButton.Name = "uploadButton";
            this.uploadButton.Size = new System.Drawing.Size(106, 47);
            this.uploadButton.TabIndex = 30;
            this.uploadButton.Text = "Upload";
            this.uploadButton.UseVisualStyleBackColor = false;
            this.uploadButton.Click += new System.EventHandler(this.uploadButton_ClickAsync);
            // 
            // scanScrollBar
            // 
            this.scanScrollBar.Location = new System.Drawing.Point(-5, 417);
            this.scanScrollBar.Name = "scanScrollBar";
            this.scanScrollBar.Size = new System.Drawing.Size(585, 31);
            this.scanScrollBar.TabIndex = 0;
            // 
            // analyseButton
            // 
            this.analyseButton.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(79)))), ((int)(((byte)(169)))), ((int)(((byte)(164)))));
            this.analyseButton.FlatAppearance.BorderColor = System.Drawing.Color.Black;
            this.analyseButton.FlatAppearance.BorderSize = 0;
            this.analyseButton.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.analyseButton.Font = new System.Drawing.Font("Segoe UI Semibold", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.analyseButton.Location = new System.Drawing.Point(807, 97);
            this.analyseButton.Name = "analyseButton";
            this.analyseButton.Size = new System.Drawing.Size(106, 47);
            this.analyseButton.TabIndex = 29;
            this.analyseButton.Text = "Analyse";
            this.analyseButton.UseVisualStyleBackColor = false;
            this.analyseButton.Click += new System.EventHandler(this.analyseButton_Click);
            // 
            // panel2
            // 
            this.panel2.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(117)))), ((int)(((byte)(117)))), ((int)(((byte)(117)))));
            this.panel2.Controls.Add(this.scanScrollBar);
            this.panel2.Location = new System.Drawing.Point(212, 97);
            this.panel2.Name = "panel2";
            this.panel2.Size = new System.Drawing.Size(585, 450);
            this.panel2.TabIndex = 22;
            // 
            // analysingButton
            // 
            this.analysingButton.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(79)))), ((int)(((byte)(169)))), ((int)(((byte)(164)))));
            this.analysingButton.FlatAppearance.BorderColor = System.Drawing.Color.Black;
            this.analysingButton.FlatAppearance.BorderSize = 0;
            this.analysingButton.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.analysingButton.Font = new System.Drawing.Font("Segoe UI Semibold", 15.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.analysingButton.Location = new System.Drawing.Point(0, 173);
            this.analysingButton.Name = "analysingButton";
            this.analysingButton.Size = new System.Drawing.Size(206, 70);
            this.analysingButton.TabIndex = 28;
            this.analysingButton.Text = "Analysing";
            this.analysingButton.UseVisualStyleBackColor = false;
            // 
            // viewResultsButton
            // 
            this.viewResultsButton.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(79)))), ((int)(((byte)(169)))), ((int)(((byte)(164)))));
            this.viewResultsButton.FlatAppearance.BorderColor = System.Drawing.Color.Black;
            this.viewResultsButton.FlatAppearance.BorderSize = 0;
            this.viewResultsButton.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.viewResultsButton.Font = new System.Drawing.Font("Segoe UI Semibold", 15.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.viewResultsButton.Location = new System.Drawing.Point(0, 249);
            this.viewResultsButton.Name = "viewResultsButton";
            this.viewResultsButton.Size = new System.Drawing.Size(206, 70);
            this.viewResultsButton.TabIndex = 27;
            this.viewResultsButton.Text = "View Results";
            this.viewResultsButton.UseVisualStyleBackColor = false;
            // 
            // saveScanButton
            // 
            this.saveScanButton.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(79)))), ((int)(((byte)(169)))), ((int)(((byte)(164)))));
            this.saveScanButton.FlatAppearance.BorderColor = System.Drawing.Color.Black;
            this.saveScanButton.FlatAppearance.BorderSize = 0;
            this.saveScanButton.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.saveScanButton.Font = new System.Drawing.Font("Segoe UI Semibold", 15.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.saveScanButton.Location = new System.Drawing.Point(0, 325);
            this.saveScanButton.Name = "saveScanButton";
            this.saveScanButton.Size = new System.Drawing.Size(206, 70);
            this.saveScanButton.TabIndex = 26;
            this.saveScanButton.Text = "Save Scan";
            this.saveScanButton.UseVisualStyleBackColor = false;
            // 
            // archiveButton
            // 
            this.archiveButton.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(79)))), ((int)(((byte)(169)))), ((int)(((byte)(164)))));
            this.archiveButton.FlatAppearance.BorderColor = System.Drawing.Color.Black;
            this.archiveButton.FlatAppearance.BorderSize = 0;
            this.archiveButton.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.archiveButton.Font = new System.Drawing.Font("Segoe UI Semibold", 15.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.archiveButton.Location = new System.Drawing.Point(0, 401);
            this.archiveButton.Name = "archiveButton";
            this.archiveButton.Size = new System.Drawing.Size(206, 70);
            this.archiveButton.TabIndex = 25;
            this.archiveButton.Text = "Archive";
            this.archiveButton.UseVisualStyleBackColor = false;
            // 
            // logOutButton
            // 
            this.logOutButton.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(79)))), ((int)(((byte)(169)))), ((int)(((byte)(164)))));
            this.logOutButton.FlatAppearance.BorderColor = System.Drawing.Color.Black;
            this.logOutButton.FlatAppearance.BorderSize = 0;
            this.logOutButton.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.logOutButton.Font = new System.Drawing.Font("Segoe UI Semibold", 15.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.logOutButton.Location = new System.Drawing.Point(0, 477);
            this.logOutButton.Name = "logOutButton";
            this.logOutButton.Size = new System.Drawing.Size(206, 70);
            this.logOutButton.TabIndex = 24;
            this.logOutButton.Text = "Log Out";
            this.logOutButton.UseVisualStyleBackColor = false;
            // 
            // uploadScanButton
            // 
            this.uploadScanButton.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(79)))), ((int)(((byte)(169)))), ((int)(((byte)(164)))));
            this.uploadScanButton.FlatAppearance.BorderColor = System.Drawing.Color.Black;
            this.uploadScanButton.FlatAppearance.BorderSize = 0;
            this.uploadScanButton.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.uploadScanButton.Font = new System.Drawing.Font("Segoe UI Semibold", 15.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.uploadScanButton.Location = new System.Drawing.Point(0, 97);
            this.uploadScanButton.Name = "uploadScanButton";
            this.uploadScanButton.Size = new System.Drawing.Size(206, 70);
            this.uploadScanButton.TabIndex = 23;
            this.uploadScanButton.Text = "Upload Scan";
            this.uploadScanButton.UseVisualStyleBackColor = false;
            // 
            // panel1
            // 
            this.panel1.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(79)))), ((int)(((byte)(169)))), ((int)(((byte)(164)))));
            this.panel1.Location = new System.Drawing.Point(0, 1);
            this.panel1.Name = "panel1";
            this.panel1.Size = new System.Drawing.Size(922, 90);
            this.panel1.TabIndex = 20;
            // 
            // openFileDialog1
            // 
            this.openFileDialog1.FileName = "openFileDialog1";
            // 
            // uploadForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.BackColor = System.Drawing.SystemColors.ActiveCaptionText;
            this.ClientSize = new System.Drawing.Size(921, 551);
            this.Controls.Add(this.exportButton);
            this.Controls.Add(this.panel3);
            this.Controls.Add(this.sliceNoButton);
            this.Controls.Add(this.deleteButton);
            this.Controls.Add(this.uploadButton);
            this.Controls.Add(this.analyseButton);
            this.Controls.Add(this.panel2);
            this.Controls.Add(this.analysingButton);
            this.Controls.Add(this.viewResultsButton);
            this.Controls.Add(this.saveScanButton);
            this.Controls.Add(this.archiveButton);
            this.Controls.Add(this.logOutButton);
            this.Controls.Add(this.uploadScanButton);
            this.Controls.Add(this.panel1);
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.Name = "uploadForm";
            this.Text = "UPLOAD SCAN";
            this.panel3.ResumeLayout(false);
            this.panel3.PerformLayout();
            this.panel2.ResumeLayout(false);
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.Button exportButton;
        private System.Windows.Forms.Label cancerResultsLab;
        private System.Windows.Forms.Label results4Lab;
        private System.Windows.Forms.Label results3Label;
        private System.Windows.Forms.Label cancerSlidesLab;
        private System.Windows.Forms.Label results2Lab;
        private System.Windows.Forms.Panel panel3;
        private System.Windows.Forms.Label results1Label;
        private System.Windows.Forms.Button sliceNoButton;
        private System.Windows.Forms.Button deleteButton;
        private System.Windows.Forms.Button uploadButton;
        private System.Windows.Forms.HScrollBar scanScrollBar;
        private System.Windows.Forms.Button analyseButton;
        private System.Windows.Forms.Panel panel2;
        private System.Windows.Forms.Button analysingButton;
        private System.Windows.Forms.Button viewResultsButton;
        private System.Windows.Forms.Button saveScanButton;
        private System.Windows.Forms.Button archiveButton;
        private System.Windows.Forms.Button logOutButton;
        private System.Windows.Forms.Button uploadScanButton;
        private System.Windows.Forms.Panel panel1;
        private System.Windows.Forms.OpenFileDialog openFileDialog1;
    }
}