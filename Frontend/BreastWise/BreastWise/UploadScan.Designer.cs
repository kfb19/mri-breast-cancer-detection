namespace BreastWise
{
    partial class UploadScan
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
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(UploadScan));
            analyseBtn = new Button();
            scanPanel = new Panel();
            scanScrollBar = new HScrollBar();
            uploadBtn = new Button();
            exportBtn = new Button();
            deleteBtn = new Button();
            panel1 = new Panel();
            uploadMenu = new Panel();
            uploadLabel = new Label();
            analysingMenu = new Panel();
            analysingLab = new Label();
            resultsMenu = new Panel();
            resultsLab = new Label();
            countPan = new Panel();
            countLab = new Label();
            resultsPanel = new Panel();
            cancerStatusLab = new Label();
            statusLab4 = new Label();
            statusLab3 = new Label();
            cancerSliceLab = new Label();
            statusLab2 = new Label();
            statusLab1 = new Label();
            scanPanel.SuspendLayout();
            uploadMenu.SuspendLayout();
            analysingMenu.SuspendLayout();
            resultsMenu.SuspendLayout();
            countPan.SuspendLayout();
            resultsPanel.SuspendLayout();
            SuspendLayout();
            // 
            // analyseBtn
            // 
            analyseBtn.BackColor = Color.FromArgb(79, 169, 164);
            analyseBtn.FlatStyle = FlatStyle.Flat;
            analyseBtn.Font = new Font("Segoe UI Semibold", 12F, FontStyle.Bold, GraphicsUnit.Point);
            analyseBtn.ForeColor = Color.Black;
            analyseBtn.Location = new Point(813, 98);
            analyseBtn.Name = "analyseBtn";
            analyseBtn.Size = new Size(106, 47);
            analyseBtn.TabIndex = 0;
            analyseBtn.Text = "Analyse";
            analyseBtn.UseVisualStyleBackColor = false;
            analyseBtn.Click += analyseBtn_Click;
            // 
            // scanPanel
            // 
            scanPanel.BackColor = Color.FromArgb(117, 117, 117);
            scanPanel.Controls.Add(scanScrollBar);
            scanPanel.Location = new Point(222, 98);
            scanPanel.Name = "scanPanel";
            scanPanel.Size = new Size(585, 450);
            scanPanel.TabIndex = 1;
            // 
            // scanScrollBar
            // 
            scanScrollBar.Location = new Point(0, 419);
            scanScrollBar.Name = "scanScrollBar";
            scanScrollBar.Size = new Size(585, 31);
            scanScrollBar.TabIndex = 2;
            scanScrollBar.Scroll += scanScrollBar_Scroll;
            // 
            // uploadBtn
            // 
            uploadBtn.BackColor = Color.FromArgb(79, 169, 164);
            uploadBtn.FlatStyle = FlatStyle.Flat;
            uploadBtn.Font = new Font("Segoe UI Semibold", 12F, FontStyle.Bold, GraphicsUnit.Point);
            uploadBtn.ForeColor = Color.Black;
            uploadBtn.Location = new Point(813, 395);
            uploadBtn.Name = "uploadBtn";
            uploadBtn.Size = new Size(106, 47);
            uploadBtn.TabIndex = 2;
            uploadBtn.Text = "Upload";
            uploadBtn.UseVisualStyleBackColor = false;
            uploadBtn.Click += uploadBtn_Click;
            // 
            // exportBtn
            // 
            exportBtn.BackColor = Color.FromArgb(28, 116, 106);
            exportBtn.FlatStyle = FlatStyle.Flat;
            exportBtn.Font = new Font("Segoe UI Semibold", 12F, FontStyle.Bold, GraphicsUnit.Point);
            exportBtn.ForeColor = Color.Black;
            exportBtn.Location = new Point(813, 342);
            exportBtn.Name = "exportBtn";
            exportBtn.Size = new Size(106, 47);
            exportBtn.TabIndex = 3;
            exportBtn.Text = "Export";
            exportBtn.UseVisualStyleBackColor = false;
            exportBtn.Click += exportBtn_Click;
            // 
            // deleteBtn
            // 
            deleteBtn.BackColor = Color.FromArgb(79, 169, 164);
            deleteBtn.FlatStyle = FlatStyle.Flat;
            deleteBtn.Font = new Font("Segoe UI Semibold", 12F, FontStyle.Bold, GraphicsUnit.Point);
            deleteBtn.ForeColor = Color.Black;
            deleteBtn.Location = new Point(813, 448);
            deleteBtn.Name = "deleteBtn";
            deleteBtn.Size = new Size(106, 47);
            deleteBtn.TabIndex = 4;
            deleteBtn.Text = "Delete";
            deleteBtn.UseVisualStyleBackColor = false;
            deleteBtn.Click += deleteBtn_Click;
            // 
            // panel1
            // 
            panel1.BackColor = Color.FromArgb(79, 169, 164);
            panel1.Location = new Point(0, 0);
            panel1.Name = "panel1";
            panel1.Size = new Size(931, 92);
            panel1.TabIndex = 5;
            // 
            // uploadMenu
            // 
            uploadMenu.BackColor = Color.FromArgb(28, 116, 106);
            uploadMenu.Controls.Add(uploadLabel);
            uploadMenu.Font = new Font("Segoe UI Semibold", 15.75F, FontStyle.Bold, GraphicsUnit.Point);
            uploadMenu.Location = new Point(0, 98);
            uploadMenu.Name = "uploadMenu";
            uploadMenu.Size = new Size(216, 70);
            uploadMenu.TabIndex = 9;
            // 
            // uploadLabel
            // 
            uploadLabel.AutoSize = true;
            uploadLabel.Location = new Point(42, 20);
            uploadLabel.Name = "uploadLabel";
            uploadLabel.Size = new Size(133, 30);
            uploadLabel.TabIndex = 0;
            uploadLabel.Text = "Upload Scan";
            // 
            // analysingMenu
            // 
            analysingMenu.BackColor = Color.FromArgb(79, 169, 164);
            analysingMenu.Controls.Add(analysingLab);
            analysingMenu.Font = new Font("Segoe UI Semibold", 15.75F, FontStyle.Bold, GraphicsUnit.Point);
            analysingMenu.Location = new Point(0, 174);
            analysingMenu.Name = "analysingMenu";
            analysingMenu.Size = new Size(216, 70);
            analysingMenu.TabIndex = 10;
            // 
            // analysingLab
            // 
            analysingLab.AutoSize = true;
            analysingLab.Location = new Point(56, 20);
            analysingLab.Name = "analysingLab";
            analysingLab.Size = new Size(105, 30);
            analysingLab.TabIndex = 0;
            analysingLab.Text = "Analysing";
            // 
            // resultsMenu
            // 
            resultsMenu.BackColor = Color.FromArgb(79, 169, 164);
            resultsMenu.Controls.Add(resultsLab);
            resultsMenu.Font = new Font("Segoe UI Semibold", 15.75F, FontStyle.Bold, GraphicsUnit.Point);
            resultsMenu.Location = new Point(0, 250);
            resultsMenu.Name = "resultsMenu";
            resultsMenu.Size = new Size(216, 70);
            resultsMenu.TabIndex = 11;
            // 
            // resultsLab
            // 
            resultsLab.AutoSize = true;
            resultsLab.Location = new Point(42, 20);
            resultsLab.Name = "resultsLab";
            resultsLab.Size = new Size(130, 30);
            resultsLab.TabIndex = 0;
            resultsLab.Text = "View Results";
            // 
            // countPan
            // 
            countPan.BackColor = Color.FromArgb(28, 116, 106);
            countPan.Controls.Add(countLab);
            countPan.Font = new Font("Segoe UI Semibold", 15.75F, FontStyle.Bold, GraphicsUnit.Point);
            countPan.Location = new Point(813, 501);
            countPan.Name = "countPan";
            countPan.Size = new Size(106, 47);
            countPan.TabIndex = 10;
            // 
            // countLab
            // 
            countLab.AutoSize = true;
            countLab.Font = new Font("Segoe UI Semibold", 12F, FontStyle.Bold, GraphicsUnit.Point);
            countLab.Location = new Point(30, 13);
            countLab.Name = "countLab";
            countLab.Size = new Size(35, 21);
            countLab.TabIndex = 0;
            countLab.Text = "0/0";
            // 
            // resultsPanel
            // 
            resultsPanel.BackColor = Color.FromArgb(28, 116, 106);
            resultsPanel.Controls.Add(cancerStatusLab);
            resultsPanel.Controls.Add(statusLab4);
            resultsPanel.Controls.Add(statusLab3);
            resultsPanel.Controls.Add(cancerSliceLab);
            resultsPanel.Controls.Add(statusLab2);
            resultsPanel.Controls.Add(statusLab1);
            resultsPanel.Font = new Font("Segoe UI Semibold", 15.75F, FontStyle.Bold, GraphicsUnit.Point);
            resultsPanel.Location = new Point(0, 326);
            resultsPanel.Name = "resultsPanel";
            resultsPanel.Size = new Size(216, 218);
            resultsPanel.TabIndex = 11;
            // 
            // cancerStatusLab
            // 
            cancerStatusLab.AutoSize = true;
            cancerStatusLab.Font = new Font("Segoe UI", 15.75F, FontStyle.Bold, GraphicsUnit.Point);
            cancerStatusLab.Location = new Point(47, 150);
            cancerStatusLab.Name = "cancerStatusLab";
            cancerStatusLab.Size = new Size(114, 30);
            cancerStatusLab.TabIndex = 5;
            cancerStatusLab.Text = "NEGATIVE";
            // 
            // statusLab4
            // 
            statusLab4.AutoSize = true;
            statusLab4.Font = new Font("Segoe UI Semibold", 9.75F, FontStyle.Bold, GraphicsUnit.Point);
            statusLab4.Location = new Point(27, 98);
            statusLab4.Name = "statusLab4";
            statusLab4.Size = new Size(0, 17);
            statusLab4.TabIndex = 4;
            // 
            // statusLab3
            // 
            statusLab3.AutoSize = true;
            statusLab3.Font = new Font("Segoe UI Semibold", 14.25F, FontStyle.Bold, GraphicsUnit.Point);
            statusLab3.Location = new Point(21, 115);
            statusLab3.Name = "statusLab3";
            statusLab3.Size = new Size(180, 25);
            statusLab3.TabIndex = 3;
            statusLab3.Text = "Current slice status:";
            // 
            // cancerSliceLab
            // 
            cancerSliceLab.AutoSize = true;
            cancerSliceLab.Font = new Font("Segoe UI", 15.75F, FontStyle.Bold, GraphicsUnit.Point);
            cancerSliceLab.Location = new Point(68, 69);
            cancerSliceLab.Name = "cancerSliceLab";
            cancerSliceLab.Size = new Size(97, 30);
            cancerSliceLab.TabIndex = 2;
            cancerSliceLab.Text = "0 SLICES";
            // 
            // statusLab2
            // 
            statusLab2.AutoSize = true;
            statusLab2.Font = new Font("Segoe UI Semibold", 14.25F, FontStyle.Bold, GraphicsUnit.Point);
            statusLab2.Location = new Point(21, 44);
            statusLab2.Name = "statusLab2";
            statusLab2.Size = new Size(177, 25);
            statusLab2.TabIndex = 1;
            statusLab2.Text = "Cancer detected in:";
            // 
            // statusLab1
            // 
            statusLab1.AutoSize = true;
            statusLab1.Font = new Font("Segoe UI Semibold", 9.75F, FontStyle.Bold, GraphicsUnit.Point);
            statusLab1.Location = new Point(42, 11);
            statusLab1.Name = "statusLab1";
            statusLab1.Size = new Size(0, 17);
            statusLab1.TabIndex = 0;
            // 
            // UploadScan
            // 
            AutoScaleDimensions = new SizeF(7F, 15F);
            AutoScaleMode = AutoScaleMode.Font;
            BackColor = SystemColors.ActiveCaptionText;
            ClientSize = new Size(921, 551);
            Controls.Add(resultsPanel);
            Controls.Add(countPan);
            Controls.Add(resultsMenu);
            Controls.Add(analysingMenu);
            Controls.Add(uploadMenu);
            Controls.Add(panel1);
            Controls.Add(deleteBtn);
            Controls.Add(exportBtn);
            Controls.Add(uploadBtn);
            Controls.Add(scanPanel);
            Controls.Add(analyseBtn);
            Icon = (Icon)resources.GetObject("$this.Icon");
            Name = "UploadScan";
            Text = "BREASTWISE";
            scanPanel.ResumeLayout(false);
            uploadMenu.ResumeLayout(false);
            uploadMenu.PerformLayout();
            analysingMenu.ResumeLayout(false);
            analysingMenu.PerformLayout();
            resultsMenu.ResumeLayout(false);
            resultsMenu.PerformLayout();
            countPan.ResumeLayout(false);
            countPan.PerformLayout();
            resultsPanel.ResumeLayout(false);
            resultsPanel.PerformLayout();
            ResumeLayout(false);
        }

        #endregion

        private Button analyseBtn;
        private Panel scanPanel;
        private HScrollBar scanScrollBar;
        private Button uploadBtn;
        private Button exportBtn;
        private Button deleteBtn;
        private Panel panel1;
        private Panel uploadMenu;
        private Label uploadLabel;
        private Panel analysingMenu;
        private Label analysingLab;
        private Panel resultsMenu;
        private Label resultsLab;
        private Panel countPan;
        private Label countLab;
        private Panel resultsPanel;
        private Label statusLab4;
        private Label statusLab3;
        private Label cancerSliceLab;
        private Label statusLab2;
        private Label statusLab1;
        private Label cancerStatusLab;
    }
}