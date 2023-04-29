namespace frontend
{
    partial class splashScreenForm
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
            this.components = new System.ComponentModel.Container();
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(splashScreenForm));
            this.logImgList = new System.Windows.Forms.ImageList(this.components);
            this.listViewForImg = new System.Windows.Forms.ListView();
            this.timer1 = new System.Windows.Forms.Timer(this.components);
            this.SuspendLayout();
            // 
            // logImgList
            // 
            this.logImgList.ImageStream = ((System.Windows.Forms.ImageListStreamer)(resources.GetObject("logImgList.ImageStream")));
            this.logImgList.TransparentColor = System.Drawing.Color.Transparent;
            this.logImgList.Images.SetKeyName(0, "breastwise.png");
            // 
            // listViewForImg
            // 
            this.listViewForImg.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(79)))), ((int)(((byte)(169)))), ((int)(((byte)(164)))));
            this.listViewForImg.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("listViewForImg.BackgroundImage")));
            this.listViewForImg.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(79)))), ((int)(((byte)(169)))), ((int)(((byte)(164)))));
            this.listViewForImg.HideSelection = false;
            this.listViewForImg.Location = new System.Drawing.Point(218, 22);
            this.listViewForImg.Name = "listViewForImg";
            this.listViewForImg.Size = new System.Drawing.Size(544, 495);
            this.listViewForImg.StateImageList = this.logImgList;
            this.listViewForImg.TabIndex = 0;
            this.listViewForImg.UseCompatibleStateImageBehavior = false;
            // 
            // timer1
            // 
            this.timer1.Tick += new System.EventHandler(this.timer1_Tick);
            // 
            // splashScreenForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(79)))), ((int)(((byte)(169)))), ((int)(((byte)(164)))));
            this.ClientSize = new System.Drawing.Size(921, 551);
            this.Controls.Add(this.listViewForImg);
            this.ForeColor = System.Drawing.SystemColors.ControlLightLight;
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.Name = "splashScreenForm";
            this.Text = "SPLASH SCREEN";
            this.Load += new System.EventHandler(this.splashScreenForm_Load);
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.ImageList logImgList;
        private System.Windows.Forms.ListView listViewForImg;
        private System.Windows.Forms.Timer timer1;
    }
}