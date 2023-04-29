namespace frontend
{
    partial class LOGIN
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
            this.emailAddressInput = new System.Windows.Forms.TextBox();
            this.contextMenuStrip1 = new System.Windows.Forms.ContextMenuStrip(this.components);
            this.passwordInput = new System.Windows.Forms.TextBox();
            this.emailAddressLab = new System.Windows.Forms.Label();
            this.passwordLab = new System.Windows.Forms.Label();
            this.loginButton = new System.Windows.Forms.Button();
            this.reg1Lab = new System.Windows.Forms.Label();
            this.reg2Lab = new System.Windows.Forms.Label();
            this.registerButton = new System.Windows.Forms.Label();
            this.SuspendLayout();
            // 
            // emailAddressInput
            // 
            this.emailAddressInput.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(79)))), ((int)(((byte)(169)))), ((int)(((byte)(164)))));
            this.emailAddressInput.BorderStyle = System.Windows.Forms.BorderStyle.None;
            this.emailAddressInput.Font = new System.Drawing.Font("Segoe UI", 15.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.emailAddressInput.Location = new System.Drawing.Point(336, 184);
            this.emailAddressInput.Name = "emailAddressInput";
            this.emailAddressInput.RightToLeft = System.Windows.Forms.RightToLeft.No;
            this.emailAddressInput.Size = new System.Drawing.Size(227, 28);
            this.emailAddressInput.TabIndex = 2;
            this.emailAddressInput.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            this.emailAddressInput.TextChanged += new System.EventHandler(this.username_text_TextChanged);
            // 
            // contextMenuStrip1
            // 
            this.contextMenuStrip1.Name = "contextMenuStrip1";
            this.contextMenuStrip1.Size = new System.Drawing.Size(61, 4);
            // 
            // passwordInput
            // 
            this.passwordInput.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(79)))), ((int)(((byte)(169)))), ((int)(((byte)(164)))));
            this.passwordInput.BorderStyle = System.Windows.Forms.BorderStyle.None;
            this.passwordInput.Font = new System.Drawing.Font("Segoe UI", 15.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.passwordInput.Location = new System.Drawing.Point(336, 259);
            this.passwordInput.Name = "passwordInput";
            this.passwordInput.PasswordChar = '*';
            this.passwordInput.RightToLeft = System.Windows.Forms.RightToLeft.No;
            this.passwordInput.Size = new System.Drawing.Size(227, 28);
            this.passwordInput.TabIndex = 4;
            this.passwordInput.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            this.passwordInput.UseSystemPasswordChar = true;
            this.passwordInput.TextChanged += new System.EventHandler(this.password_text_TextChanged);
            // 
            // emailAddressLab
            // 
            this.emailAddressLab.AutoSize = true;
            this.emailAddressLab.BackColor = System.Drawing.Color.Black;
            this.emailAddressLab.Font = new System.Drawing.Font("Segoe UI", 15.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.emailAddressLab.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(28)))), ((int)(((byte)(116)))), ((int)(((byte)(106)))));
            this.emailAddressLab.Location = new System.Drawing.Point(331, 151);
            this.emailAddressLab.Name = "emailAddressLab";
            this.emailAddressLab.RightToLeft = System.Windows.Forms.RightToLeft.No;
            this.emailAddressLab.Size = new System.Drawing.Size(148, 30);
            this.emailAddressLab.TabIndex = 5;
            this.emailAddressLab.Text = "Email Address:";
            // 
            // passwordLab
            // 
            this.passwordLab.AutoSize = true;
            this.passwordLab.BackColor = System.Drawing.Color.Black;
            this.passwordLab.Font = new System.Drawing.Font("Segoe UI", 15.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.passwordLab.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(28)))), ((int)(((byte)(116)))), ((int)(((byte)(106)))));
            this.passwordLab.Location = new System.Drawing.Point(331, 226);
            this.passwordLab.Name = "passwordLab";
            this.passwordLab.RightToLeft = System.Windows.Forms.RightToLeft.No;
            this.passwordLab.Size = new System.Drawing.Size(104, 30);
            this.passwordLab.TabIndex = 6;
            this.passwordLab.Text = "Password:";
            // 
            // loginButton
            // 
            this.loginButton.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(28)))), ((int)(((byte)(116)))), ((int)(((byte)(106)))));
            this.loginButton.Font = new System.Drawing.Font("Segoe UI Semibold", 15.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.loginButton.ForeColor = System.Drawing.SystemColors.ActiveCaptionText;
            this.loginButton.Location = new System.Drawing.Point(376, 318);
            this.loginButton.Name = "loginButton";
            this.loginButton.RightToLeft = System.Windows.Forms.RightToLeft.No;
            this.loginButton.Size = new System.Drawing.Size(143, 61);
            this.loginButton.TabIndex = 7;
            this.loginButton.Text = "LOG IN";
            this.loginButton.UseVisualStyleBackColor = false;
            // 
            // reg1Lab
            // 
            this.reg1Lab.AutoSize = true;
            this.reg1Lab.BackColor = System.Drawing.Color.Black;
            this.reg1Lab.Font = new System.Drawing.Font("Segoe UI", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.reg1Lab.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(28)))), ((int)(((byte)(116)))), ((int)(((byte)(106)))));
            this.reg1Lab.Location = new System.Drawing.Point(363, 392);
            this.reg1Lab.Name = "reg1Lab";
            this.reg1Lab.RightToLeft = System.Windows.Forms.RightToLeft.No;
            this.reg1Lab.Size = new System.Drawing.Size(171, 21);
            this.reg1Lab.TabIndex = 9;
            this.reg1Lab.Text = "Don\'t have an account?";
            this.reg1Lab.Click += new System.EventHandler(this.click_to_reg_label_Click);
            // 
            // reg2Lab
            // 
            this.reg2Lab.AutoSize = true;
            this.reg2Lab.BackColor = System.Drawing.Color.Black;
            this.reg2Lab.Font = new System.Drawing.Font("Segoe UI", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.reg2Lab.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(28)))), ((int)(((byte)(116)))), ((int)(((byte)(106)))));
            this.reg2Lab.Location = new System.Drawing.Point(363, 413);
            this.reg2Lab.Name = "reg2Lab";
            this.reg2Lab.RightToLeft = System.Windows.Forms.RightToLeft.No;
            this.reg2Lab.Size = new System.Drawing.Size(100, 21);
            this.reg2Lab.TabIndex = 10;
            this.reg2Lab.Text = "Click here to ";
            this.reg2Lab.Click += new System.EventHandler(this.label1_Click);
            // 
            // registerButton
            // 
            this.registerButton.AutoSize = true;
            this.registerButton.BackColor = System.Drawing.Color.Black;
            this.registerButton.Font = new System.Drawing.Font("Segoe UI", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.registerButton.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(79)))), ((int)(((byte)(169)))), ((int)(((byte)(164)))));
            this.registerButton.Location = new System.Drawing.Point(456, 413);
            this.registerButton.Name = "registerButton";
            this.registerButton.RightToLeft = System.Windows.Forms.RightToLeft.No;
            this.registerButton.Size = new System.Drawing.Size(78, 21);
            this.registerButton.TabIndex = 11;
            this.registerButton.Text = "REGISTER";
            // 
            // LOGIN
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.BackColor = System.Drawing.Color.Black;
            this.ClientSize = new System.Drawing.Size(921, 551);
            this.Controls.Add(this.registerButton);
            this.Controls.Add(this.reg2Lab);
            this.Controls.Add(this.reg1Lab);
            this.Controls.Add(this.loginButton);
            this.Controls.Add(this.passwordLab);
            this.Controls.Add(this.emailAddressLab);
            this.Controls.Add(this.passwordInput);
            this.Controls.Add(this.emailAddressInput);
            this.Name = "LOGIN";
            this.Text = "LOG IN";
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.TextBox emailAddressInput;
        private System.Windows.Forms.ContextMenuStrip contextMenuStrip1;
        private System.Windows.Forms.TextBox passwordInput;
        private System.Windows.Forms.Label emailAddressLab;
        private System.Windows.Forms.Label passwordLab;
        private System.Windows.Forms.Button loginButton;
        private System.Windows.Forms.Label reg1Lab;
        private System.Windows.Forms.Label reg2Lab;
        private System.Windows.Forms.Label registerButton;
    }
}

