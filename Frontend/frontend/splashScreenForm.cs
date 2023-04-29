using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace frontend
{
    public partial class splashScreenForm : Form
    {
        public splashScreenForm()
        {
            InitializeComponent();
        }

        private void splashScreenForm_Load(object sender, EventArgs e)
        {
            Thread.Sleep(5000); // Wait for 5 seconds.
            timer1.Start(); // Start the timer.
        }

        private void timer1_Tick(object sender, EventArgs e)
        {
            timer1.Stop(); // Stop the timer to prevent it from firing again.
            loginForm newLoginForm = new loginForm(); // Create a new instance of the LoginForm.
            newLoginForm.Show(); // Show the LoginForm.
            this.Hide(); // Hide the current form.
        }
    }
}
