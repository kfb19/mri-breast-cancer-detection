using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace frontend
{
    public partial class registerForm : Form
    {
        public registerForm()
        {
            InitializeComponent();
        }

        private void gotoLoginButton_Click(object sender, EventArgs e)
        {
            loginForm newLoginForm = new loginForm(); // Opens a 'login' page.
            newLoginForm.Show();

            this.Close();
        }
    }
    
}
