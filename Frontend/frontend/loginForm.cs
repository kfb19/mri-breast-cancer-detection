﻿using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Data.OleDb;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace frontend
{
    public partial class loginForm : Form
    {
        public loginForm()
        {
            InitializeComponent();
        }

        private void loginButton_Click(object sender, EventArgs e)
        {
            string input_email, input_password; // Declares all variables needed.
            clsDBConnector dbConnector = new clsDBConnector();
            OleDbDataReader dr;
            string sqlStr;

            bool loggedin = false; // To determine whether a user has logged in.

            input_email = emailAddressInput.Text; // Takes the user inputs.  
            input_password = passwordInput.Text;

            dbConnector.Connect();

            sqlStr = "SELECT email, [password] FROM tblUser"; //states the sql statement

            dr = dbConnector.DoSQL(sqlStr);//executes the SQL statement 

            while (dr.Read())
            {
                if (input_email == dr[0].ToString() && input_password == dr[1].ToString()) //checks to see if the login credentials are correct 
                {
                    loggedin = true;
                    int userType = Convert.ToInt32(dr[2].ToString());

                    uploadScanForm uploadScanForm = new uploadScanForm(); //input_email
                    uploadScanForm.Show();

                    this.Close();
                }
            }

            dr.Close();
            dbConnector.close();


            if (!loggedin) // Unable to log in error message.
            {
                MessageBox.Show("Email or password is incorrect. Please try again.");
                emailAddressInput.Clear();
                passwordInput.Clear();
            }

            dbConnector.close();
        }

        private void registerButton_Click(object sender, EventArgs e)
        {
            registerForm newRegisterForm = new registerForm(); // Opens a 'register' page.
            newRegisterForm.Show();

            this.Close();
        }
    }
}
    

