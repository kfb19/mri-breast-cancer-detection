namespace BreastWise
{
    public partial class SplashScreenForm : Form
    {
        private int tickCount = 0;

        public SplashScreenForm()
        {
            InitializeComponent();
        }

        private void SplashScreenForm_Load(object sender, EventArgs e)
        {
            timer1.Start(); // Start the timer.
        }

        private void timer1_Tick_2(object sender, EventArgs e)
        {
            tickCount++;
            if (tickCount >= 5)
            {
                timer1.Stop();
                UploadScan uploadScanForm = new UploadScan();
                uploadScanForm.Show();
                this.Hide();
            }
        }
    }
}