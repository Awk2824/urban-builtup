using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace BuildUpLandApp.Pages.Panduan
{
    /// <summary>
    /// Interaction logic for Panduan.xaml
    /// </summary>
    public partial class Panduan : Page
    {
        public Panduan()
        {
            InitializeComponent();
            this.Loaded += Panduan_Loaded;
        }

        private void Panduan_Loaded(object sender, RoutedEventArgs e)
        {
            GuideContentControl.Content = FindResource("PanduanSelamatDatang") as StackPanel;
        }

        private void Back_Click(object sender, RoutedEventArgs e)
        {
            if (NavigationService.CanGoBack)
            {
                NavigationService.GoBack();
            }
        }

        private void PanduanKlasifikasi_Click(object sender, RoutedEventArgs e)
        {
            GuideContentControl.Content = FindResource("PanduanKlasifikasi") as StackPanel;
        }

        private void PanduanTimeSeries_Click(object sender, RoutedEventArgs e)
        {
            GuideContentControl.Content = FindResource("PanduanPrediksi") as StackPanel;
        }
    }
}
