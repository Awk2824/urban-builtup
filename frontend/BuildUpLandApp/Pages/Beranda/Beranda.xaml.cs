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

namespace BuildUpLandApp.Pages.Beranda
{
    /// <summary>
    /// Interaction logic for Beranda.xaml
    /// </summary>
    public partial class Beranda : Page
    {
        public Beranda()
        {
            InitializeComponent();
        }

        //private void Pelatihan_Click(object sender, RoutedEventArgs e)
        //{
        //    NavigationService.Navigate(new Pages.Pelatihan.Pelatihan());
        //}

        private void Klasifikasi_Click(object sender, RoutedEventArgs e)
        {
            NavigationService.Navigate(new Pages.Klasifikasi.Klasifikasi());
        }

        private void TimeSeries_Click(object sender, RoutedEventArgs e)
        {
            NavigationService.Navigate(new Pages.TimeSeries.TimeSeries());
        }

        private void Panduan_Click(object sender, RoutedEventArgs e)
        {
            NavigationService.Navigate(new Pages.Panduan.Panduan());
        }

        private void Tentang_Click(object sender, RoutedEventArgs e)
        {
            NavigationService.Navigate(new Pages.Tentang.Tentang());
        }
    }
}
