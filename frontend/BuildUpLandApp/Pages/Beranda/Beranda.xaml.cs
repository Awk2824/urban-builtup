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

namespace TerraBuild.Pages.Beranda
{
    /// <summary>
    /// Interaction logic for Beranda.xaml
    /// </summary>
    public partial class Beranda : Page
    {
        public Beranda()
        {
            InitializeComponent();
            this.Loaded += Page_Loaded;
            this.Unloaded += Page_Unloaded;
        }

        private void Page_Loaded(object sender, RoutedEventArgs e)
        {
            if (Mouse.Captured != null)
                Mouse.Capture(null);

            Keyboard.ClearFocus();
        }

        private void Page_Unloaded(object sender, RoutedEventArgs e)
        {
            if (Mouse.Captured != null)
                Mouse.Capture(null);

            Keyboard.ClearFocus();
        }

        #region Navigation
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
        #endregion
    }
}
