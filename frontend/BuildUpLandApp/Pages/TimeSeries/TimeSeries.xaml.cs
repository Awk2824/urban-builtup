using BuildUpLandApp.Constant;
using Npgsql;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text.Json;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;

namespace BuildUpLandApp.Pages.TimeSeries
{
    /// <summary>
    /// Interaction logic for TimeSeries.xaml
    /// </summary>
    public partial class TimeSeries : Page
    {
        #region Initialization
        private string ConnectionString => $"Host={Constants.DB_HOST};Username={Constants.DB_USER};Password={Constants.DB_PASSWORD};Port={Constants.DB_PORT};Database={Constants.DB_NAME}";
        private Dictionary<string, int> _kecamatanNameToIdMap = new Dictionary<string, int>();
        private bool _isUpdatingDropdowns = false;
        #endregion

        #region Page Load
        public TimeSeries()
        {
            InitializeComponent();
            this.Loaded += Page_Loaded;
            InitializeDefaultTexts();
        }

        private void InitializeDefaultTexts()
        {
            MaeText.Text = "MAE\t\t: -";
            MseText.Text = "MSE\t\t: -";
            RmseText.Text = "RMSE\t\t: -";
            R2Text.Text = "R-Squared\t: -";

            KecamatanOutputText.Text = "-";
            TahunPrediksiText.Text = "-";
            LuasTerbangunOutputText.Text = "-";
            PersentaseOutputText.Text = "-";
        }

        private async void Page_Loaded(object sender, RoutedEventArgs e)
        {
            await LoadInitialDataAsync();
            DisablePlaceholderItem(KecamatanPrediksi);
            DisablePlaceholderItem(TahunPrediksi);

            await LoadLatestMetricsAsync();
        }

        private async Task LoadInitialDataAsync()
        {
            await GetKecamatan();
            PopulateTahunPrediksi();
        }

        private async Task LoadLatestMetricsAsync()
        {
            string lstmJson = await GetLatestMetricsJsonAsync("LSTM");
            if (!string.IsNullOrEmpty(lstmJson))
            {
                UpdateLstmMetricsUI(lstmJson);
            }
        }

        private async Task<string> GetLatestMetricsJsonAsync(string modelType)
        {
            try
            {
                await using var conn = new NpgsqlConnection(ConnectionString);
                await conn.OpenAsync();
                var sql = "SELECT metrics FROM model WHERE model_type = @type ORDER BY train_date DESC LIMIT 1;";
                await using var cmd = new NpgsqlCommand(sql, conn);
                cmd.Parameters.AddWithValue("type", modelType);
                var result = await cmd.ExecuteScalarAsync();

                return result?.ToString();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Gagal mengambil metrik {modelType}: {ex.Message}");
                return null;
            }
        }

        private void UpdateLstmMetricsUI(string json)
        {
            try
            {
                var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
                LstmMetrics metrics = JsonSerializer.Deserialize<LstmMetrics>(json, options);

                MaeText.Text = $"MAE\t\t: {metrics.MAE:F4}";
                MseText.Text = $"MSE\t\t: {metrics.MSE:F4}";
                RmseText.Text = $"RMSE\t\t: {metrics.RMSE:F4}";
                MapeText.Text = $"MAPE\t\t: {metrics.MAPE * 100:F2}%";
                R2Text.Text = $"R-Squared\t: {metrics.R2:F4}";

                DisplayGraph(R2GraphContainer, metrics.R2GraphPath);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Gagal parse JSON LSTM: {ex.Message}");
                MaeText.Text = "MAE\t\t: -";
                MseText.Text = "MSE\t\t: -";
                RmseText.Text = "RMSE\t\t: -";
                MapeText.Text = "MAPE\t\t: -";
                R2Text.Text = "R-Squared\t: -";
                DisplayGraph(R2GraphContainer, null);
            }
        }

        private async Task GetKecamatan()
        {
            _kecamatanNameToIdMap.Clear();
            var kecamatanList = new List<string>();
            try
            {
                await using var conn = new NpgsqlConnection(ConnectionString);
                await conn.OpenAsync();
                var sql = "SELECT id_wilayah, nama_kec FROM wilayah_administrasi ORDER BY nama_kec;";
                await using var cmd = new NpgsqlCommand(sql, conn);
                await using var reader = await cmd.ExecuteReaderAsync();
                while (await reader.ReadAsync())
                {
                    int id = reader.GetInt32(0);
                    string name = reader.GetString(1);
                    string displayName = name.Replace("_", " ");
                    kecamatanList.Add(displayName);
                    _kecamatanNameToIdMap[displayName] = id;
                }
            }
            catch (Exception ex) 
            { 
                HandleDbError("mengambil data kecamatan", ex); 
                return; 
            }

            PopulateComboBox(KecamatanPrediksi, kecamatanList, "Pilih kecamatan");
        }

        private void PopulateTahunPrediksi()
        {
            int currentYear = DateTime.Now.Year;
            for (int year = currentYear; year <= 2030; year++)
            {
                TahunPrediksi.Items.Add(year);
            }
        }

        private void PopulateComboBox(ComboBox comboBox, List<string> items, string placeholderText)
        {
            if (comboBox == null || items == null) return;
            string currentSelection = (comboBox.SelectedIndex > 0 && comboBox.SelectedItem != null) ? comboBox.SelectedItem.ToString() : null;
            comboBox.Items.Clear();
            comboBox.Items.Add(new ComboBoxItem 
            { 
                Content = placeholderText, 
                IsEnabled = false 
            });
            foreach (var item in items) 
            { 
                comboBox.Items.Add(item); 
            }
            var itemToSelect = comboBox.Items.Cast<object>().FirstOrDefault(i => i is string && i.ToString() == currentSelection);
            if (itemToSelect != null) comboBox.SelectedItem = itemToSelect; else comboBox.SelectedIndex = 0;
        }

        private void DisablePlaceholderItem(ComboBox comboBox)
        {
            if (comboBox.Items.Count > 0 && comboBox.Items[0] is ComboBoxItem placeholder) 
            { 
                placeholder.IsEnabled = false; 
            }
        }

        private void HandleDbError(string action, Exception ex) 
        { 
            MessageBox.Show($"Gagal {action}.\n\nDetail: {ex.Message}", "Error Database", MessageBoxButton.OK, MessageBoxImage.Error); 
        }
        #endregion

        #region Button Click
        private void Back_Click(object sender, RoutedEventArgs e)
        {
            NavigationService nav = NavigationService.GetNavigationService(this);
            if (nav != null && nav.CanGoBack)
                nav.GoBack();
        }

        private async void LatihModelLSTM_Click(object sender, RoutedEventArgs e)
        {
            if (!int.TryParse(SeqLengthTextBox.Text, out int seqLength) || seqLength <= 0 ||
                !int.TryParse(EpochsTextBox.Text, out int epochs) || epochs <= 0 ||
                !double.TryParse(LearningRateTextBox.Text, NumberStyles.Any, CultureInfo.InvariantCulture, out double learningRate) || learningRate <= 0 ||
                !int.TryParse(HiddenUnitsTextBox.Text, out int hiddenUnits) || hiddenUnits <= 0 ||
                !int.TryParse(BatchSizeTextBox.Text, out int batchSize) || batchSize <= 0 ||
                !double.TryParse(DropoutRateTextBox.Text, NumberStyles.Any, CultureInfo.InvariantCulture, out double dropoutRate) || dropoutRate < 0.0 || dropoutRate >= 1.0 ||
                OptimizerComboBox.SelectedIndex < 0)
            {
                MessageBox.Show("Pastikan semua hyperparameter diisi dengan angka positif yang valid.", "Input Tidak Valid", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            if (!await CheckTimeSeriesDataExistsAsync())
            {
                MessageBox.Show("Belum ada data luas lahan terbangun di database.", "Data Tidak Ada", MessageBoxButton.OK, MessageBoxImage.Warning);
            }

            string selectedOptimizer = (OptimizerComboBox.SelectedItem as ComboBoxItem)?.Content.ToString() ?? "Adam";

            string metricsFilePath = Path.GetTempFileName();
            string r2GraphFinalPath = null;
            SetUIEnabled(false);
            bool success = await RunPythonScript(
                Constants.PYTHON_TRAINING_LSTM,
                metricsFilePath: metricsFilePath,
                args: new List<string> {
                    seqLength.ToString(), 
                    epochs.ToString(),
                    learningRate.ToString(CultureInfo.InvariantCulture), 
                    hiddenUnits.ToString(),
                    batchSize.ToString(), 
                    dropoutRate.ToString(CultureInfo.InvariantCulture),
                    selectedOptimizer
                }
            );
            SetUIEnabled(true);

            if (success && File.Exists(metricsFilePath))
            {
                try
                {
                    string json = await File.ReadAllTextAsync(metricsFilePath);
                    if (!string.IsNullOrWhiteSpace(json))
                    {
                        var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
                        LstmMetrics metrics = JsonSerializer.Deserialize<LstmMetrics>(json, options);
                        MaeText.Text = $"MAE\t\t: {metrics.MAE:F4}";
                        MseText.Text = $"MSE\t\t: {metrics.MSE:F4}";
                        RmseText.Text = $"RMSE\t\t: {metrics.RMSE:F4}";
                        R2Text.Text = $"R-Squared\t: {metrics.R2:F4}";

                        if (!string.IsNullOrEmpty(metrics.R2GraphPath))
                        {
                            DisplayGraph(R2GraphContainer, metrics.R2GraphPath);
                            r2GraphFinalPath = metrics.R2GraphPath;
                        }
                        else
                        {
                            DisplayGraph(R2GraphContainer, null);
                        }
                        MessageBox.Show("Pelatihan LSTM selesai.", "Sukses", MessageBoxButton.OK, MessageBoxImage.Information);
                    }
                }
                catch (Exception ex) 
                {
                    HandleFileError("membaca file metrik LSTM", ex); 
                }
            }

            if (File.Exists(metricsFilePath)) 
                TryDeleteFile(metricsFilePath);
            if (r2GraphFinalPath != null && r2GraphFinalPath.Contains(Path.GetTempPath())) 
                TryDeleteFile(r2GraphFinalPath);
        }

        private async void Prediksi_Click(object sender, RoutedEventArgs e)
        {
            if (KecamatanPrediksi.SelectedIndex <= 0) 
            { 
                MessageBox.Show("Silakan pilih kecamatan.", "Input Kurang", MessageBoxButton.OK, MessageBoxImage.Warning); 
                return; 
            }
            if (TahunPrediksi.SelectedIndex <= 0) 
            { 
                MessageBox.Show("Silakan pilih tahun target.", "Input Kurang", MessageBoxButton.OK, MessageBoxImage.Warning); 
                return; 
            }

            string selectedKecamatanDisplay = KecamatanPrediksi.SelectedItem.ToString();
            int targetYear = (int)TahunPrediksi.SelectedItem;

            if (!_kecamatanNameToIdMap.TryGetValue(selectedKecamatanDisplay, out int idWilayah))
            {
                MessageBox.Show($"ID Wilayah tidak ditemukan untuk kecamatan '{selectedKecamatanDisplay}'.", "Error Data", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }

            if (!await CheckModelExistsAsync("LSTM"))
            {
                MessageBox.Show("Model LSTM belum dilatih. Silakan latih model terlebih dahulu di tab 'Pelatihan Model'.", "Model Tidak Ditemukan", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            string resultFilePath = Path.GetTempFileName();
            string predictionGraphFinalPath = null;
            SetUIEnabled(false);
            bool success = await RunPythonScript(
                Constants.PYTHON_PREDICTION_LSTM,
                resultFilePath: resultFilePath,
                args: new List<string> 
                { 
                    targetYear.ToString(), 
                    idWilayah.ToString() 
                }
            );
            SetUIEnabled(true);

            if (success && File.Exists(resultFilePath))
            {
                try
                {
                    string json = await File.ReadAllTextAsync(resultFilePath);
                    if (!string.IsNullOrWhiteSpace(json))
                    {
                        var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
                        LstmPredictionResult result = JsonSerializer.Deserialize<LstmPredictionResult>(json, options);

                        if (result.Predictions != null && result.Predictions.TryGetValue(targetYear, out double predictedValue))
                        {
                            KecamatanOutputText.Text = selectedKecamatanDisplay;
                            TahunPrediksiText.Text = targetYear.ToString();
                            LuasTerbangunOutputText.Text = $"{predictedValue:F3}";
                            PersentaseOutputText.Text = $"{result.PredictedPercentage:F2}";

                            if (!string.IsNullOrEmpty(result.PredictionGraphPath) && File.Exists(result.PredictionGraphPath))
                            {
                                DisplayGraph(PredictionGraphContainer, result.PredictionGraphPath);
                                predictionGraphFinalPath = result.PredictionGraphPath;
                            }
                            else
                            {
                                DisplayGraph(PredictionGraphContainer, null);
                            }

                            MessageBox.Show("Prediksi LSTM selesai.", "Sukses", MessageBoxButton.OK, MessageBoxImage.Information);
                        }
                        else
                        {
                            MessageBox.Show($"Hasil prediksi untuk tahun {targetYear} tidak ditemukan dalam output backend.", "Data Tidak Ditemukan", MessageBoxButton.OK, MessageBoxImage.Warning);
                        }
                    }
                }
                catch (Exception ex) 
                { 
                    HandleFileError("memproses hasil prediksi LSTM", ex); 
                }
            }

            if (File.Exists(resultFilePath)) 
                TryDeleteFile(resultFilePath);
            if (predictionGraphFinalPath != null && predictionGraphFinalPath.Contains(Path.GetTempPath())) 
                TryDeleteFile(predictionGraphFinalPath);
        }

        private async Task<bool> RunPythonScript(string scriptConstant, string resultFilePath = null, string metricsFilePath = null, List<string> args = null)
        {
            try
            {
                string executablePath = AppDomain.CurrentDomain.BaseDirectory;
                string solutionPath = FindSolutionRoot(executablePath);
                if (solutionPath == null) return false;

                string workingDirectory = Path.Combine(solutionPath, Constants.BACKEND_ROOT_FOLDER);
                string modulePath = scriptConstant.Replace(".py", "").Replace(Path.DirectorySeparatorChar, '.').Replace(Path.AltDirectorySeparatorChar, '.');

                // Urutan: -m module [metrics/result file path] [argumen lain...]
                string arguments = $"-m {modulePath}";

                if (!string.IsNullOrEmpty(metricsFilePath)) 
                    arguments += $" \"{metricsFilePath}\""; // Untuk Training

                if (!string.IsNullOrEmpty(resultFilePath)) 
                    arguments += $" \"{resultFilePath}\"";  // Untuk Prediksi

                if (args != null && args.Count > 0)
                {
                    arguments += " " + string.Join(" ", args.Select(a => $"\"{a}\""));
                }

                ProcessStartInfo psi = new ProcessStartInfo
                {
                    FileName = "cmd.exe",
                    Arguments = $"/C {Constants.PYTHON} {arguments}",
                    WorkingDirectory = workingDirectory,
                    CreateNoWindow = false,
                    UseShellExecute = false
                };

                using (Process process = Process.Start(psi))
                {
                    await process.WaitForExitAsync();
                    return true;
                }
            }
            catch (Exception ex) 
            { 
                HandleScriptError(scriptConstant, ex); 
                return false; 
            }
        }
        #endregion

        #region Helper
        private void SetUIEnabled(bool isEnabled)
        {
            // Tab Pelatihan
            LatihModelLSTMButton.IsEnabled = isEnabled;
            SeqLengthTextBox.IsEnabled = isEnabled;
            EpochsTextBox.IsEnabled = isEnabled;
            LearningRateTextBox.IsEnabled = isEnabled;
            HiddenUnitsTextBox.IsEnabled = isEnabled;
            BatchSizeTextBox.IsEnabled = isEnabled;

            // Tab Prediksi
            PrediksiButton.IsEnabled = isEnabled;
            KecamatanPrediksi.IsEnabled = isEnabled;
            TahunPrediksi.IsEnabled = isEnabled;

            this.Cursor = isEnabled ? Cursors.Arrow : Cursors.Wait;
        }

        private async Task<bool> CheckModelExistsAsync(string modelType)
        {
            try
            {
                await using var conn = new NpgsqlConnection(ConnectionString); await conn.OpenAsync();
                var sql = "SELECT COUNT(*) FROM model WHERE model_type = @type;";
                await using var cmd = new NpgsqlCommand(sql, conn);
                cmd.Parameters.AddWithValue("type", modelType);
                return (long)await cmd.ExecuteScalarAsync() > 0;
            }
            catch (Exception ex) 
            { 
                HandleDbError($"mengecek model {modelType}", ex); 
                return false; 
            }
        }

        private async Task<bool> CheckTimeSeriesDataExistsAsync()
        {
            try
            {
                await using var conn = new NpgsqlConnection(ConnectionString); await conn.OpenAsync();
                var sql = "SELECT COUNT(*) FROM luas_terbangun;";
                await using var cmd = new NpgsqlCommand(sql, conn);
                return (long)await cmd.ExecuteScalarAsync() > 0;
            }
            catch (Exception ex) 
            { 
                HandleDbError("mengecek data luas terbangun", ex); 
                return false; 
            }
        }

        private async Task<int?> GetIdWilayahAsync(string kecamatanName)
        {
            try
            {
                await using var conn = new NpgsqlConnection(ConnectionString); await conn.OpenAsync();
                var sql = "SELECT id_wilayah FROM wilayah_administrasi WHERE nama_kec = @nama_kec LIMIT 1;";
                await using var cmd = new NpgsqlCommand(sql, conn);
                cmd.Parameters.AddWithValue("nama_kec", kecamatanName.Replace(" ", "_"));
                var result = await cmd.ExecuteScalarAsync();
                if (result != null && result != DBNull.Value) return Convert.ToInt32(result);
                else return null;
            }
            catch (Exception ex) 
            { 
                HandleDbError($"mencari ID untuk {kecamatanName}", ex); 
                return null; 
            }
        }

        private void DisplayGraph(Border container, string graphData)
        {
            if (container == null) return;
            container.Background = (SolidColorBrush)(new BrushConverter().ConvertFrom("#E2E8F0"));
            container.Child = new TextBlock 
            { 
                Text = "Grafik akan tampil di sini", 
                HorizontalAlignment = HorizontalAlignment.Center, 
                VerticalAlignment = VerticalAlignment.Center, 
                Foreground = (SolidColorBrush)(new BrushConverter().ConvertFrom("#718096")) 
            };


            if (string.IsNullOrEmpty(graphData))
            {
                container.Child = new TextBlock
                {
                    Text = "Grafik R-Squared akan tampil di sini",
                    HorizontalAlignment = HorizontalAlignment.Center,
                    VerticalAlignment = VerticalAlignment.Center
                };
                return;
            }

            try
            {
                byte[] imageBytes = Convert.FromBase64String(graphData);
                using (var memStream = new MemoryStream(imageBytes))
                {
                    BitmapImage bitmap = new BitmapImage();
                    bitmap.BeginInit();
                    bitmap.CacheOption = BitmapCacheOption.OnLoad;
                    bitmap.StreamSource = memStream;
                    bitmap.EndInit();
                    bitmap.Freeze();

                    Image imageControl = new Image { Source = bitmap, Stretch = Stretch.Uniform };
                    container.Background = Brushes.Transparent;
                    container.Child = imageControl;
                }
            }
            catch (FormatException)
            {
                if (File.Exists(graphData))
                {
                    try
                    {
                        BitmapImage bitmap = new BitmapImage();
                        bitmap.BeginInit();
                        bitmap.UriSource = new Uri(graphData, UriKind.Absolute);
                        bitmap.CacheOption = BitmapCacheOption.OnLoad;
                        bitmap.EndInit();
                        bitmap.Freeze();

                        Image imageControl = new Image { Source = bitmap, Stretch = Stretch.Uniform };
                        container.Background = Brushes.Transparent;
                        container.Child = imageControl;
                    }
                    catch (Exception)
                    {
                        container.Child = new TextBlock
                        {
                            Text = $"Gagal memuat file gambar",
                            Foreground = Brushes.Red,
                            TextWrapping = TextWrapping.Wrap,
                            HorizontalAlignment = HorizontalAlignment.Center,
                            VerticalAlignment = VerticalAlignment.Center
                        };
                    }
                }
            }
            catch (Exception ex)
            {
                container.Child = new TextBlock
                {
                    Text = $"Gagal memuat grafik:\n{ex.Message}",
                    Foreground = Brushes.Red,
                    TextWrapping = TextWrapping.Wrap,
                    HorizontalAlignment = HorizontalAlignment.Center,
                    VerticalAlignment = VerticalAlignment.Center
                };
            }
        }

        private void HandleScriptError(string scriptName, Exception ex)
        {
            MessageBox.Show($"Gagal menjalankan skrip {Path.GetFileName(scriptName)}: {ex.Message}", "Error Eksekusi", MessageBoxButton.OK, MessageBoxImage.Error);
        }

        private void HandleFileError(string action, Exception ex)
        {
            MessageBox.Show($"Gagal {action}: {ex.Message}", "Error File", MessageBoxButton.OK, MessageBoxImage.Error);
        }

        private void TryDeleteFile(string filePath)
        {
            try
            {
                if (File.Exists(filePath)) File.Delete(filePath);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Gagal hapus file sementara {filePath}: {ex.Message}");
            }
        }

        private string FindSolutionRoot(string startPath)
        {
            DirectoryInfo currentDir = new DirectoryInfo(startPath);
            while (currentDir != null && currentDir.Parent != null)
            {
                if (Directory.Exists(Path.Combine(currentDir.FullName, Constants.BACKEND_ROOT_FOLDER)))
                {
                    return currentDir.FullName;
                }
                currentDir = currentDir.Parent;
            }
            return null;
        }

        private string FindDataRoot(string filePath)
        {
            if (string.IsNullOrWhiteSpace(filePath))
            {
                return null;
            }

            try
            {
                DirectoryInfo currentDir = new DirectoryInfo(Path.GetDirectoryName(filePath));
                while (currentDir != null)
                {
                    if (Directory.Exists(Path.Combine(currentDir.FullName, Constants.INPUT_FOLDER)))
                    {
                        return currentDir.FullName;
                    }
                    currentDir = currentDir.Parent;
                }
            }
            catch (Exception)
            {
                return null;
            }

            return null;
        }
        #endregion
    }

    public class LstmMetrics
    {
        public double MAE { get; set; }
        public double MSE { get; set; }
        public double RMSE { get; set; }
        public double MAPE { get; set; }
        public double R2 { get; set; }
        public string R2GraphPath { get; set; }
    }

    public class LstmPredictionResult
    {
        public Dictionary<int, double> Predictions { get; set; }
        public string PredictionGraphPath { get; set; }
        public double PredictedValue { get; set; }
        public double PredictedPercentage { get; set; }
        public double TotalAreaKm2 { get; set; }
    }
}
