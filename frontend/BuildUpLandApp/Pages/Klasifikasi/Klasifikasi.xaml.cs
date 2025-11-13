using BuildUpLandApp.Constant;
using Microsoft.Win32;
using Npgsql;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;

namespace BuildUpLandApp.Pages.Klasifikasi
{
    /// <summary>
    /// Interaction logic for Klasifikasi.xaml
    /// </summary>
    public partial class Klasifikasi : Page
    {
        #region Initialization
        public string SelectedFolderPath { get; set; }
        public string SelectedShapefilePath { get; set; }
        public string SelectedFolderCitraBaruPath { get; set; }

        private string _selectedKecamatanCitraBaru;
        private string _selectedKecamatanKlasifikasi;
        private string _selectedTanggalKlasifikasi;
        private string _classifiedImagePath;
        private string _overlayImagePath;

        private bool _isUpdatingDropdowns = false;
        private string ConnectionString => $"Host={Constants.DB_HOST};Username={Constants.DB_USER};Password={Constants.DB_PASSWORD};Port={Constants.DB_PORT};Database={Constants.DB_NAME}";
        #endregion

        #region Page Load
        public Klasifikasi()
        {
            InitializeComponent();
            this.Loaded += Page_Loaded;
            InitializeDefaultTexts();
        }

        private void InitializeDefaultTexts()
        {
            LatihFolderCitraText.Text = "Pilih folder citra...";
            LatihShapefileText.Text = "Pilih shapefile...";
            KlasifikasiCitraBaruText.Text = "Pilih folder citra baru...";
            AkurasiText.Text = "Akurasi\t: -";
            PresisiText.Text = "Presisi\t: -";
            RecallText.Text = "Recall\t: -";
            F1ScoreText.Text = "F1-Score\t: -";
            LuasTerbangunText.Text = "-";
            LuasNonTerbangunText.Text = "-";
        }

        private async void Page_Loaded(object sender, RoutedEventArgs e)
        {
            await LoadKlasifikasiDropdownDataAsync();
            DisablePlaceholderItem(KecamatanKlasifikasi);
            DisablePlaceholderItem(TanggalKlasifikasi);
            DisablePlaceholderItem(KecamatanCitraBaru);

            await LoadLatestMetricsAsync();
        }

        private void DisablePlaceholderItem(ComboBox comboBox)
        {
            if (comboBox.Items.Count > 0 && comboBox.Items[0] is ComboBoxItem placeholder)
            {
                placeholder.IsEnabled = false;
            }
        }

        private async Task LoadKlasifikasiDropdownDataAsync()
        {
            await GetKecamatanForAllComboBoxes();
            await GetTanggalForComboBox();
        }

        private async Task GetKecamatanForAllComboBoxes(DateTime? filterDate = null)
        {
            var kecamatanList = await FetchKecamatanListAsync(filterDate);
            if (kecamatanList == null) return;
            PopulateComboBox(KecamatanKlasifikasi, kecamatanList, "Pilih kecamatan");
            PopulateComboBox(KecamatanCitraBaru, kecamatanList, "Pilih kecamatan");
        }

        private async Task GetTanggalForComboBox(string filterKecamatan = null)
        {
            var tanggalList = await FetchTanggalListAsync(filterKecamatan);
            if (tanggalList == null) 
                return;
            PopulateComboBox(TanggalKlasifikasi, tanggalList, "Pilih tanggal");
        }

        private async Task LoadLatestMetricsAsync()
        {
            string logRegJson = await GetLatestMetricsJsonAsync("LogReg");
            if (!string.IsNullOrEmpty(logRegJson))
            {
                UpdateLogRegMetricsUI(logRegJson);
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

        private void UpdateLogRegMetricsUI(string json)
        {
            try
            {
                var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
                LogRegMetrics metrics = JsonSerializer.Deserialize<LogRegMetrics>(json, options);

                AkurasiText.Text = $"Akurasi\t: {metrics.Accuracy:F4}";
                PresisiText.Text = $"Presisi\t: {metrics.Precision:F4}";
                RecallText.Text = $"Recall\t: {metrics.Recall:F4}";
                F1ScoreText.Text = $"F1-Score\t: {metrics.F1Score:F4}";
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Gagal parse JSON LogReg: {ex.Message}");
                AkurasiText.Text = "Akurasi\t: -";
                PresisiText.Text = "Presisi\t: -";
                RecallText.Text = "Recall\t: -";
                F1ScoreText.Text = "F1-Score\t: -";
            }
        }

        private async Task<List<string>> FetchKecamatanListAsync(DateTime? filterDate = null)
        {
            var kecamatanList = new List<string>();
            try
            {
                await using var conn = new NpgsqlConnection(ConnectionString); await conn.OpenAsync();
                string sql; NpgsqlCommand cmd;
                if (filterDate.HasValue)
                {
                    sql = @"SELECT DISTINCT w.nama_kec FROM wilayah_administrasi w JOIN pelatihan p ON w.id_wilayah = p.id_wilayah WHERE p.img_date = @date ORDER BY w.nama_kec;";
                    cmd = new NpgsqlCommand(sql, conn);
                    cmd.Parameters.AddWithValue("date", NpgsqlTypes.NpgsqlDbType.Date, filterDate.Value);
                }
                else
                {
                    sql = "SELECT nama_kec FROM wilayah_administrasi ORDER BY nama_kec;";
                    cmd = new NpgsqlCommand(sql, conn);
                }
                await using var reader = await cmd.ExecuteReaderAsync();
                while (await reader.ReadAsync()) 
                { 
                    kecamatanList.Add(reader.GetString(0)); 
                }
                return kecamatanList;
            }
            catch (Exception ex) 
            { 
                HandleDbError("mengambil data kecamatan", ex); 
                return null; 
            }
        }

        private async Task<List<string>> FetchTanggalListAsync(string filterKecamatan = null)
        {
            var tanggalList = new List<string>();
            try
            {
                await using var conn = new NpgsqlConnection(ConnectionString); await conn.OpenAsync();
                string sql; NpgsqlCommand cmd;
                if (!string.IsNullOrEmpty(filterKecamatan))
                {
                    sql = @"SELECT DISTINCT TO_CHAR(p.img_date, 'DD/MM/YYYY') AS img_date_char, p.img_date FROM pelatihan p JOIN wilayah_administrasi w ON p.id_wilayah = w.id_wilayah WHERE w.nama_kec = @kecamatan ORDER BY p.img_date;";
                    cmd = new NpgsqlCommand(sql, conn);
                    cmd.Parameters.AddWithValue("kecamatan", filterKecamatan);
                }
                else
                {
                    sql = "SELECT DISTINCT TO_CHAR(img_date, 'DD/MM/YYYY') AS img_date_char, img_date FROM pelatihan ORDER BY img_date;";
                    cmd = new NpgsqlCommand(sql, conn);
                }
                await using var reader = await cmd.ExecuteReaderAsync();
                while (await reader.ReadAsync()) 
                { 
                    tanggalList.Add(reader.GetString(0)); 
                }
                return tanggalList;
            }
            catch (Exception ex) 
            { 
                HandleDbError("mengambil data tanggal", ex); 
                return null; 
            }
        }

        private void PopulateComboBox(ComboBox comboBox, List<string> items, string placeholderText)
        {
            if (comboBox == null || items == null) 
                return;

            string currentSelection = (comboBox.SelectedIndex > 0 && comboBox.SelectedItem != null) ? comboBox.SelectedItem.ToString() : null;
            comboBox.Items.Clear();
            comboBox.Items.Add(new ComboBoxItem 
            { 
                Content = placeholderText, 
                IsEnabled = false 
            });

            foreach (var item in items) 
            { 
                comboBox.Items.Add(item.Replace("_", " ")); 
            }

            var itemToSelect = comboBox.Items.Cast<object>().FirstOrDefault(i => i is string && i.ToString() == currentSelection);
            if (itemToSelect != null) 
                comboBox.SelectedItem = itemToSelect;
            else 
                comboBox.SelectedIndex = 0;
        }

        private void HandleDbError(string action, Exception ex)
        {
            MessageBox.Show($"Gagal {action} dari database.\n\nDetail: {ex.Message}", "Error Database", MessageBoxButton.OK, MessageBoxImage.Error);
        }
        #endregion

        #region Event Handlers
        private void Back_Click(object sender, RoutedEventArgs e)
        {
            NavigationService nav = NavigationService.GetNavigationService(this);
            if (nav != null && nav.CanGoBack)
                nav.GoBack();
        }

        private void SelectFolder_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new OpenFolderDialog();
            if (dialog.ShowDialog() == true)
            {
                SelectedFolderPath = dialog.FolderName;
                LatihFolderCitraText.Text = SelectedFolderPath;
            }
        }

        private void SelectShapefile_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new OpenFileDialog 
            { 
                Filter = Constants.SHAPEFILE_FILTER 
            };
            if (dialog.ShowDialog() == true)
            {
                SelectedShapefilePath = dialog.FileName;
                LatihShapefileText.Text = SelectedShapefilePath;
            }
        }

        private void ShowOverlayCheckbox_Checked(object sender, RoutedEventArgs e)
        {
            if (!string.IsNullOrEmpty(_overlayImagePath) && File.Exists(_overlayImagePath))
            {
                DisplayImage(ClassifiedImageContainer, _overlayImagePath);
            }
        }

        private void ShowOverlayCheckbox_Unchecked(object sender, RoutedEventArgs e)
        {
            if (!string.IsNullOrEmpty(_classifiedImagePath) && File.Exists(_classifiedImagePath))
            {
                DisplayImage(ClassifiedImageContainer, _classifiedImagePath);
            }
        }
        #endregion

        #region Klasifikasi
        private async void KecamatanKlasifikasi_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (_isUpdatingDropdowns) 
                return;

            ResetCitraBaruInput();
            var comboBox = sender as ComboBox;
            _selectedKecamatanKlasifikasi = (comboBox?.SelectedIndex > 0) ? comboBox.SelectedItem.ToString().Replace(" ", "_") : null;

            _isUpdatingDropdowns = true;
            await UpdateTanggalDropdownBasedOnKecamatan(_selectedKecamatanKlasifikasi);
            _isUpdatingDropdowns = false;
        }

        private async void TanggalKlasifikasi_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (_isUpdatingDropdowns) 
                return;

            ResetCitraBaruInput();
            var comboBox = sender as ComboBox;
            _selectedTanggalKlasifikasi = (comboBox?.SelectedIndex > 0) ? comboBox.SelectedItem.ToString() : null;

            DateTime? selectedDate = null;
            if (_selectedTanggalKlasifikasi != null) 
            {
                try
                {
                    selectedDate = DateTime.ParseExact(_selectedTanggalKlasifikasi, "dd/MM/yyyy", CultureInfo.InvariantCulture);
                }
                catch { }
            }

            _isUpdatingDropdowns = true;
            await UpdateKecamatanDropdownBasedOnTanggal(selectedDate);
            _isUpdatingDropdowns = false;
        }

        private async Task UpdateTanggalDropdownBasedOnKecamatan(string kecamatanName)
        {
            if (TanggalKlasifikasi == null)
                return;

            TanggalKlasifikasi.IsEnabled = false;
            await GetTanggalForComboBox(filterKecamatan: kecamatanName);
            TanggalKlasifikasi.IsEnabled = true;
        }

        private async Task UpdateKecamatanDropdownBasedOnTanggal(DateTime? date)
        {
            KecamatanKlasifikasi.IsEnabled = false;
            KecamatanCitraBaru.IsEnabled = false;
            await GetKecamatanForAllComboBoxes(filterDate: date);
            KecamatanKlasifikasi.IsEnabled = true;
            KecamatanCitraBaru.IsEnabled = true;
        }

        private void CitraBaru_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new OpenFolderDialog();
            if (dialog.ShowDialog() == true)
            {
                KlasifikasiCitraBaruText.Text = dialog.FolderName;
                SelectedFolderCitraBaruPath = dialog.FolderName;
                _isUpdatingDropdowns = true;
                TanggalKlasifikasi.SelectedIndex = 0;
                KecamatanKlasifikasi.SelectedIndex = 0;
                _selectedTanggalKlasifikasi = null;
                _selectedKecamatanKlasifikasi = null;
                _isUpdatingDropdowns = false;
            }
        }

        private void KecamatanCitraBaru_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (_isUpdatingDropdowns) return;
            var comboBox = sender as ComboBox;
            _selectedKecamatanCitraBaru = (comboBox?.SelectedIndex > 0) ? comboBox.SelectedItem.ToString().Replace(" ", "_") : null;
            if (_selectedKecamatanCitraBaru != null)
            {
                _isUpdatingDropdowns = true;
                TanggalKlasifikasi.SelectedIndex = 0;
                KecamatanKlasifikasi.SelectedIndex = 0;
                _selectedTanggalKlasifikasi = null;
                _selectedKecamatanKlasifikasi = null;
                _isUpdatingDropdowns = false;
            }
        }

        private void ResetCitraBaruInput()
        {
            if (KlasifikasiCitraBaruText != null)
            {
                KlasifikasiCitraBaruText.Text = "Pilih folder citra baru...";
                SelectedFolderCitraBaruPath = null;
                if (KecamatanCitraBaru != null) KecamatanCitraBaru.SelectedIndex = 0;
                _selectedKecamatanCitraBaru = null;
            }
        }
        #endregion

        #region Button Click
        private async void PotongCitra_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrWhiteSpace(SelectedFolderPath) || string.IsNullOrWhiteSpace(SelectedShapefilePath))
            {
                MessageBox.Show("Pilih pilih folder citra dan file shapefile yang ingin diproses.", "Input Tidak Lengkap", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            SetUIEnabled(false);
            bool success = await RunPythonScript(Constants.PYTHON_PREPROCESSING, SelectedFolderPath, SelectedShapefilePath);
            SetUIEnabled(true);

            if (success)
            {
                string dataRoot = FindDataRoot(SelectedShapefilePath);
                string dateName = Path.GetFileName(SelectedFolderPath.TrimEnd(Path.DirectorySeparatorChar));
                string shapefileName = Path.GetFileNameWithoutExtension(SelectedShapefilePath);
                string finalOutputPath = (dataRoot != null) ? Path.Combine(dataRoot, "OutputData", dateName, shapefileName) : "(Tidak dapat menentukan path)";

                MessageBox.Show($"Proses pemotongan citra selesai.\nHasil tersimpan di:\n{finalOutputPath}", "Proses Selesai", MessageBoxButton.OK, MessageBoxImage.Information);
            }
        }

        // Training Logistic Regression
        private async void LatihModelLogReg_Click(object sender, RoutedEventArgs e)
        {
            if (!await CheckTrainingDataExistsAsync())
            {
                MessageBox.Show("Belum ada data citra yang diproses (dipotong) dan disimpan di database. Silakan jalankan 'Potong Citra' terlebih dahulu.", "Data Pelatihan Kosong", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            string metricsFilePath = Path.GetTempFileName();
            SetUIEnabled(false);
            bool success = await RunPythonScript(Constants.PYTHON_TRAINING_LOGREG, metricsFilePath: metricsFilePath);
            SetUIEnabled(true);

            if (success && File.Exists(metricsFilePath))
            {
                try
                {
                    string json = await File.ReadAllTextAsync(metricsFilePath);
                    if (!string.IsNullOrWhiteSpace(json))
                    {
                        var options = new JsonSerializerOptions 
                        { 
                            PropertyNameCaseInsensitive = true 
                        };
                        LogRegMetrics metrics = JsonSerializer.Deserialize<LogRegMetrics>(json, options);
                        AkurasiText.Text = $"Akurasi\t: {metrics.Accuracy:F4}";
                        PresisiText.Text = $"Presisi\t: {metrics.Precision:F4}";
                        RecallText.Text = $"Recall\t: {metrics.Recall:F4}";
                        F1ScoreText.Text = $"F1-Score\t: {metrics.F1Score:F4}";
                        MessageBox.Show("Pelatihan Regresi Logistik selesai.", "Sukses", MessageBoxButton.OK, MessageBoxImage.Information);
                    }
                }
                catch (Exception ex) 
                { 
                    HandleFileError("membaca file metrik", ex); 
                }
            }
            if (File.Exists(metricsFilePath)) 
                TryDeleteFile(metricsFilePath);
        }

        // Proses Klasifikasi
        private async void Klasifikasi_Click(object sender, RoutedEventArgs e)
        {
            if (!await CheckModelExistsAsync("LogReg"))
            {
                MessageBox.Show("Model Regresi Logistik belum dilatih. Silakan latih model terlebih dahulu di tab 'Pelatihan Model'.", "Model Tidak Ditemukan", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            string inputFolderPath = null;
            string shapefilePath = null;
            string selectedKecamatanForLog = null;

            if (!string.IsNullOrEmpty(SelectedFolderCitraBaruPath))
            {
                _selectedKecamatanCitraBaru = (KecamatanCitraBaru.SelectedIndex > 0) ? KecamatanCitraBaru.SelectedItem.ToString().Replace(" ", "_") : null;
                if (string.IsNullOrEmpty(_selectedKecamatanCitraBaru)) 
                { 
                    MessageBox.Show("Pilih kecamatan untuk citra baru.", "Input Kurang", MessageBoxButton.OK, MessageBoxImage.Warning); 
                    return; 
                }

                inputFolderPath = SelectedFolderCitraBaruPath;
                shapefilePath = await GetShapefilePathForKecamatanAsync(_selectedKecamatanCitraBaru);
                selectedKecamatanForLog = _selectedKecamatanCitraBaru;
            }
            else if (!string.IsNullOrEmpty(_selectedTanggalKlasifikasi) && !string.IsNullOrEmpty(_selectedKecamatanKlasifikasi))
            {
                inputFolderPath = await GetExistingImageFolderPathAsync(_selectedKecamatanKlasifikasi, _selectedTanggalKlasifikasi);
                shapefilePath = await GetShapefilePathForKecamatanAsync(_selectedKecamatanKlasifikasi);
                selectedKecamatanForLog = _selectedKecamatanKlasifikasi;
            }
            else 
            { 
                MessageBox.Show("Lengkapi input Opsi 1 atau Opsi 2.", "Input Kurang", MessageBoxButton.OK, MessageBoxImage.Warning); 
                return; 
            }

            if (string.IsNullOrEmpty(inputFolderPath) || string.IsNullOrEmpty(shapefilePath)) 
            { 
                MessageBox.Show("Gagal mendapatkan path citra atau shapefile.", "Error Data", MessageBoxButton.OK, MessageBoxImage.Error); 
                return; 
            }

            string resultFilePath = Path.GetTempFileName();
            SetUIEnabled(false);
            ShowOverlayCheckbox.IsChecked = false;
            bool success = await RunPythonScript(Constants.PYTHON_CLASSIFICATION, inputFolderPath, shapefilePath, metricsFilePath: resultFilePath);
            SetUIEnabled(true);

            if (success && File.Exists(resultFilePath))
            {
                try
                {
                    string json = await File.ReadAllTextAsync(resultFilePath);
                    if (!string.IsNullOrWhiteSpace(json))
                    {
                        var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
                        ClassificationResult result = JsonSerializer.Deserialize<ClassificationResult>(json, options);

                        LuasTerbangunText.Text = $"{result.BuiltUpArea:F3}";
                        LuasNonTerbangunText.Text = $"{result.NonBuiltUpArea:F3}";

                        _classifiedImagePath = result.ImagePath;
                        _overlayImagePath = result.OverlayPath;

                        // Default: tampilkan raster tanpa overlay
                        DisplayImage(ClassifiedImageContainer, _classifiedImagePath);
                        if (result.ImagePath != null && File.Exists(result.ImagePath))
                        {
                            string dateFolderName = new DirectoryInfo(result.ImagePath).Parent.Name;
                            string formattedDate = FormatDateFolderName(dateFolderName);
                            RasterKlasifikasi.Text = $"{selectedKecamatanForLog.Replace("_", " ")} - {formattedDate}";
                            LegendPanel.Visibility = Visibility.Visible;
                        }
                        else
                        {
                            RasterKlasifikasi.Text = "-";
                            LegendPanel.Visibility = Visibility.Collapsed;
                            DisplayImage(ClassifiedImageContainer, null);
                        }

                        DisplayImage(TccImageContainer, result.RealImagePath);

                        MessageBox.Show("Klasifikasi berhasil.", "Sukses", MessageBoxButton.OK, MessageBoxImage.Information);
                    }
                }
                catch (Exception ex) { HandleFileError("memproses hasil klasifikasi", ex); }
            }

            if (File.Exists(resultFilePath)) 
                TryDeleteFile(resultFilePath);
        }

        // Skrip Python
        private async Task<bool> RunPythonScript(string scriptConstant, string arg1 = null, string arg2 = null, string resultFilePath = null, string metricsFilePath = null)
        {
            try
            {
                string executablePath = AppDomain.CurrentDomain.BaseDirectory;
                string solutionPath = FindSolutionRoot(executablePath);
                if (solutionPath == null) return false;

                string workingDirectory = Path.Combine(solutionPath, Constants.BACKEND_ROOT_FOLDER);
                string modulePath = scriptConstant.Replace(".py", "").Replace(Path.DirectorySeparatorChar, '.').Replace(Path.AltDirectorySeparatorChar, '.');
                string arguments = $"-m {modulePath}";

                if (arg1 != null) 
                    arguments += $" \"{arg1}\"";
                if (arg2 != null) 
                    arguments += $" \"{arg2}\"";

                if ((scriptConstant == Constants.PYTHON_PREPROCESSING || scriptConstant == Constants.PYTHON_CLASSIFICATION) && arg2 != null)
                {
                    string dataRoot = FindDataRoot(arg2);
                    if (dataRoot == null) 
                    { 
                        MessageBox.Show("Tidak dapat menemukan folder root data.", "Error Path", MessageBoxButton.OK, MessageBoxImage.Error); 
                        return false; 
                    }
                    string outputDataFolder = Path.Combine(dataRoot, "OutputData");
                    Directory.CreateDirectory(outputDataFolder);
                    arguments += $" \"{outputDataFolder}\"";
                }

                if (!string.IsNullOrEmpty(resultFilePath)) 
                    arguments += $" \"{resultFilePath}\"";
                if (!string.IsNullOrEmpty(metricsFilePath)) 
                    arguments += $" \"{metricsFilePath}\"";

                ProcessStartInfo psi = new ProcessStartInfo
                {
                    FileName = "cmd.exe",
                    Arguments = $"/C {Constants.PYTHON} {arguments}",
                    WorkingDirectory = workingDirectory,
                    CreateNoWindow = false,
                    UseShellExecute = false,
                    RedirectStandardOutput = false,
                    RedirectStandardError = false
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
        private void DisplayImage(Border container, string imagePath)
        {
            if (container == null) return;
            container.Background = (SolidColorBrush)(new BrushConverter().ConvertFrom("#E2E8F0"));
            container.Child = new TextBlock 
            { 
                Text = "Gambar akan tampil di sini", 
                HorizontalAlignment = HorizontalAlignment.Center, 
                VerticalAlignment = VerticalAlignment.Center, 
                Foreground = (SolidColorBrush)(new BrushConverter().ConvertFrom("#718096")) 
            };

            if (!string.IsNullOrEmpty(imagePath) && File.Exists(imagePath))
            {
                try
                {
                    BitmapImage bitmap = new BitmapImage();
                    bitmap.BeginInit(); bitmap.UriSource = new Uri(imagePath, UriKind.Absolute);
                    bitmap.CacheOption = BitmapCacheOption.OnLoad; bitmap.EndInit();
                    Image imageControl = new Image 
                    { 
                        Source = bitmap, 
                        Stretch = Stretch.Uniform 
                    };
                    container.Background = Brushes.Transparent; 
                    container.Child = imageControl;
                }
                catch (Exception ex) { 
                    container.Child = new TextBlock 
                    { 
                        Text = $"Gagal memuat:\n{Path.GetFileName(imagePath)}\n{ex.Message}", 
                        Foreground = Brushes.Red, 
                        TextWrapping = TextWrapping.Wrap 
                    }; 
                }
            }
        }

        private string FormatDateFolderName(string dateFolderName) 
        {
            string formattedDate = dateFolderName;

            string[] dateParts = dateFolderName.Split('_');

            if (dateParts.Length == 3)
            {
                formattedDate = $"{dateParts[2]}/{dateParts[1]}/{dateParts[0]}";
            }

            return formattedDate;
        }

        private void SetUIEnabled(bool isEnabled)
        {
            // Tab Pelatihan
            PotongCitraButton.IsEnabled = isEnabled;
            LatihModelLogRegButton.IsEnabled = isEnabled;
            BrowseCitraButton.IsEnabled = isEnabled;
            BrowseShapefileButton.IsEnabled = isEnabled;

            // Tab Klasifikasi
            KlasifikasiButton.IsEnabled = isEnabled;
            BrowseCitraBaruButton.IsEnabled = isEnabled;
            KecamatanKlasifikasi.IsEnabled = isEnabled;
            TanggalKlasifikasi.IsEnabled = isEnabled;
            KecamatanCitraBaru.IsEnabled = isEnabled;

            this.Cursor = isEnabled ? Cursors.Arrow : Cursors.Wait;
        }

        private async Task<bool> CheckModelExistsAsync(string modelType)
        {
            try
            {
                await using var conn = new NpgsqlConnection(ConnectionString); 
                await conn.OpenAsync();
                var sql = "SELECT COUNT(*) FROM model WHERE model_type = @type;";
                await using var cmd = new NpgsqlCommand(sql, conn);
                cmd.Parameters.AddWithValue("type", modelType);
                var count = (long)await cmd.ExecuteScalarAsync();
                return count > 0;
            }
            catch (Exception ex) 
            { 
                HandleDbError($"mengecek model {modelType}", ex); 
                return false; 
            }
        }

        private async Task<bool> CheckTrainingDataExistsAsync()
        {
            try
            {
                await using var conn = new NpgsqlConnection(ConnectionString); await conn.OpenAsync();
                var sql = "SELECT COUNT(*) FROM pelatihan;";
                await using var cmd = new NpgsqlCommand(sql, conn);
                var count = (long)await cmd.ExecuteScalarAsync();
                return count > 0;
            }
            catch (Exception ex) 
            { 
                HandleDbError("mengecek data pelatihan", ex); 
                return false; 
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

        private async Task<string> GetShapefilePathForKecamatanAsync(string kecamatanName)
        {
            try
            {
                await using var conn = new NpgsqlConnection(ConnectionString);
                await conn.OpenAsync();
                var sql = "SELECT path_shp FROM wilayah_administrasi WHERE nama_kec = @nama_kec LIMIT 1;";
                await using var cmd = new NpgsqlCommand(sql, conn);
                cmd.Parameters.AddWithValue("nama_kec", kecamatanName);
                var result = await cmd.ExecuteScalarAsync();
                return result?.ToString();
            }
            catch (NpgsqlException ex)
            {
                MessageBox.Show($"Gagal mengambil path shapefile: {ex.Message}", "Error Database", MessageBoxButton.OK, MessageBoxImage.Error);
                return null;
            }
        }

        private async Task<string> GetExistingImageFolderPathAsync(string kecamatanName, string dateString)
        {
            try
            {
                DateTime date = DateTime.ParseExact(dateString, "dd/MM/yyyy", CultureInfo.InvariantCulture);

                await using var conn = new NpgsqlConnection(ConnectionString);
                await conn.OpenAsync();
                var sql = @"
                    SELECT p.b2_path 
                    FROM pelatihan p 
                    JOIN wilayah_administrasi w ON p.id_wilayah = w.id_wilayah 
                    WHERE w.nama_kec = @nama_kec AND p.img_date = @img_date 
                    LIMIT 1;";
                await using var cmd = new NpgsqlCommand(sql, conn);
                cmd.Parameters.AddWithValue("nama_kec", kecamatanName);
                cmd.Parameters.AddWithValue("img_date", date);

                var b2Path = (await cmd.ExecuteScalarAsync())?.ToString();
                if (string.IsNullOrEmpty(b2Path)) return null;

                string dateFolderName = new DirectoryInfo(b2Path).Parent.Parent.Name;
                string dataRoot = FindDataRoot(b2Path);

                return Path.Combine(dataRoot, Constants.INPUT_FOLDER, "Dataset", dateFolderName);
            }
            catch (NpgsqlException ex)
            {
                MessageBox.Show($"Gagal mengambil path citra yang ada: {ex.Message}", "Error Database", MessageBoxButton.OK, MessageBoxImage.Error);
                return null;
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

    public class ClassificationResult
    {
        public double BuiltUpArea { get; set; }
        public double NonBuiltUpArea { get; set; }
        public string ImagePath { get; set; }
        public string RealImagePath { get; set; }
        public string OverlayPath { get; set; }
    }

    public class LogRegMetrics
    {
        public double Accuracy { get; set; }
        public double Precision { get; set; }
        public double Recall { get; set; }
        public double F1Score { get; set; }
    }
}
