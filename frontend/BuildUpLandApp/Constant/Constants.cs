using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BuildUpLandApp.Constant
{
    public static class Constants
    {
        #region Frontend
        public const string DEFAULT_SELECT_FOLDER_CITRA_TEXT = "Pilih folder citra latih...";
        public const string DEFAULT_SELECT_SHAPEFILE_TEXT = "Pilih file shapefile...";
        #endregion

        #region Backend
        public const string PYTHON = "py";
        public const string BACKEND_ROOT_FOLDER = "backend";
        public const string PYTHON_PREPROCESSING = "app/data/Preprocessing.py";
        public const string PYTHON_TRAINING_LOGREG = "app/model/LogisticRegression/Training.py";
        public const string PYTHON_CLASSIFICATION = "app/model/LogisticRegression/LogReg.py";
        public const string PYTHON_TRAINING_LSTM = "app/model/LSTM/Training.py";
        public const string PYTHON_PREDICTION_LSTM = "app/model/LSTM/LSTM.py";
        #endregion

        #region DB CONFIG
        public const string DB_HOST = "localhost";
        public const string DB_USER = "postgres";
        public const string DB_PASSWORD = "123456";
        public const string DB_PORT = "5432";
        public const string DB_NAME = "build_up_land";
        #endregion

        public const string INPUT_FOLDER = "InputData";
        public const string OUTPUT_FOLDER = "OutputData";
        public const string SHAPEFILE_FILTER = "Shapefile (*.shp)|*.shp";
    }
}
