"""
Tests unitarios para utilidades de tracking (MLflow) del pipeline de entrenamiento.
"""

import unittest
from unittest.mock import Mock, patch
import os
import json
import tempfile
import shutil


class TestMLflowTracking(unittest.TestCase):
    """Tests para funciones de tracking de MLflow"""

    def setUp(self):
        """Configuración inicial para tests"""
        self.temp_dir = tempfile.mkdtemp()
        self.experiment_name = "test_insurance_prediction"

    def tearDown(self):
        """Limpieza después de tests"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch("mlflow.set_tracking_uri")
    @patch("mlflow.set_experiment")
    def test_setup_mlflow(self, mock_set_experiment, mock_set_tracking_uri):
        """Test de configuración de MLflow"""
        # Importar función después de mock para evitar ejecución real
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from training import setup_mlflow

        # Ejecutar función
        setup_mlflow()

        # Verificar llamadas
        mock_set_tracking_uri.assert_called_once()
        mock_set_experiment.assert_called_once()

    @patch("mlflow.log_param")
    def test_log_parameters(self, mock_log_param):
        """Test de registro de parámetros"""
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from training import log_parameters

        params = {
            "model__n_estimators": 100,
            "model__max_depth": 3,
            "model__learning_rate": 0.01,
        }

        # Ejecutar función
        log_parameters(
            params,
            random_state=42,
            test_size=0.2,
            n_iter=50,
            cv_folds=5,
            feature_list=["age", "bmi"],
        )

        # Verificar que se llamó log_param múltiples veces
        self.assertGreater(mock_log_param.call_count, 5)

    @patch("mlflow.log_metric")
    def test_log_metrics(self, mock_log_metric):
        """Test de registro de métricas"""
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from training import log_metrics

        metrics = {
            "train": {"rmse": 4000, "mae": 2000, "r2": 0.85},
            "validation": {"rmse": 4500, "mae": 2200, "r2": 0.83},
            "overfitting_score": 0.02,
        }

        # Ejecutar función
        log_metrics(metrics)

        # Verificar que se llamó log_metric múltiples veces
        self.assertGreater(mock_log_metric.call_count, 5)

    @patch("mlflow.sklearn.log_model")
    @patch("mlflow.log_artifact")
    def test_log_artifacts(self, mock_log_artifact, mock_log_model):
        """Test de registro de artefactos"""
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from training import log_artifacts

        # Crear mock del modelo
        mock_model = Mock()

        # Crear archivos temporales
        report_path = os.path.join(self.temp_dir, "report.txt")
        metadata_path = os.path.join(self.temp_dir, "metadata.json")

        with open(report_path, "w") as f:
            f.write("Test report")

        with open(metadata_path, "w") as f:
            json.dump({"test": "metadata"}, f)

        # Crear mock de search results
        mock_search_results = Mock()
        mock_search_results.cv_results_ = {
            "mean_test_score": [0.8, 0.85, 0.82],
            "rank_test_score": [2, 1, 3],
            "params": [
                {"n_estimators": 100},
                {"n_estimators": 200},
                {"n_estimators": 150},
            ],
        }

        # Ejecutar función
        log_artifacts(
            mock_model,
            {"train": {}, "validation": {}},
            {},
            mock_search_results,
            report_path,
            metadata_path,
        )

        # Verificar llamadas
        mock_log_model.assert_called_once()
        self.assertGreaterEqual(mock_log_artifact.call_count, 2)

    @patch("mlflow.register_model")
    @patch("mlflow.tracking.MlflowClient")
    def test_register_best_model(self, mock_client_class, mock_register_model):
        """Test de registro de modelo en Model Registry"""
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from training import register_best_model

        # Configurar mocks
        mock_model_version = Mock()
        mock_model_version.version = "1"
        mock_register_model.return_value = mock_model_version

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        metrics = {
            "train": {"rmse": 4000, "r2": 0.85},
            "validation": {"rmse": 4500, "r2": 0.83},
            "overfitting_score": 0.02,
        }

        # Ejecutar función
        register_best_model("test_run_id", Mock(), metrics)

        # Verificar llamadas
        mock_register_model.assert_called_once()
        mock_client.transition_model_version_stage.assert_called_once()
        self.assertEqual(mock_client.set_model_version_tag.call_count, 2)


class TestModelPromotion(unittest.TestCase):
    """Tests para script de promoción de modelos"""

    def setUp(self):
        """Configuración inicial para tests"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Limpieza después de tests"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch("mlflow.tracking.MlflowClient")
    @patch("promote_model.MlflowClient")
    def test_get_staging_models(self, mock_client_class_local, mock_client_class):
        """Test de obtención de modelos en Staging"""
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from promote_model import get_staging_models

        # Configurar mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client_class_local.return_value = mock_client

        mock_model_version = Mock()
        mock_model_version.current_stage = "Staging"
        mock_model_version.name = "test_model"
        mock_model_version.version = "1"

        mock_client.search_model_versions.return_value = [mock_model_version]

        # Ejecutar función
        staging_models = get_staging_models()

        # Verificar resultados
        self.assertEqual(len(staging_models), 1)
        self.assertEqual(staging_models[0].current_stage, "Staging")

    @patch("mlflow.tracking.MlflowClient")
    @patch("promote_model.MlflowClient")
    def test_get_model_metrics(self, mock_client_class_local, mock_client_class):
        """Test de obtención de métricas de modelo"""
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from promote_model import get_model_metrics

        # Configurar mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client_class_local.return_value = mock_client

        mock_model_version = Mock()
        mock_model_version.name = "test_model"
        mock_model_version.version = "1"

        mock_model = Mock()
        mock_model.tags = {"validation_rmse": "4500", "validation_r2": "0.83"}

        mock_client.get_model_version.return_value = mock_model

        # Ejecutar función
        metrics = get_model_metrics(mock_model_version)

        # Verificar resultados
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics["rmse"], 4500.0)
        self.assertEqual(metrics["r2"], 0.83)

    def test_check_production_eligibility(self):
        """Test de verificación de elegibilidad para Production"""
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from promote_model import check_production_eligibility

        # Test caso elegible
        metrics_eligible = {"rmse": 4000, "r2": 0.85}
        self.assertTrue(check_production_eligibility(metrics_eligible))

        # Test caso no elegible (RMSE alto)
        metrics_not_eligible_rmse = {"rmse": 6000, "r2": 0.85}
        self.assertFalse(check_production_eligibility(metrics_not_eligible_rmse))

        # Test caso no elegible (R² bajo)
        metrics_not_eligible_r2 = {"rmse": 4000, "r2": 0.75}
        self.assertFalse(check_production_eligibility(metrics_not_eligible_r2))

    @patch("mlflow.tracking.MlflowClient")
    @patch("promote_model.MlflowClient")
    def test_promote_to_production(self, mock_client_class_local, mock_client_class):
        """Test de promoción a Production"""
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from promote_model import promote_to_production

        # Configurar mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client_class_local.return_value = mock_client

        mock_model_version = Mock()
        mock_model_version.name = "test_model"
        mock_model_version.version = "1"

        # Simular que no hay versión actual en Production
        mock_client.get_latest_versions.return_value = []

        # Ejecutar función
        result = promote_to_production(mock_model_version)

        # Verificar resultados
        self.assertTrue(result)
        mock_client.transition_model_version_stage.assert_called_once()


if __name__ == "__main__":
    unittest.main()
