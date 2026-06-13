"""
Tests unitarios para utilidades de monitoreo del pipeline de scoring.
"""

import unittest
from unittest.mock import patch
import os
import json
import tempfile
import shutil
import pandas as pd


class TestMonitoringUtilities(unittest.TestCase):
    """Tests para funciones de monitoreo de scoring"""

    def setUp(self):
        """Configuración inicial para tests"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Limpieza después de tests"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_load_production_batch(self):
        """Test de carga de batch de producción"""
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from scoring import load_production_batch

        # Crear archivos CSV temporales
        feats_data = pd.DataFrame(
            {
                "age": [25, 30, 35],
                "sex": ["male", "female", "male"],
                "bmi": [25.0, 30.0, 35.0],
                "children": [0, 1, 2],
                "smoker": ["no", "yes", "no"],
                "region": ["northeast", "northwest", "southeast"],
            }
        )

        target_data = pd.DataFrame({"charges": [5000, 10000, 15000]})

        feats_path = os.path.join(self.temp_dir, "dataset_test_feats.csv.csv")
        target_path = os.path.join(self.temp_dir, "dataset_test_target.csv.csv")

        feats_data.to_csv(feats_path, index=False)
        target_data.to_csv(target_path, index=False)

        # Ejecutar función
        df_features, df_target = load_production_batch("dataset_test", self.temp_dir)

        # Verificar resultados
        self.assertIsNotNone(df_features)
        self.assertIsNotNone(df_target)
        self.assertEqual(len(df_features), 3)
        self.assertEqual(len(df_target), 3)

    def test_load_production_batch_no_target(self):
        """Test de carga de batch sin target"""
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from scoring import load_production_batch

        # Crear solo archivo de features
        feats_data = pd.DataFrame(
            {
                "age": [25, 30, 35],
                "sex": ["male", "female", "male"],
                "bmi": [25.0, 30.0, 35.0],
                "children": [0, 1, 2],
                "smoker": ["no", "yes", "no"],
                "region": ["northeast", "northwest", "southeast"],
            }
        )

        feats_path = os.path.join(self.temp_dir, "dataset_test_feats.csv.csv")
        feats_data.to_csv(feats_path, index=False)

        # Ejecutar función
        df_features, df_target = load_production_batch("dataset_test", self.temp_dir)

        # Verificar resultados
        self.assertIsNotNone(df_features)
        self.assertIsNone(df_target)
        self.assertEqual(len(df_features), 3)

    def test_calculate_performance_metrics(self):
        """Test de cálculo de métricas de performance"""
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from scoring import calculate_performance_metrics

        # Crear DataFrame con predicciones y valores reales
        results_df = pd.DataFrame(
            {
                "predicted_charges": [5000, 10000, 15000],
                "actual_charges": [4800, 10200, 14800],
            }
        )

        # Ejecutar función
        metrics = calculate_performance_metrics(results_df)

        # Verificar resultados
        self.assertIsNotNone(metrics)
        self.assertIn("rmse", metrics)
        self.assertIn("mae", metrics)
        self.assertIn("r2", metrics)
        self.assertIn("mape", metrics)
        self.assertGreater(metrics["r2"], 0)

    def test_calculate_performance_metrics_no_target(self):
        """Test de cálculo de métricas sin target"""
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from scoring import calculate_performance_metrics

        # Crear DataFrame sin valores reales
        results_df = pd.DataFrame({"predicted_charges": [5000, 10000, 15000]})

        # Ejecutar función
        metrics = calculate_performance_metrics(results_df)

        # Verificar resultados
        self.assertIsNone(metrics)

    def test_detect_drift(self):
        """Test de detección de drift"""
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from scoring import detect_drift

        # Crear DataFrame de features
        df_features = pd.DataFrame(
            {"age": [25, 30, 35], "bmi": [25.0, 30.0, 35.0], "children": [0, 1, 2]}
        )

        df_target = pd.DataFrame({"charges": [5000, 10000, 15000]})

        # Estadísticas de entrenamiento
        training_stats = {
            "age_mean": 30.0,
            "age_std": 5.0,
            "bmi_mean": 30.0,
            "bmi_std": 5.0,
            "children_mean": 1.0,
            "children_std": 1.0,
            "target_mean": 10000.0,
            "target_std": 5000.0,
        }

        # Ejecutar función
        drift_results = detect_drift(df_features, df_target, training_stats)

        # Verificar resultados
        self.assertIsNotNone(drift_results)
        self.assertIn("features_drift", drift_results)
        self.assertIn("target_drift", drift_results)
        self.assertIn("overall_status", drift_results)
        self.assertIn(drift_results["overall_status"], ["OK", "WARNING"])

    def test_detect_drift_no_target(self):
        """Test de detección de drift sin target"""
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from scoring import detect_drift

        # Crear DataFrame de features sin target
        df_features = pd.DataFrame(
            {"age": [25, 30, 35], "bmi": [25.0, 30.0, 35.0], "children": [0, 1, 2]}
        )

        # Estadísticas de entrenamiento
        training_stats = {
            "age_mean": 30.0,
            "age_std": 5.0,
            "bmi_mean": 30.0,
            "bmi_std": 5.0,
            "children_mean": 1.0,
            "children_std": 1.0,
        }

        # Ejecutar función
        drift_results = detect_drift(df_features, None, training_stats)

        # Verificar resultados
        self.assertIsNotNone(drift_results)
        self.assertIn("features_drift", drift_results)
        self.assertIn("target_drift", drift_results)
        self.assertIsNone(drift_results["target_drift"])

    def test_compare_with_training(self):
        """Test de comparación con entrenamiento"""
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from scoring import compare_with_training

        # Métricas del batch
        batch_metrics = {"rmse": 4500, "r2": 0.85}

        # Métricas de entrenamiento
        training_metrics = {"validation_rmse": 4000, "validation_r2": 0.87}

        # Ejecutar función
        comparison = compare_with_training(batch_metrics, training_metrics)

        # Verificar resultados
        self.assertIsNotNone(comparison)
        self.assertIn("status", comparison)
        self.assertIn("details", comparison)
        self.assertIn("train_rmse", comparison)
        self.assertIn("batch_rmse", comparison)
        self.assertIn("train_r2", comparison)
        self.assertIn("batch_r2", comparison)
        self.assertIn(comparison["status"], ["OK", "WARNING"])

    def test_compare_with_training_no_metrics(self):
        """Test de comparación sin métricas del batch"""
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from scoring import compare_with_training

        # Métricas de entrenamiento
        training_metrics = {"validation_rmse": 4000, "validation_r2": 0.87}

        # Ejecutar función sin métricas del batch
        comparison = compare_with_training(None, training_metrics)

        # Verificar resultados
        self.assertIsNotNone(comparison)
        self.assertEqual(comparison["status"], "N/A")

    def test_save_predictions_to_csv(self):
        """Test de guardado de predicciones en CSV"""
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from scoring import save_predictions_to_csv

        # Crear DataFrame de predicciones
        predictions_df = pd.DataFrame(
            {"age": [25, 30, 35], "predicted_charges": [5000, 10000, 15000]}
        )

        # Ejecutar función
        csv_path = save_predictions_to_csv(predictions_df, "test_batch", self.temp_dir)

        # Verificar que el archivo existe
        self.assertTrue(os.path.exists(csv_path))

        # Verificar contenido
        loaded_df = pd.read_csv(csv_path)
        self.assertEqual(len(loaded_df), 3)

    def test_get_training_stats_from_metadata(self):
        """Test de obtención de estadísticas de entrenamiento desde metadata"""
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from scoring import get_training_stats_from_metadata

        # Metadata con estadísticas
        metadata = {
            "training_stats": {
                "age_mean": 30.0,
                "age_std": 5.0,
                "bmi_mean": 30.0,
                "bmi_std": 5.0,
            }
        }

        # Ejecutar función
        training_stats = get_training_stats_from_metadata(metadata)

        # Verificar resultados
        self.assertIsNotNone(training_stats)
        # La función usa valores por defecto si no están en metadata, así que verificamos que tenga las claves
        self.assertIn("age_mean", training_stats)
        self.assertIn("bmi_mean", training_stats)

    def test_get_training_stats_from_metadata_default(self):
        """Test de obtención de estadísticas con valores por defecto"""
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from scoring import get_training_stats_from_metadata

        # Metadata sin estadísticas
        metadata = {}

        # Ejecutar función
        training_stats = get_training_stats_from_metadata(metadata)

        # Verificar resultados
        self.assertIsNotNone(training_stats)
        self.assertIn("age_mean", training_stats)
        self.assertIn("bmi_mean", training_stats)
        self.assertIn("target_mean", training_stats)


class TestMonitoringDashboard(unittest.TestCase):
    """Tests para funciones de dashboard de monitoreo"""

    def setUp(self):
        """Configuración inicial para tests"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Limpieza después de tests"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_load_monitoring_report(self):
        """Test de carga de reporte de monitoreo"""
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from monitoring_dashboard import load_monitoring_report

        # Crear reporte JSON temporal
        report = {
            "total_batches": 3,
            "total_predictions": 1000,
            "batches_with_target": 2,
            "status_summary": {"OK": 1, "WARNING": 2},
            "batches": {
                "batch1": {
                    "n_records": 500,
                    "metrics": {"rmse": 4500, "r2": 0.85},
                    "final_status": "OK",
                }
            },
            "training_metrics": {
                "validation_rmse": 4000,
                "validation_r2": 0.87,
                "validation_mae": 2000,
            },
        }

        report_path = os.path.join(self.temp_dir, "monitoring_report_test.json")
        with open(report_path, "w") as f:
            json.dump(report, f)

        # Ejecutar función
        loaded_report = load_monitoring_report(report_path)

        # Verificar resultados
        self.assertIsNotNone(loaded_report)
        self.assertEqual(loaded_report["total_batches"], 3)
        self.assertEqual(loaded_report["total_predictions"], 1000)

    def test_load_monitoring_report_auto(self):
        """Test de carga automática de reporte más reciente"""
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from monitoring_dashboard import load_monitoring_report

        # Crear múltiples reportes
        for i in range(3):
            report = {"total_batches": i + 1, "total_predictions": (i + 1) * 100}
            report_path = os.path.join(
                self.temp_dir, f"monitoring_report_test_{i}.json"
            )
            with open(report_path, "w") as f:
                json.dump(report, f)

        # Simular OUTPUT_DIR apuntando a temp_dir
        import monitoring_dashboard

        original_output_dir = monitoring_dashboard.OUTPUT_DIR
        monitoring_dashboard.OUTPUT_DIR = self.temp_dir

        try:
            # Ejecutar función sin especificar ruta
            loaded_report = load_monitoring_report()

            # Verificar que cargó un reporte
            self.assertIsNotNone(loaded_report)
        finally:
            # Restaurar valor original
            monitoring_dashboard.OUTPUT_DIR = original_output_dir

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_plot_performance_metrics(self, mock_close, mock_savefig):
        """Test de generación de gráfico de métricas de performance"""
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from monitoring_dashboard import plot_performance_metrics

        # Crear reporte de prueba
        report = {
            "training_metrics": {"validation_rmse": 4000, "validation_r2": 0.87},
            "batches": {
                "batch1": {"metrics": {"rmse": 4500, "r2": 0.85}},
                "batch2": {"metrics": {"rmse": 4200, "r2": 0.86}},
            },
        }

        # Ejecutar función
        plot_performance_metrics(report, self.temp_dir)

        # Verificar que se llamó a savefig
        self.assertTrue(mock_savefig.called)

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_plot_drift_detection(self, mock_close, mock_savefig):
        """Test de generación de gráfico de drift detection"""
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from monitoring_dashboard import plot_drift_detection

        # Crear reporte de prueba
        report = {
            "batches": {
                "batch1": {
                    "drift": {
                        "features": {
                            "age": {"mean_diff": 0.1},
                            "bmi": {"mean_diff": 0.2},
                            "children": {"mean_diff": 0.05},
                        }
                    }
                }
            }
        }

        # Ejecutar función
        plot_drift_detection(report, self.temp_dir)

        # Verificar que se llamó a savefig
        self.assertTrue(mock_savefig.called)

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_plot_batch_status_summary(self, mock_close, mock_savefig):
        """Test de generación de gráfico de resumen de estados"""
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from monitoring_dashboard import plot_batch_status_summary

        # Crear reporte de prueba
        report = {"status_summary": {"OK": 1, "WARNING": 2, "ERROR": 0}}

        # Ejecutar función
        plot_batch_status_summary(report, self.temp_dir)

        # Verificar que se llamó a savefig
        self.assertTrue(mock_savefig.called)

    def test_generate_html_dashboard(self):
        """Test de generación de dashboard HTML"""
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from monitoring_dashboard import generate_html_dashboard

        # Crear reporte de prueba
        report = {
            "total_batches": 3,
            "total_predictions": 1000,
            "batches_with_target": 2,
            "status_summary": {"OK": 1, "WARNING": 2},
            "batches": {
                "batch1": {
                    "n_records": 500,
                    "metrics": {"rmse": 4500, "r2": 0.85},
                    "final_status": "OK",
                }
            },
            "training_metrics": {
                "validation_rmse": 4000,
                "validation_r2": 0.87,
                "validation_mae": 2000,
            },
        }

        # Ejecutar función
        html_path = generate_html_dashboard(report, self.temp_dir)

        # Verificar que el archivo existe
        self.assertTrue(os.path.exists(html_path))

        # Verificar contenido HTML
        with open(html_path, "r") as f:
            content = f.read()
            self.assertIn("Dashboard de Monitoreo", content)
            self.assertIn("batch1", content)


if __name__ == "__main__":
    unittest.main()
