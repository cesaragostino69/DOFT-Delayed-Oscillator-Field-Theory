# tests/test_simulation.py
import unittest
import numpy as np

# Importamos el modelo que queremos testear
from doft.models.model import DOFTModel

class TestDOFTModel(unittest.TestCase):
    
    def setUp(self):
        """
        Configura los parámetros base para todos los tests.
        Se ejecuta antes de cada método de test.
        """
        self.base_params = {
            'grid_size': 32, # Usamos un grid pequeño para que los tests sean rápidos
            'a': 1.0,
            'tau': 1.0,
            'a_ref': 1.0,
            'tau_ref': 1.0,
            'gamma': 0.05,
            'seed': 42
        }

    def test_run_completes_and_returns_correct_format(self):
        """
        FIXED: Test original actualizado.
        Verifica que la simulación se ejecuta sin errores y que los outputs
        (run_metrics y blocks_df) tienen el formato esperado.
        """
        print("\nRunning: test_run_completes_and_returns_correct_format...")
        # FIX: Añadimos los nuevos parámetros a_ref y tau_ref a la instanciación
        model = DOFTModel(**self.base_params)
        
        run_metrics, blocks_df = model.run()
        
        # Verificar que las métricas principales existen y son numéricas
        self.assertIn('ceff_pulse', run_metrics)
        self.assertTrue(np.isfinite(run_metrics['ceff_pulse']))
        
        self.assertIn('var_c_over_c2', run_metrics)
        self.assertTrue(np.isfinite(run_metrics['var_c_over_c2']))
        
        self.assertIn('lpc_deltaK_neg_frac', run_metrics)
        self.assertTrue(np.isfinite(run_metrics['lpc_deltaK_neg_frac']))
        
        # Verificar que el conteo de frenos es cero en modo pasivo
        self.assertIn('lpc_brake_count', run_metrics)
        self.assertEqual(run_metrics['lpc_brake_count'], 0)
        print("✅ OK")

    def test_integrator_stability(self):
        """
        NEW: Test de estabilidad para el nuevo integrador IMEX.
        En un sistema sin acoplamiento (a=0) y con amortiguamiento (gamma>0),
        la energía debe decaer.
        """
        print("\nRunning: test_integrator_stability...")
        params = self.base_params.copy()
        params['a'] = 0.0 # Sin acoplamiento
        params['gamma'] = 0.1 # Con amortiguamiento

        model = DOFTModel(**params)
        
        # Estado inicial con algo de energía
        model.Q = model.rng.normal(size=model.Q.shape)
        model.P = model.rng.normal(size=model.P.shape)
        
        # Energía inicial (proporcional a P² + Q²)
        initial_energy = np.sum(model.P**2) + np.sum(model.Q**2)
        
        # Simular por un tiempo
        for _ in range(500):
            model._step_imex()
            
        # Energía final
        final_energy = np.sum(model.P**2) + np.sum(model.Q**2)
        
        print(f"  Initial energy: {initial_energy:.4f}, Final energy: {final_energy:.4f}")
        self.assertLess(final_energy, initial_energy)
        print("✅ OK")

    def test_determinism(self):
        """
        NEW: Test de determinismo.
        Dos modelos inicializados con la misma semilla deben producir resultados idénticos.
        """
        print("\nRunning: test_determinism...")
        # Crear dos instancias con exactamente los mismos parámetros
        model1 = DOFTModel(**self.base_params)
        model2 = DOFTModel(**self.base_params)
        
        # Ejecutar ambas simulaciones
        run_metrics1, _ = model1.run()
        run_metrics2, _ = model2.run()
        
        # Comparar una métrica clave. Deberían ser idénticas.
        ceff1 = run_metrics1['ceff_pulse']
        ceff2 = run_metrics2['ceff_pulse']
        
        print(f"  Model 1 ceff_pulse: {ceff1:.8f}")
        print(f"  Model 2 ceff_pulse: {ceff2:.8f}")
        self.assertAlmostEqual(ceff1, ceff2, places=10)
        print("✅ OK")

# Esto permite ejecutar los tests desde la línea de comandos
if __name__ == '__main__':
    unittest.main()