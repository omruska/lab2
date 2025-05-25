import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import pytest

# Константи
UNIVERSE_RANGE = np.arange(0, 1.1, 0.1)


class FuzzyLogicSystem:
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file, sep=r'\s+')
        self.x1 = ctrl.Antecedent(UNIVERSE_RANGE, 'X1')
        self.x2 = ctrl.Antecedent(UNIVERSE_RANGE, 'X2')
        self.y = ctrl.Consequent(UNIVERSE_RANGE, 'Y')
        self._define_membership_functions()
        self._define_rules()
        self.control_system = ctrl.ControlSystem(self.rules)
        self.control_simulation = ctrl.ControlSystemSimulation(self.control_system)

    def _define_membership_functions(self):
        for var in [self.x1, self.x2, self.y]:
            var['low'] = fuzz.trimf(var.universe, [0, 0, 0.5])
            var['medium'] = fuzz.trimf(var.universe, [0, 0.5, 1])
            var['high'] = fuzz.trimf(var.universe, [0.5, 1, 1])

    def _define_rules(self):
        self.rules = [
            ctrl.Rule(self.x1['low'] & self.x2['low'], self.y['low']),
            ctrl.Rule(self.x1['low'] & self.x2['medium'], self.y['low']),
            ctrl.Rule(self.x1['low'] & self.x2['high'], self.y['medium']),
            ctrl.Rule(self.x1['medium'] & self.x2['low'], self.y['low']),
            ctrl.Rule(self.x1['medium'] & self.x2['medium'], self.y['medium']),
            ctrl.Rule(self.x1['medium'] & self.x2['high'], self.y['high']),
            ctrl.Rule(self.x1['high'] & self.x2['low'], self.y['medium']),
            ctrl.Rule(self.x1['high'] & self.x2['medium'], self.y['high']),
            ctrl.Rule(self.x1['high'] & self.x2['high'], self.y['high'])
        ]

    def run_simulation(self):
        results = []
        for x1_val, x2_val in zip(self.data['X1'], self.data['X2']):
            self.control_simulation.input['X1'] = x1_val
            self.control_simulation.input['X2'] = x2_val
            self.control_simulation.compute()
            results.append(self.control_simulation.output['Y'])
        return results


def visualize_results(X1, X2, Y_real):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X1, X2, Y_real, color='blue', label='Data Points', s=50)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y')
    ax.set_title('Fuzzy Logic Output in 3D')
    plt.legend()
    plt.show()


# Модульні тести
@pytest.fixture
def fuzzy_system():
    return FuzzyLogicSystem('test_data.dat')


def test_membership_functions(fuzzy_system):
    assert len(fuzzy_system.x1.terms) == 3
    assert len(fuzzy_system.x2.terms) == 3
    assert len(fuzzy_system.y.terms) == 3


def test_rule_count(fuzzy_system):
    assert len(fuzzy_system.rules) == 9


def test_simulation_output(fuzzy_system):
    results = fuzzy_system.run_simulation()
    assert len(results) == len(fuzzy_system.data)
    assert all(0 <= val <= 1 for val in results)


def test_visualization():
    X1 = np.array([0.1, 0.5, 0.9])
    Y_real = np.array([0.2, 0.6, 0.8])
    Y_pred = np.array([0.3, 0.55, 0.85])
    visualize_results(X1, Y_real, Y_pred)


def test_fuzzy_logic_boundaries(fuzzy_system):
    fuzzy_system.control_simulation.input['X1'] = 0
    fuzzy_system.control_simulation.input['X2'] = 0
    fuzzy_system.control_simulation.compute()
    assert 0 <= fuzzy_system.control_simulation.output['Y'] <= 1

    fuzzy_system.control_simulation.input['X1'] = 1
    fuzzy_system.control_simulation.input['X2'] = 1
    fuzzy_system.control_simulation.compute()
    assert 0 <= fuzzy_system.control_simulation.output['Y'] <= 1


def test_fuzzy_logic_midpoint(fuzzy_system):
    fuzzy_system.control_simulation.input['X1'] = 0.5
    fuzzy_system.control_simulation.input['X2'] = 0.5
    fuzzy_system.control_simulation.compute()
    assert 0 <= fuzzy_system.control_simulation.output['Y'] <= 1


def test_fuzzy_system_data_loaded(fuzzy_system):
    assert not fuzzy_system.data.empty


def test_fuzzy_system_data_columns(fuzzy_system):
    assert set(fuzzy_system.data.columns) == {'X1', 'X2', 'Y'}


def test_simulation_output_length(fuzzy_system):
    results = fuzzy_system.run_simulation()
    assert len(results) == len(fuzzy_system.data)


def test_simulation_output_range(fuzzy_system):
    results = fuzzy_system.run_simulation()
    assert all(0 <= val <= 1 for val in results)


if __name__ == "__main__":
    system = FuzzyLogicSystem('test_data.dat')
    results = system.run_simulation()
    visualize_results(system.data['X1'], system.data['Y'], results)