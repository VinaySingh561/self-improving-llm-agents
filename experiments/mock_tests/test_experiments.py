"""Test suite for experiment framework."""

import pytest
import tempfile
import numpy as np

from experiments.experiment_runner import (
    BaselineType,
    ExperimentResult,
    AggregatedResults,
    StaticBaseline,
    RandomMutationBaseline,
    ProposedMethodBaseline,
    ExperimentRunner,
)


class TestBaselineType:
    """Test BaselineType enum."""
    
    def test_baseline_types_exist(self):
        """Test all baseline types are defined."""
        assert BaselineType.STATIC.value == "static"
        assert BaselineType.RANDOM_MUTATION.value == "random_mutation"
        assert BaselineType.PROPOSED.value == "proposed"


class TestExperimentResult:
    """Test ExperimentResult dataclass."""
    
    def test_result_creation(self):
        """Test creating experiment result."""
        result = ExperimentResult(
            experiment_id="exp1",
            baseline_type=BaselineType.STATIC,
            seed=42,
            prompt_version="v1.0.0",
            num_episodes=10,
            num_steps_total=1000,
            success_rate=0.8,
            average_score=100.0,
            average_steps=100.0,
            average_survival=1.0,
            min_score=50.0,
            max_score=150.0,
            score_std=20.0,
        )
        
        assert result.experiment_id == "exp1"
        assert result.success_rate == 0.8
        assert result.average_score == 100.0


class TestAggregatedResults:
    """Test AggregatedResults dataclass."""
    
    def test_initialization(self):
        """Test creating aggregated results."""
        agg = AggregatedResults(
            baseline_type=BaselineType.STATIC,
            num_seeds=5,
        )
        
        assert agg.baseline_type == BaselineType.STATIC
        assert agg.num_seeds == 5
        assert len(agg.all_results) == 0
        assert not agg.is_complete()
    
    def test_completion_check(self):
        """Test completion status check."""
        agg = AggregatedResults(
            baseline_type=BaselineType.STATIC,
            num_seeds=2,
        )
        
        # Add one result
        result = ExperimentResult(
            experiment_id="exp1",
            baseline_type=BaselineType.STATIC,
            seed=42,
            prompt_version="v1.0.0",
            num_episodes=10,
            num_steps_total=1000,
            success_rate=0.8,
            average_score=100.0,
            average_steps=100.0,
            average_survival=1.0,
            min_score=50.0,
            max_score=150.0,
            score_std=20.0,
        )
        
        agg.all_results.append(result)
        assert not agg.is_complete()
        
        # Add second result
        agg.all_results.append(result)
        assert agg.is_complete()


class TestStaticBaseline:
    """Test static baseline."""
    
    def test_initialization(self):
        """Test baseline initialization."""
        baseline = StaticBaseline()
        assert baseline.baseline_type == BaselineType.STATIC
    
    def test_mock_episode_generation(self):
        """Test mock episode generation."""
        baseline = StaticBaseline()
        
        episode = baseline._mock_episode(ep_id=0, seed=42)
        
        assert episode.episode_id.startswith("static_ep")
        assert episode.prompt_version == "v1.0.0"
        assert episode.total_steps > 0


class TestRandomMutationBaseline:
    """Test random mutation baseline."""
    
    def test_initialization(self):
        """Test baseline initialization."""
        baseline = RandomMutationBaseline()
        assert baseline.baseline_type == BaselineType.RANDOM_MUTATION
    
    def test_mock_episode_with_mutation(self):
        """Test mock episode with mutation."""
        baseline = RandomMutationBaseline()
        
        episode = baseline._mock_episode_with_mutation(ep_id=0, seed=42)
        
        assert episode.episode_id.startswith("random_ep")
        assert episode.prompt_version == "v1.0.1"
        assert episode.total_steps > 0


class TestProposedMethodBaseline:
    """Test proposed method baseline."""
    
    def test_initialization(self):
        """Test baseline initialization."""
        baseline = ProposedMethodBaseline()
        assert baseline.baseline_type == BaselineType.PROPOSED
    
    def test_mock_episode_proposed(self):
        """Test mock episode with proposed method."""
        baseline = ProposedMethodBaseline()
        
        episode = baseline._mock_episode_proposed(ep_id=0, seed=42)
        
        assert episode.episode_id.startswith("proposed_ep")
        assert episode.prompt_version == "v1.2.0"
        assert episode.total_steps > 0


class TestExperimentRunner:
    """Test experiment runner."""
    
    def test_initialization(self):
        """Test runner initialization."""
        runner = ExperimentRunner(log_dir="/tmp")
        assert runner.log_dir == "/tmp"
        assert len(runner.results) == 0
    
    def test_ci_computation(self):
        """Test confidence interval computation."""
        values = [100, 102, 98, 101, 99]
        
        ci = ExperimentRunner._compute_ci(values)
        
        assert len(ci) == 2
        assert ci[0] < 100  # Lower bound
        assert ci[1] > 100  # Upper bound
        assert ci[0] < ci[1]
    
    def test_ci_single_value(self):
        """Test CI with single value (edge case)."""
        values = [100]
        
        ci = ExperimentRunner._compute_ci(values)
        
        assert ci[0] == 100.0
        assert ci[1] == 100.0
    
    def test_results_aggregation(self):
        """Test results aggregation."""
        agg = AggregatedResults(
            baseline_type=BaselineType.STATIC,
            num_seeds=3,
        )
        
        # Add results with different scores
        for i, score in enumerate([100, 110, 90]):
            result = ExperimentResult(
                experiment_id=f"exp{i}",
                baseline_type=BaselineType.STATIC,
                seed=42 + i,
                prompt_version="v1.0.0",
                num_episodes=10,
                num_steps_total=score * 10,
                success_rate=0.8,
                average_score=float(score),
                average_steps=float(score),
                average_survival=1.0,
                min_score=float(score - 10),
                max_score=float(score + 10),
                score_std=5.0,
            )
            agg.all_results.append(result)
        
        runner = ExperimentRunner()
        runner._aggregate_results(agg)
        
        assert agg.average_score_mean == pytest.approx(100.0, rel=0.01)
        assert agg.average_score_std > 0
        assert agg.average_score_ci[0] < agg.average_score_ci[1]


class TestIntegrationExperiments:
    """Integration tests for experiment framework."""
    
    def test_experiment_runner_workflow(self):
        """Test complete experiment workflow."""
        runner = ExperimentRunner(log_dir="/tmp")
        
        # Mock agent and env
        agent = None  # Not used in mock mode
        env = None
        
        # Run experiments (will use mock baselines)
        results = runner.run_experiments(
            agent=agent,
            env_wrapper=env,
            num_episodes=5,
            seeds=[42, 123],
        )
        
        assert len(results) == 3  # 3 baselines
        assert BaselineType.STATIC in results
        assert BaselineType.RANDOM_MUTATION in results
        assert BaselineType.PROPOSED in results
        
        # Check completeness
        for baseline_type, agg in results.items():
            assert agg.num_seeds == 2
            assert agg.is_complete()
            assert agg.average_score_mean > 0
    
    def test_results_summary_generation(self):
        """Test results summary generation."""
        runner = ExperimentRunner()
        
        # Run minimal experiment
        results = runner.run_experiments(
            agent=None,
            env_wrapper=None,
            num_episodes=2,
            seeds=[42],
        )
        
        summary = runner.get_results_summary()
        
        assert "EXPERIMENT RESULTS SUMMARY" in summary
        assert "STATIC" in summary
        assert "RANDOM_MUTATION" in summary
        assert "PROPOSED" in summary
        assert "Success Rate" in summary
        assert "Average Score" in summary
    
    def test_performance_comparison(self):
        """Test that proposed method shows better performance."""
        runner = ExperimentRunner()
        
        results = runner.run_experiments(
            agent=None,
            env_wrapper=None,
            num_episodes=10,
            seeds=[42, 123, 456],
        )
        
        static_score = results[BaselineType.STATIC].average_score_mean
        random_score = results[BaselineType.RANDOM_MUTATION].average_score_mean
        proposed_score = results[BaselineType.PROPOSED].average_score_mean
        
        # Proposed should be better than static (in mock, it's designed to be)
        assert proposed_score >= static_score


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
