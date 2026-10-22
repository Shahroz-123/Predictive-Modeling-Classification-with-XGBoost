"""Run both beginner project experiments from the command line."""

from pathlib import Path

from src.modeling import run_and_save_results


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    output_path = project_root / "outputs" / "test_metrics.csv"
    results = run_and_save_results(output_path)

    print("Test-set results (higher is better except for RMSE):")
    print(
        results.to_string(
            index=False,
            na_rep="-",
            float_format=lambda value: f"{value:.3f}",
        )
    )
    print(f"\nSaved metrics to: {output_path.relative_to(project_root)}")


if __name__ == "__main__":
    main()
