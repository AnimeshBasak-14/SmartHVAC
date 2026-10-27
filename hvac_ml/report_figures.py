"""
report_figures.py - builds every figure used by the report, slides and lifecycle documents.

    python -m hvac_ml.report_figures
"""
import logging

from hvac_ml import figs_data, figs_diagrams, figs_models, figs_ops, figs_summary


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for fn in figs_diagrams.ALL + figs_data.ALL + figs_models.ALL_STAGE6_8 + figs_ops.ALL + figs_summary.ALL:
        fn()


if __name__ == "__main__":
    main()
