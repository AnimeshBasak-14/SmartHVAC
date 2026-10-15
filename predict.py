"""
predict.py - get advice for ONE building from the command line (no GUI needed).

Examples:
    python predict.py --list-shapes
    python predict.py --shape 1 --orientation 2 --glazing-area 0.25 --glazing-dist 1
    python predict.py --shape 12 --glazing-area 0.4 --glazing-dist 5 --price 9.5
"""
import argparse

from src import config
from src.recommend import (BUILDING_SHAPES, annual_energy_cost, building_from_shape, design_optimizer,
                           load_bundle, predict_building, service_plan, shape_label)


def main():
    p = argparse.ArgumentParser(description="SmartHVAC Advisor - predict loads for one building")
    p.add_argument("--list-shapes", action="store_true", help="show the 12 building shapes and exit")
    p.add_argument("--shape", type=int, default=1, choices=range(1, 13), help="building shape 1-12")
    p.add_argument("--orientation", type=int, default=2, choices=[2, 3, 4, 5], help="2=N 3=E 4=S 5=W")
    p.add_argument("--glazing-area", type=float, default=0.25, choices=[0.0, 0.1, 0.25, 0.4])
    p.add_argument("--glazing-dist", type=int, default=1, choices=range(0, 6),
                   help="0=none 1=uniform 2=N 3=E 4=S 5=W")
    p.add_argument("--price", type=float, default=config.ELECTRICITY_PRICE_INR, help="₹ per kWh")
    a = p.parse_args()

    if a.list_shapes:
        for i in range(len(BUILDING_SHAPES)):
            print(shape_label(i))
        return

    bundle = load_bundle()
    b = building_from_shape(a.shape - 1, a.orientation, a.glazing_area, a.glazing_dist)
    r = predict_building(bundle, b)
    cost = annual_energy_cost(b, r, a.price)
    plan = service_plan(r["demand_class"])

    print("\n=== SmartHVAC Advisor ===")
    print(shape_label(a.shape - 1))
    print(f"Orientation {config.ORIENTATION_NAMES[a.orientation]}, glazing {a.glazing_area:.0%} "
          f"({config.GLAZING_DIST_NAMES[a.glazing_dist]})\n")
    for t in config.TARGETS:
        print(f"{config.PRETTY[t]:<26} {r[t]:6.2f}   (90% range {r[t + '_low']:.2f} - {r[t + '_high']:.2f})")
    print(f"{'Total load':<26} {r['total_load']:6.2f} kWh/m²  ->  demand class: {r['demand_class']}")
    print(f"Classifier says: {r['classifier_class']} ({r['classifier_confidence']:.0%} confident)")
    print(f"\nFloor area {cost['floor_area_m2']:.0f} m²  |  ~{cost['annual_kwh']:,.0f} kWh/yr  |  "
          f"~₹{cost['annual_cost_inr']:,.0f} per year")
    print(f"\nRecommended equipment : {plan['equipment']}")
    print(f"Maintenance plan      : {plan['maintenance']}")
    print(f"Advice                : {plan['advice']}")
    print("\nTop 3 better designs (windows / orientation):")
    print(design_optimizer(bundle, b).head(3).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
