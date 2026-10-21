"""
business_financial_engine.py
Enterprise Financial KPI & Business Simulation Engine for HVAC & Smart Appliance GML.
Translates GNN Model Predictions (Demand, Failure Risk, Churn, Inventory) into
an Executive P&L Statement with fully configurable company business cost drivers.
"""
from dataclasses import dataclass, field
import numpy as np
import torch
from config import Config

@dataclass
class CompanyBusinessInputs:
    """
    Configurable Business Inputs from the Enterprise / Corporate POV.
    Allows CFO / Operations leaders to parameterize tariffs, margins, costs, and penalties.
    """
    # 1. Service Pricing & Revenue Inputs (INR)
    base_metro_ticket_cooling: float = 2400.0   # Average HVAC service ticket price in Metro
    base_metro_ticket_heating: float = 1500.0   # Average heating appliance service ticket in Metro
    base_village_ticket_cooling: float = 1650.0 # Subsidized / competitive ticket in Village Hubs
    base_village_ticket_heating: float = 1250.0 # Winter heating ticket in Village Hubs
    amc_subscription_monthly_fee: float = 450.0 # Monthly AMC subscription revenue per active contract
    amc_coverage_ratio: float = 0.35            # Fraction of customer base on recurring AMC

    # 2. Cross-Sell & Customer Retention Economics
    cross_sell_cooling_rev: float = 4500.0      # Net margin from cross-selling cooling upgrade
    cross_sell_heating_rev: float = 3200.0      # Net margin from cross-selling induction kettle/geyser
    marketing_campaign_cost_per_user: float = 180.0  # App push / SMS / Tele-calling CAC per sleeper
    sleeper_reactivation_rate: float = 0.42     # Conversion rate of targeted sleeper retention

    # 3. Preventive vs. Reactive Maintenance Costs (INR)
    preventive_inspection_cost: float = 950.0   # Scheduled proactive technician visit + tuning
    emergency_breakdown_cost: float = 4200.0    # Unplanned catastrophic failure repair + parts replacement
    avoided_breakdown_efficiency: float = 0.78  # % of flagged high-risk devices saved from catastrophic failure
    unavoidable_emergency_rate: float = 0.04    # Baseline unavoidable sudden failures (grid surges, etc.)

    # 4. Supply Chain, Inventory & SLA Costs
    stockout_penalty_per_unit: float = 85.0     # Expedited courier surcharge + customer SLA credit per missing part
    holding_cost_annual_rate: float = 0.25      # Annual warehouse holding & depreciation rate
    logistics_overhead_pct: float = 0.08        # Logistics dispatch cost as % of inventory value

class BusinessFinancialEngine:
    @staticmethod
    def evaluate_monthly_financials(
        month: int,
        data,
        model_outputs,
        tables,
        business_inputs: CompanyBusinessInputs = None
    ) -> dict:
        """
        Calculates audited monthly financials based on GNN predictions and company business parameters.
        """
        if business_inputs is None:
            business_inputs = CompanyBusinessInputs()

        df_res, df_loc, df_dev, df_part, _, _, _ = tables
        season_name, _, cool_mult, heat_mult = Config.MONTH_REGIME[month]

        demand_pred = model_outputs["demand_pred"].detach().cpu().numpy().flatten()
        failure_prob = torch.sigmoid(model_outputs["failure_logits"]).detach().cpu().numpy().flatten()
        sleeper_prob = torch.sigmoid(model_outputs["sleeper_logits"]).detach().cpu().numpy().flatten()
        inventory_pred = model_outputs["inventory_pred"].detach().cpu().numpy().flatten()

        is_metro_loc = df_loc["is_metropolitan"].values == 1
        metro_demand = float(np.sum(demand_pred[is_metro_loc]))
        village_demand = float(np.sum(demand_pred[~is_metro_loc]))

        # =========================================================
        # 1. REVENUE STREAMS (INR)
        # =========================================================
        # A. On-Demand Servicing Revenue
        avg_ticket_metro = (
            business_inputs.base_metro_ticket_cooling * cool_mult +
            business_inputs.base_metro_ticket_heating * heat_mult
        )
        avg_ticket_village = (
            business_inputs.base_village_ticket_cooling * cool_mult +
            business_inputs.base_village_ticket_heating * heat_mult
        )
        ondemand_service_revenue = metro_demand * avg_ticket_metro + village_demand * avg_ticket_village

        # B. AMC Recurring Subscription Revenue
        n_residents = len(df_res)
        amc_subscribers = int(n_residents * business_inputs.amc_coverage_ratio)
        amc_recurring_revenue = amc_subscribers * business_inputs.amc_subscription_monthly_fee

        # C. Sleeper Customer Reactivation & Cross-Sell Revenue
        detected_sleepers = int(np.sum(sleeper_prob > 0.55))
        reactivated_customers = int(detected_sleepers * business_inputs.sleeper_reactivation_rate)
        cross_sell_unit_rev = (
            business_inputs.cross_sell_heating_rev if heat_mult > cool_mult
            else business_inputs.cross_sell_cooling_rev
        )
        retention_revenue = reactivated_customers * cross_sell_unit_rev

        total_revenue = ondemand_service_revenue + amc_recurring_revenue + retention_revenue

        # =========================================================
        # 2. EXPENDITURE STREAMS (INR)
        # =========================================================
        # A. Preventive Maintenance Cost (Proactive inspection of GNN-flagged devices)
        high_risk_devices = int(np.sum(failure_prob > 0.50))
        preventive_maint_cost = high_risk_devices * business_inputs.preventive_inspection_cost

        # Breakdown savings vs catastrophic repairs
        avoided_breakdown_savings = (
            high_risk_devices *
            (business_inputs.emergency_breakdown_cost - business_inputs.preventive_inspection_cost) *
            business_inputs.avoided_breakdown_efficiency
        )
        residual_emergency_cost = max(
            5, int(len(df_dev) * business_inputs.unavoidable_emergency_rate)
        ) * business_inputs.emergency_breakdown_cost

        # B. Inventory Holding & Stockout Penalty Costs
        unit_holding = df_part["holding_cost_monthly"].values
        inventory_holding_cost = float(np.sum(np.clip(inventory_pred, 0, None) * unit_holding))
        
        target_inv = data["part"].y_inventory.cpu().numpy().flatten()
        stockout_penalty = float(
            np.sum(np.abs(np.clip(inventory_pred, 0, None) - target_inv) * business_inputs.stockout_penalty_per_unit)
        )

        # C. Marketing CAC Cost for Sleeper Retention Campaigns
        campaign_cost = detected_sleepers * business_inputs.marketing_campaign_cost_per_user

        total_expenditure = (
            preventive_maint_cost +
            residual_emergency_cost +
            inventory_holding_cost +
            stockout_penalty +
            campaign_cost
        )

        # Baseline: Unoptimized reactive operation (no GNN, high breakdown rates, stockouts)
        baseline_unoptimized_expenditure = (
            total_expenditure +
            avoided_breakdown_savings +
            (stockout_penalty * 2.4)
        )

        # =========================================================
        # 3. EXECUTIVE KPI METRICS
        # =========================================================
        net_profit = total_revenue - total_expenditure
        profit_margin_pct = (net_profit / max(total_revenue, 1.0)) * 100.0
        cost_efficiency_ratio = total_revenue / max(total_expenditure, 1.0)
        cost_savings_pct = (
            (baseline_unoptimized_expenditure - total_expenditure) /
            max(baseline_unoptimized_expenditure, 1.0)
        ) * 100.0

        return {
            "month": month,
            "season_name": season_name,
            "cooling_multiplier": round(cool_mult, 2),
            "heating_multiplier": round(heat_mult, 2),
            "metro_demand_units": round(metro_demand, 1),
            "village_demand_units": round(village_demand, 1),
            "high_risk_devices_flagged": high_risk_devices,
            "sleeper_customers_identified": detected_sleepers,
            "sleepers_reactivated": reactivated_customers,
            "optimal_spare_parts_units": int(np.sum(np.clip(inventory_pred, 0, None))),
            # Revenue breakdown
            "ondemand_service_revenue": round(ondemand_service_revenue, 2),
            "amc_recurring_revenue": round(amc_recurring_revenue, 2),
            "retention_revenue": round(retention_revenue, 2),
            "revenue_inr": round(total_revenue, 2),
            # Expenditure breakdown
            "preventive_maint_cost": round(preventive_maint_cost, 2),
            "residual_emergency_cost": round(residual_emergency_cost, 2),
            "inventory_holding_cost": round(inventory_holding_cost, 2),
            "stockout_penalty": round(stockout_penalty, 2),
            "campaign_cost": round(campaign_cost, 2),
            "expenditure_inr": round(total_expenditure, 2),
            "baseline_expenditure_inr": round(baseline_unoptimized_expenditure, 2),
            # Executive KPIs
            "net_profit_inr": round(net_profit, 2),
            "profit_margin_pct": round(profit_margin_pct, 2),
            "cost_efficiency_ratio": round(cost_efficiency_ratio, 3),
            "cost_efficiency_gain_pct": round(cost_savings_pct, 2)
        }
