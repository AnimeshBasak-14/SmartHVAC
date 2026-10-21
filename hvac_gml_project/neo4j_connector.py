"""
neo4j_connector.py
Bi-directional Neo4j Graph Database Connector for the HVAC Heterogeneous Graph.
"""
import logging
from config import Config

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

class Neo4jGraphManager:
    def __init__(self, uri=Config.NEO4J_URI, user=Config.NEO4J_USER, password=Config.NEO4J_PASSWORD):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        if NEO4J_AVAILABLE and Config.PUSH_TO_NEO4J:
            try:
                self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
                self.driver.verify_connectivity()
                logging.info(f"Connected to Neo4j at {self.uri}")
            except Exception as e:
                logging.warning(f"Neo4j server not reachable at {self.uri} ({e}). Running in local PyG mode.")
                self.driver = None

    def close(self):
        if self.driver is not None:
            try:
                self.driver.close()
            except Exception:
                pass

    def sync_graph_tables(self, df_res, df_loc, df_dev, df_part, e_res_loc, e_res_dev, e_dev_part, active_month=5):
        if self.driver is None:
            return False

        try:
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")

                session.run("""
                    UNWIND $rows AS row
                    MERGE (l:Location {location_id: row.location_id})
                    SET l.name = row.name,
                        l.location_type = row.location_type,
                        l.is_metropolitan = row.is_metropolitan,
                        l.base_temp_c = row.base_temp_c,
                        l.grid_stability = row.grid_stability,
                        l.active_month = $month
                """, rows=df_loc.to_dict("records"), month=active_month)

                session.run("""
                    UNWIND $rows AS row
                    MERGE (d:Device {device_id: row.device_id})
                    SET d.sku_name = row.sku_name,
                        d.category = row.category,
                        d.is_cooling_hvac = row.is_cooling_hvac,
                        d.unit_price_inr = row.unit_price_inr,
                        d.rated_power_kw = row.rated_power_kw,
                        d.base_failure_risk = row.base_failure_risk
                """, rows=df_dev.to_dict("records"))

                session.run("""
                    UNWIND $rows AS row
                    MERGE (p:InventoryPart {part_id: row.part_id})
                    SET p.part_name = row.part_name,
                        p.for_cooling_hvac = row.for_cooling_hvac,
                        p.unit_cost_inr = row.unit_cost_inr,
                        p.current_stock = row.current_stock
                """, rows=df_part.to_dict("records"))

                session.run("""
                    UNWIND $rows AS row
                    MERGE (r:Resident {resident_id: row.resident_id})
                    SET r.is_metropolitan = row.is_metropolitan,
                        r.customer_app_active = row.customer_app_active,
                        r.base_app_engagement = row.base_app_engagement,
                        r.tenure_months = row.tenure_months
                """, rows=df_res.to_dict("records"))

                session.run("""
                    UNWIND $rows AS row
                    MATCH (r:Resident {resident_id: row.resident_id})
                    MATCH (l:Location {location_id: row.location_id})
                    MERGE (r)-[:LOCATED_IN {month: $month}]->(l)
                """, rows=e_res_loc.to_dict("records"), month=active_month)

                session.run("""
                    UNWIND $rows AS row
                    MATCH (r:Resident {resident_id: row.resident_id})
                    MATCH (d:Device {device_id: row.device_id})
                    MERGE (r)-[:OWNS {month: $month}]->(d)
                """, rows=e_res_dev.to_dict("records"), month=active_month)

                session.run("""
                    UNWIND $rows AS row
                    MATCH (d:Device {device_id: row.device_id})
                    MATCH (p:InventoryPart {part_id: row.part_id})
                    MERGE (d)-[:REQUIRES_PART]->(p)
                """, rows=e_dev_part.to_dict("records"))

            return True
        except Exception as e:
            logging.warning(f"Neo4j sync failed: {e}. Falling back to local PyG mode.")
            return False

    @staticmethod
    def get_cypher_visualization_queries():
        return {
            "1_full_hetero_subgraph": (
                "MATCH (l:Location)<-[:LOCATED_IN]-(r:Resident)-[:OWNS]->(d:Device)-[:REQUIRES_PART]->(p:InventoryPart) "
                "RETURN l, r, d, p LIMIT 150"
            ),
            "2_sleeper_churn_cross_sell": (
                "MATCH (r:Resident)-[:OWNS]->(d:Device {category: 'Cooling_HVAC'}) "
                "WHERE r.base_app_engagement < 0.45 "
                "MATCH (h:Device {category: 'Heating_Induction_Kettle'}) "
                "RETURN r, d, h LIMIT 50"
            ),
            "3_high_risk_hvac_parts": (
                "MATCH (l:Location {location_type: 'Metropolitan'})<-[:LOCATED_IN]-(r:Resident)-[:OWNS]->(d:Device)-[:REQUIRES_PART]->(p:InventoryPart) "
                "WHERE d.base_failure_risk > 0.12 "
                "RETURN l, d, p LIMIT 80"
            )
        }

