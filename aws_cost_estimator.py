"""
aws_cost_estimator.py
Exports: estimate_costs() -> None
"""

import boto3
import json

# ---------- helpers ----------
def get_ec2_price(instance_type="t2.micro",
                  region="US East (N. Virginia)") -> float:
    pricing = boto3.client("pricing", region_name="us-east-1")
    resp = pricing.get_products(
        ServiceCode="AmazonEC2",
        Filters=[
            {"Type": "TERM_MATCH", "Field": "instanceType", "Value": instance_type},
            {"Type": "TERM_MATCH", "Field": "location", "Value": region},
            {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"},
            {"Type": "TERM_MATCH", "Field": "preInstalledSw", "Value": "NA"},
            {"Type": "TERM_MATCH", "Field": "capacitystatus", "Value": "Used"},
            {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "Shared"},
        ],
        MaxResults=1,
    )
    product = json.loads(resp["PriceList"][0])
    dims = list(product["terms"]["OnDemand"].values())[0]["priceDimensions"]
    return float(list(dims.values())[0]["pricePerUnit"]["USD"])


def get_rds_price(instance_class="db.t3.micro",
                  region="US East (N. Virginia)") -> float:
    pricing = boto3.client("pricing", region_name="us-east-1")
    resp = pricing.get_products(
        ServiceCode="AmazonRDS",
        Filters=[
            {"Type": "TERM_MATCH", "Field": "instanceType", "Value": instance_class},
            {"Type": "TERM_MATCH", "Field": "location", "Value": region},
            {"Type": "TERM_MATCH", "Field": "databaseEngine", "Value": "MySQL"},
            {"Type": "TERM_MATCH", "Field": "deploymentOption", "Value": "Single-AZ"},
        ],
        MaxResults=1,
    )
    product = json.loads(resp["PriceList"][0])
    dims = list(product["terms"]["OnDemand"].values())[0]["priceDimensions"]
    return float(list(dims.values())[0]["pricePerUnit"]["USD"])


# ---------- exported API ----------
def estimate_costs() -> None:
    """
    Prints a quick cost estimate for one EC2 (t2.micro) and one RDS (db.t3.micro)
    running in US-EAST-1.
    """
    ec2_price = get_ec2_price()
    rds_price = get_rds_price()

    print("\nEstimated Cost Breakdown (Per Hour):")
    print(f"  - EC2 (t2.micro): ${ec2_price:.4f}/hour")
    print(f"  - RDS (db.t3.micro, MySQL): ${rds_price:.4f}/hour")

    monthly = (ec2_price + rds_price) * 24 * 30
    print(f"\nTotal Estimated Monthly Cost: ${monthly:.2f}")


# ---------- CLI ----------
if __name__ == "__main__":
    estimate_costs()