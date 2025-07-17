import boto3
import json

def get_ec2_price(instance_type="t2.micro", region="US East (N. Virginia)"):
    pricing = boto3.client("pricing", region_name="us-east-1")  # Only supported region

    response = pricing.get_products(
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

    product = json.loads(response["PriceList"][0])
    price_dimensions = list(product["terms"]["OnDemand"].values())[0]["priceDimensions"]
    price_per_hour = list(price_dimensions.values())[0]["pricePerUnit"]["USD"]
    return float(price_per_hour)


def get_rds_price(instance_class="db.t3.micro", region="US East (N. Virginia)"):
    pricing = boto3.client("pricing", region_name="us-east-1")

    response = pricing.get_products(
        ServiceCode="AmazonRDS",
        Filters=[
            {"Type": "TERM_MATCH", "Field": "instanceType", "Value": instance_class},
            {"Type": "TERM_MATCH", "Field": "location", "Value": region},
            {"Type": "TERM_MATCH", "Field": "databaseEngine", "Value": "MySQL"},
            {"Type": "TERM_MATCH", "Field": "deploymentOption", "Value": "Single-AZ"},
        ],
        MaxResults=1,
    )

    product = json.loads(response["PriceList"][0])
    price_dimensions = list(product["terms"]["OnDemand"].values())[0]["priceDimensions"]
    price_per_hour = list(price_dimensions.values())[0]["pricePerUnit"]["USD"]
    return float(price_per_hour)


if __name__ == "__main__":
    ec2_price = get_ec2_price()
    rds_price = get_rds_price()

    print(f"\n💰 Estimated Cost Breakdown (Per Hour):")
    print(f"  - EC2 (t2.micro): ${ec2_price}/hour")
    print(f"  - RDS (db.t3.micro, MySQL): ${rds_price}/hour")

    monthly_cost = (ec2_price + rds_price) * 24 * 30
    print(f"\n📦 Total Estimated Monthly Cost: ${monthly_cost:.2f}")
