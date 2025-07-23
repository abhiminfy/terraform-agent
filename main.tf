# Configure the AWS Provider
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }


}

provider "aws" {
  region = "us-west-2" # Replace with your desired region
}

# Create a VPC
resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"

  tags = {
    Name = "main-vpc"
  }
}

# Create an internet gateway
resource "aws_internet_gateway" "gw" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "main-igw"
  }
}

# Create a public subnet
resource "aws_subnet" "public" {
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.1.0/24"
  availability_zone = data.aws_availability_zones.available.names[0] # Use the first available AZ

  tags = {
    Name = "public-subnet"
  }
}


# Create a route table
resource "aws_route_table" "public_route_table" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block        = "0.0.0.0/0"
    gateway_id        = aws_internet_gateway.gw.id
  }

  tags = {
    Name = "public-route-table"
  }
}

# Associate the route table with the subnet
resource "aws_route_table_association" "public_subnet_association" {
 subnet_id      = aws_subnet.public.id
 route_table_id = aws_route_table.public_route_table.id
}



# Get available availability zones
data "aws_availability_zones" "available" {}

# Create a security group
resource "aws_security_group" "allow_ssh" {
  name        = "allow_ssh"
  description = "Allow SSH inbound traffic"
  vpc_id      = aws_vpc.main.id

 ingress {
    description      = "SSH from Anywhere"
    from_port        = 22
    to_port          = 22
    protocol         = "tcp"
    cidr_blocks      = ["0.0.0.0/0"] #  Restrict this in production
  }

  egress {
    from_port       = 0
    to_port         = 0
    protocol        = "-1"
    cidr_blocks     = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }

  tags = {
    Name = "allow_ssh"
  }
}



# Create an EC2 instance
resource "aws_instance" "example" {
  ami           = data.aws_ami.amazon_linux_2.id # Use the latest Amazon Linux 2 AMI
  instance_type = "t2.micro"
  subnet_id     = aws_subnet.public.id
  vpc_security_group_ids = ["sg-xxxxxx"]
  key_name = "terraform-key" # Replace with the name of your existing key pair

  tags = {
    Name = "example-instance"
  }
 user_data = <<-EOT
#!/bin/bash
yum update -y
EOT

}



# Retrieve the latest Amazon Linux 2 AMI
data "aws_ami" "amazon_linux_2" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["amzn2-ami-hvm*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

  filter {
    name = "architecture"
 values = ["x86_64"]
  }


}