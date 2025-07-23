# This example demonstrates a simple AWS EC2 instance deployment
# and showcases Terraform's declarative infrastructure management.

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0" # Use a suitable version constraint
    }
  }


  required_version = ">= 1.2.0" # Specify a minimum Terraform version
}

provider "aws" {
  region = "us-west-2" # Replace with your desired AWS region
}


resource "aws_instance" "example" {
  ami           = data.aws_ami.amazon_linux_2.id # Use latest Amazon Linux 2 AMI
  instance_type = "t2.micro"

  # Security group configuration - inline creation
  vpc_security_group_ids = ["sg-xxxxxx"]


  tags = {
    Name = "example-instance"
  }
}


# Data source to fetch the latest Amazon Linux 2 AMI ID
data "aws_ami" "amazon_linux_2" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["amzn2-ami-hvm-*-x86_64-gp2"]
  }
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# Security Group configuration
resource "aws_security_group" "allow_ssh" {
 name        = "allow_ssh"
 description = "Allow SSH inbound traffic"

 ingress {
   description      = "SSH from anywhere"
   from_port        = 22
   to_port          = 22
   protocol        = "tcp"
   cidr_blocks      = ["0.0.0.0/0"] # Example: Open to the world. Restrict in production.
 }

 egress {
   from_port        = 0
   to_port          = 0
   protocol        = "-1"
   cidr_blocks      = ["0.0.0.0/0"]
 }

 tags = {
   Name = "allow_ssh"
 }

}



# Output the public IP of the instance
output "public_ip" {
  value = aws_instance.example.public_ip
}