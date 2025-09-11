# ⚠️  DEMO CREDENTIALS - DO NOT USE IN PRODUCTION!
# Replace all demo credentials with real values before deploying
# Use a secrets manager for production secrets

provider "aws" {
  region = var.region
}


variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}


terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "us-east-1"
}

resource "aws_instance" "example" {
  ami           = data.aws_ami.amazon_linux_2_latest.id
  instance_type = "t3.micro"

  # Enable SSM Session Manager access
  iam_instance_profile = aws_iam_instance_profile.ssm_instance_profile.name
}

resource "aws_iam_instance_profile" "ssm_instance_profile" {
  name = "ssm_instance_profile"
  role = aws_iam_role.ssm_role.name
}

resource "aws_iam_role" "ssm_role" {
  name = "ssm_role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action = "sts:AssumeRole",
        Effect = "Allow",
        Sid    = "",
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      },
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ssm_policy_attachment" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
  role       = aws_iam_role.ssm_role.name
}


data "aws_ami" "amazon_linux_2_latest" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["amzn2-ami-hvm-*-x86_64-gp2"]
  }
}

output "public_ip" {
  value = aws_instance.example.public_ip
}