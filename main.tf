resource "aws_instance" "example" {
  ami           = data.aws_ami.amazon_linux_2_latest.id
  instance_type = "t2.micro"
  subnet_id = aws_subnet.example.id
  vpc_security_group_ids = ["sg-xxxxxx"]

  user_data = <<-EOF
#!/bin/bash
echo "Hello from Terraform!" > /tmp/terraform.txt
EOF

  tags = {
    Name = "example-ec2-instance"
  }
}


data "aws_ami" "amazon_linux_2_latest" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["amzn2-ami-hvm-*-x86_64-gp2"]
  }
}


resource "aws_vpc" "example" {
  cidr_block = "10.0.0.0/16"

  tags = {
    Name = "example-vpc"
  }
}

resource "aws_subnet" "example" {
  vpc_id            = aws_vpc.example.id
  cidr_block        = "10.0.1.0/24"
  availability_zone = data.aws_availability_zones.available.names[0]

  tags = {
    Name = "example-subnet"
  }
}

data "aws_availability_zones" "available" {}

resource "aws_security_group" "example" {
  name        = "allow_ssh"
  description = "Allow SSH inbound traffic"
  vpc_id      = aws_vpc.example.id

 ingress {
    description      = "SSH from Anywhere"
    from_port        = 22
    to_port          = 22
    protocol         = "tcp"
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }

  egress {
    from_port        = 0
    to_port          = 0
    protocol         = "-1"
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }

  tags = {
    Name = "allow_ssh"
  }
}



resource "aws_internet_gateway" "example" {
 vpc_id = aws_vpc.example.id

  tags = {
    Name = "example-internet-gateway"
  }
}

resource "aws_route_table" "example" {
  vpc_id = aws_vpc.example.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.example.id
  }

  tags = {
    Name = "example-route-table"
  }
}


resource "aws_route_table_association" "example" {
  subnet_id      = aws_subnet.example.id
  route_table_id = aws_route_table.example.id
}