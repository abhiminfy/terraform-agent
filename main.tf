# VPC and Subnets

resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "main-vpc"
  }
}

resource "aws_subnet" "public_1" {
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.1.0/24"
  availability_zone = data.aws_availability_zones.available.names[0]

  tags = {
    Name = "public-subnet-1"
  }

}

resource "aws_subnet" "public_2" {
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.2.0/24"
  availability_zone = data.aws_availability_zones.available.names[1]

  tags = {
    Name = "public-subnet-2"
  }
}


resource "aws_subnet" "private_1" {
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.10.0/24"
  availability_zone = data.aws_availability_zones.available.names[0]

  tags = {
    Name = "private-subnet-1"
  }
}


resource "aws_subnet" "private_2" {
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.20.0/24"
  availability_zone = data.aws_availability_zones.available.names[1]

  tags = {
    Name = "private-subnet-2"
  }
}


data "aws_availability_zones" "available" {}



# Internet Gateway and Route Table

resource "aws_internet_gateway" "gw" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "main-igw"
  }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.gw.id
  }

  tags = {
    Name = "public-route-table"
  }
}



resource "aws_route_table_association" "public_1" {
  subnet_id      = aws_subnet.public_1.id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "public_2" {
  subnet_id      = aws_subnet.public_2.id
  route_table_id = aws_route_table.public.id
}



# Database (RDS)

resource "aws_db_subnet_group" "default" {
  name       = "main-db-subnet-group"
  subnet_ids = [aws_subnet.private_1.id, aws_subnet.private_2.id]

  tags = {
    Name = "main-rds-subnet-group"
  }
}

resource "aws_db_instance" "default" {
  identifier = "terraform-mysql-db"
  allocated_storage      = 20  # Adjust as needed
  storage_type           = "gp2" # Or provisioned IOPS for better performance
  engine                = "mysql" # or postgres
  engine_version        = "8.0"  # Adjust as needed
  instance_class         = "db.t3.micro" # Choose an instance class suitable for your workload
  username = "adminuser"  # Replace with your desired username
  password = "MyS3cur3P@ssw0rd!" # Replace with a strong password
  db_subnet_group_name  = aws_db_subnet_group.default.name
  skip_final_snapshot    = true
  vpc_security_group_ids = ["sg-xxxxxx"]

  tags = {
    Name = "main-rds-instance"
  }

}




# Security Groups

resource "aws_security_group" "web_sg" {
  name        = "web-sg"
  description = "Allow HTTP access"
  vpc_id      = aws_vpc.main.id

 ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] #  Restrict this if possible after testing.
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] # Restrict this if possible after testing.
  }


  egress {
    from_port        = 0
    to_port          = 0
    protocol        = "-1"
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }


  tags = {
    Name = "allow_http"
  }
}



resource "aws_security_group" "rds_sg" {
  name        = "rds-sg"
  description = "Allow inbound traffic from web servers"
  vpc_id      = aws_vpc.main.id


  ingress {
 from_port   = 3306 # MySQL default port, adjust for other databases
    to_port     = 3306
    protocol    = "tcp"
    security_groups = [aws_security_group.web_sg.id]
  }



  egress {
    from_port        = 0
    to_port          = 0
    protocol        = "-1"
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }

  tags = {
    Name = "allow_web_to_rds"
  }

}









# EC2 Instance (Web Server)

resource "aws_instance" "web" {
  ami                         = data.aws_ami.amazon_linux_2_latest.id  # Replace with your preferred AMI
  instance_type               = "t2.medium"  # Or another suitable instance type, t3.medium, t3.large, etc. based on your needs.
  count = 2 # 2 instances for high availability
  subnet_id                   = aws_subnet.public_1.id # Place one instance in each public subnet
  vpc_security_group_ids = ["sg-xxxxxx"]
  associate_public_ip_address = true

  user_data = <<-EOF
#!/bin/bash
sudo yum update -y
sudo yum install httpd -y
sudo systemctl start httpd
sudo systemctl enable httpd
echo "Hello World from $(hostname -f)" > /var/www/html/index.html
  EOF



  tags = {
    Name = "web-server"
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




# Load Balancer 

resource "aws_lb" "example" {
  internal           = false
  load_balancer_type = "application" # Use "application" for HTTP/HTTPS
  name               = "example-lb"

  security_groups = [aws_security_group.web_sg.id]


 subnets = [aws_subnet.public_1.id,aws_subnet.public_2.id]

}

resource "aws_lb_target_group" "example" {
  name        = "example-target-group"
  port        = 80
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "instance" # Target EC2 instances
 health_check {
    path                = "/"  # Path to check
    protocol            = "HTTP"
    matcher             = "200" # Expected status code
    interval            = 30 # Check interval in seconds
    timeout             = 5 # Timeout in seconds
    healthy_threshold   = 2 # Number of consecutive successes 
    unhealthy_threshold = 2 # Number of consecutive failures
  }
}

resource "aws_lb_listener" "example" {
  load_balancer_arn = aws_lb.example.arn
  port              = "80" # Listener port
  protocol          = "HTTP"  # Listener protocol
 default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.example.arn
  }
}




resource "aws_lb_target_group_attachment" "example" {
  count            = length(aws_instance.web)
  target_group_arn = aws_lb_target_group.example.arn
  target_id        = aws_instance.web[count.index].id
  port             = 80
}