resource "aws_instance" "example" {
  count                  = 3
  ami                    = data.aws_ami.latest_amazon_linux.id
  instance_type          = "t2.medium"
  availability_zone = data.aws_availability_zones.available.names[count.index % length(data.aws_availability_zones.available.names)] # distribute across AZs
  key_name               = aws_key_pair.example.key_name
  vpc_security_group_ids = ["sg-xxxxxx"]


  root_block_device {
    volume_type = "gp3" # gp3 is recommended for most workloads
    volume_size = 50
    delete_on_termination = true
  }

 tags = {
    Name = "example-instance-${count.index}"
 }

 lifecycle {
    create_before_destroy = true
 }
}

resource "aws_security_group" "example" {
 name        = "allow_ssh"
 description = "Allow SSH inbound traffic"
 vpc_id = data.aws_vpc.default.id

 ingress {
    description = "SSH from anywhere"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] # Open to the world for testing purposes - restrict in production
 }

 egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
 }
}

resource "aws_key_pair" "example" {
  key_name = "terraform-key"
  public_key = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQD..." # Replace with your actual public key path.  Alternatively, generate a key pair within Terraform.
}

data "aws_ami" "latest_amazon_linux" {
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


data "aws_availability_zones" "available" {}

data "aws_vpc" "default" {
 default = true
}